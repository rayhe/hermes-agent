"""Secondary ingress contracts: syntax, structured runs, scoped intact tool handles."""
import base64
import copy
import json
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _normalize_multimodal_content


def png_url():
    stream = BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.mark.parametrize("header", ["data:image/png;base64junk", "data:image/png;;base64", "data:image/png;base64;extra"])
def test_malformed_data_url_header_is_rejected(header):
    url = header + "," + png_url().partition(",")[2]
    with pytest.raises(ValueError, match="invalid_image_url"):
        _normalize_multimodal_content([{"type": "input_image", "image_url": url}])


@pytest.mark.asyncio
@pytest.mark.parametrize("history_source", ["input", "conversation_history"])
async def test_runs_http_normalizes_current_image_and_preserves_image_history(history_source, tmp_path, monkeypatch):
    import asyncio
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    captured = []
    agent = SimpleNamespace(_session_db=db, session_id="s", run_conversation=lambda **kw: captured.append(kw) or {"final_response": "ok"})
    adapter._create_agent = lambda **kw: agent
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    parts = [{"type": "input_image", "image_url": png_url(), "detail": "high"}]
    body = {"input": [{"role": "user", "content": parts}, {"role": "assistant", "content": "red"}, {"role": "user", "content": parts}]}
    if history_source == "conversation_history":
        body["conversation_history"] = body["input"][:-1]
        body["input"] = body["input"][-1:]
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/runs", json=body)
        assert response.status == 202, await response.text()
        await asyncio.gather(*list(adapter._active_run_tasks.values()))
    assert len(captured) == 1
    current = captured[0]["user_message"]
    assert "[Image attached at:" in str(current)
    assert current[-1]["image_url"] == {"url": png_url(), "detail": "high"}
    assert captured[0]["conversation_history"][0]["content"] == [{"type": "image_url", "image_url": {"url": png_url(), "detail": "high"}}]
    assert captured[0]["conversation_history"][1]["content"] == "red"
    assert body["input"][0]["content"] == parts  # history never receives current-turn hints
    db.close()


@pytest.mark.asyncio
async def test_intact_tool_handles_are_profile_scoped_a_b_a_and_idempotent(tmp_path, monkeypatch):
    from agent import secret_scope
    from gateway.platforms.api_server import _api_request_profile
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tools import vision_tools
    from tools.image_source import ResolveContext, resolve_image_source
    from tools.registry import registry
    from tools.terminal_scope import terminal_env
    from tools.vision_tools_history_budget import native_turn_images

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = {name: tmp_path / name for name in ("a", "b")}
    for name, home in homes.items():
        home.mkdir()
        (home / ".env").write_text(f"IMAGE_SCOPE_PROBE={name}\n", encoding="utf-8")
        (home / "config.yaml").write_text("terminal:\n  backend: docker\n", encoding="utf-8")
        (home / "cache" / "images").mkdir(parents=True)
        (home / "cache" / "images" / "existing").write_bytes(b"existing cached pixels")
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    captures = []
    databases = []
    history = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": png_url()}}]}, {"role": "assistant", "content": "red"}]
    original_history = copy.deepcopy(history)

    class ProbeAgent(SimpleNamespace):
        def run_conversation(self, user_message, **kw):
            assert get_hermes_home() == self.home
            assert secret_scope.get_secret("IMAGE_SCOPE_PROBE") == self.home.name
            assert terminal_env("TERMINAL_ENV") == "docker"
            captures.append((self.home, user_message))
            return {"final_response": "ok", "messages": []}

    def create(**kw):
        home = get_hermes_home()
        db = SessionDB(home / "state.db")
        databases.append(db)
        return ProbeAgent(_session_db=db, home=home, session_id=kw["session_id"])

    adapter._create_agent = create
    secret_scope.set_multiplex_active(True)
    try:
        for name in ("a", "b", "a"):
            token = _api_request_profile.set(name)
            try:
                current = [{"type": "image_url", "image_url": {"url": png_url()}}]
                await adapter._run_agent(current, history, session_id="s")
                await adapter._run_agent(current, history, session_id="s")  # same admitted turn is idempotent
                home, content = captures[-1]
                text = "\n".join(p.get("text", "") for p in content)
                handles = re.findall(r"\[Image attached at: (.+?)\]", text)
                assert len(handles) == 1 and len(handles[0]) < 300
                assert Path(handles[0]).parent == home / "images"
                with adapter._profile_scope(name):
                    resolved = await resolve_image_source(handles[0], ResolveContext())
                    assert resolved.data == base64.b64decode(png_url().partition(",")[2])
                    with native_turn_images(content):
                        result = await registry.get_entry("vision_analyze").handler({"image_url": handles[0], "question": "inspect"})
                    assert json.loads(result)["already_in_context"] is True
                assert history == original_history
                assert len(list((home / "images").iterdir())) == 1
                assert list((home / "cache" / "images").iterdir()) == [home / "cache" / "images" / "existing"]
                assert (home / "cache" / "images" / "existing").read_bytes() == b"existing cached pixels"
            finally:
                _api_request_profile.reset(token)
        assert str(homes["b"]) not in str(captures[0])
        assert str(homes["a"]) not in str(captures[2])
    finally:
        secret_scope.set_multiplex_active(False)
        for db in databases:
            db.close()
