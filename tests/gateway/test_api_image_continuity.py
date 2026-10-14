"""Real API ingress and durable image continuity regressions."""
import base64
from io import BytesIO

import pytest
from PIL import Image
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def png_url():
    stream = BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def oversized_animation_url():
    from tools.vision_tools_image_prep import _VISION_MAX_VALIDATED_FRAME_COUNT
    stream = BytesIO()
    frames = [Image.new("RGB", (2, 2), (i, 0, 0)) for i in range(_VISION_MAX_VALIDATED_FRAME_COUNT + 1)]
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=10)
    return "data:image/gif;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses", "runs"])
@pytest.mark.parametrize("image", ["data:image/jpeg;base64,[omitted]", "data:image/png;base64,bm90IGFuIGltYWdl", "data:image/png;base64,iVBORw0KGgo=", "data:image/png,abc", oversized_animation_url()], ids=["omitted", "nonimage", "truncated", "not-base64", "too-many-frames"])
async def test_bad_image_is_400_before_agent(endpoint, image):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    calls = []
    def no_create(**kw):
        calls.append(kw)
        raise AssertionError("invalid images must not create an agent")
    adapter._create_agent = no_create
    async def no_agent(**kw):
        calls.append(kw)
        return {"final_response": "not reached", "messages": []}, {}
    adapter._run_agent = no_agent
    content = [{"type": "input_image", "image_url": image}]
    body = {"model": "hermes-agent", "messages" if endpoint == "chat/completions" else "input": [{"role": "user", "content": content}]}
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/" + endpoint, json=body)
        assert response.status == 400, await response.text()
        error = (await response.json())["error"]
        assert error["code"] == "invalid_image_url"
        assert "image" in error["message"].lower()
        assert "[omitted]" not in error["message"]
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["chat/completions", "responses"])
async def test_api_handoff_has_resolvable_small_tool_handle(endpoint, tmp_path, monkeypatch):
    import re
    from pathlib import Path
    from types import SimpleNamespace
    from agent.session_persistence import _db_flush_row
    from hermes_state import SessionDB
    from tools.image_source import resolve_image_source, ResolveContext
    from tools.vision_tools_history_budget import native_turn_images, native_turn_duplicate

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "local-test-only"}))
    adapter._session_db = db
    captured = []
    histories = []
    class ProbeAgent(SimpleNamespace):
        def run_conversation(self, user_message, conversation_history, **kwargs):
            captured.append(user_message)
            histories.append(conversation_history)
            if not db.get_session(self.session_id):
                db.create_session(self.session_id, source="api_server")
            db.append_message(self.session_id, **_db_flush_row(self, {"role": "user", "content": user_message}, True))
            db.append_message(self.session_id, "assistant", "inspected")
            return {"final_response": "inspected", "messages": conversation_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "inspected"}]}
    adapter._create_agent = lambda **kwargs: ProbeAgent(_session_db=db, session_id=kwargs["session_id"])
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    url = png_url()
    body = {"model": "hermes-agent", "messages" if endpoint == "chat/completions" else "input": [{"role": "user", "content": [{"type": "text", "text": "inspect"}, {"type": "input_image", "image_url": url, "detail": "high"}]}]}
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/v1/" + endpoint, json=body, headers={"Authorization": "Bearer local-test-only"})
        assert response.status == 200, await response.text()
        first_response = await response.json()
        headers = {"Authorization": "Bearer local-test-only"}
        if endpoint == "responses":
            followup = {"input": "inspect again", "previous_response_id": first_response["id"]}
        else:
            headers["X-Hermes-Session-Id"] = response.headers["X-Hermes-Session-Id"]
            followup = {"messages": [{"role": "user", "content": "inspect again"}]}
        second = await client.post("/v1/" + endpoint, json=followup, headers=headers)
        assert second.status == 200, await second.text()
        assert histories[1][0]["content"] == captured[0]
        assert [m["role"] for m in histories[1]] == ["user", "assistant"]
    content = captured[0]
    text = "\n".join(p.get("text", "") for p in content)
    handles = re.findall(r"\[Image attached at: (.+?)\]", text)
    assert len(handles) == 1, "incoming API image needs an intact vision_analyze path, not copied base64"
    assert len(handles[0]) < 300 and "base64" not in handles[0]
    resolved = await resolve_image_source(handles[0], ResolveContext())
    assert resolved.data == base64.b64decode(url.partition(",")[2])
    assert content[1]["image_url"] == {"url": url, "detail": "high"}
    with native_turn_images(content):
        assert '"already_in_context": true' in (native_turn_duplicate(handles[0], None) or "")
    db.close()


def test_structured_run_uses_the_same_image_handle_handoff(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway.platforms import api_server
    from gateway.platforms.api_server_runs import _run_agent_sync
    from hermes_state import SessionDB
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    captured = []
    agent = SimpleNamespace(_session_db=db, session_id="s", run_conversation=lambda **kw: captured.append(kw) or {"final_response": "ok"})
    run = SimpleNamespace(session_id="s", run_id="r", request_profile=None, approval_session_key="approval-r",
                          browser_control_principal=None, browser_control_transport_family=None,
                          session_history_delivery=False, agent_kwargs={"room_dispatch": None},
                          turn_author=None, user_message=[{"type": "image_url", "image_url": {"url": png_url()}}],
                          conversation_history=[], declared_selected=False)
    _run_agent_sync(adapter, run, agent, lambda data: None, _api_server=api_server)
    assert "[Image attached at:" in str(captured[0]["user_message"])
    db.close()
