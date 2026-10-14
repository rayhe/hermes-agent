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


@pytest.mark.asyncio
async def test_authenticated_history_renders_image_after_adapter_restart(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.session_persistence import _db_flush_row
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="api_server")
    content = [{"type": "text", "text": "caption"}, {"type": "image_url", "image_url": {"url": png_url()}}]
    db.append_message("s", **_db_flush_row(SimpleNamespace(_session_db=db), {"role": "user", "content": content}, True))
    db.append_message("s", "assistant", "done")
    db.close()
    db = SessionDB(path)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "local-test-only"}))
    adapter._session_db = db
    app = web.Application()
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    async with TestClient(TestServer(app)) as client:
        denied = await client.get("/api/sessions/s/messages")
        assert denied.status == 401
        response = await client.get("/api/sessions/s/messages", headers={"Authorization": "Bearer local-test-only"})
        assert response.status == 200, await response.text()
        row = (await response.json())["data"][0]
        assert row["content"] == content
        assert "api_content" not in row
    assert (await adapter._conversation_history_for_session("s"))[0]["content"] == content
    db.close()
