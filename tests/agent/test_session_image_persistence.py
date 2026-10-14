"""Multimodal persistence keeps model bytes outside the searchable projection."""
import base64
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from agent.session_persistence import _db_flush_row
from hermes_state import SessionDB


def parts():
    stream = BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    return [{"type": "text", "text": "inspect the red pixels"}, {"type": "image_url", "image_url": {"url": url, "detail": "high"}}]


def test_user_image_replays_exactly_after_database_reopen(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("s", source="api_server")
    content = parts()
    agent = SimpleNamespace(_session_db=db)
    row = _db_flush_row(agent, {"role": "user", "content": content}, True)
    db.append_message("s", **row)
    db.append_message("s", "assistant", "red")
    db.close()
    db = SessionDB(path)
    replay = db.get_messages_as_conversation("s")
    assert replay[0]["content"] == content
    assert replay[1]["content"] == "red"
    assert "api_content" not in replay[0]  # serialized sidecar must not become wire text
    raw = db.get_messages("s")[0]
    assert raw["content"] == "inspect the red pixels\n[screenshot]"
    assert content[1]["image_url"]["url"] not in raw["api_content"]
    assert "base64" not in raw["api_content"]
    with sqlite3.connect(path) as conn:
        assert "inspect" in conn.execute("SELECT content FROM messages_fts").fetchone()[0]
        assert "base64" not in conn.execute("SELECT content FROM messages_fts").fetchone()[0]
    db.close()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "replaced", "bad_sidecar"])
def test_unavailable_attachment_is_truthful_without_losing_other_history(failure, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="api_server")
    row = _db_flush_row(SimpleNamespace(_session_db=db), {"role": "user", "content": parts()}, True)
    image = next((tmp_path / "images").iterdir())
    if failure == "missing":
        image.unlink()
    elif failure == "corrupt":
        image.write_bytes(b"not an image")
    elif failure == "replaced":
        stream = BytesIO()
        Image.new("RGB", (3, 2), "blue").save(stream, format="PNG")
        image.write_bytes(stream.getvalue())
    else:
        row["api_content"] = "hermes-multimodal-v1:{broken"
    db.append_message("s", **row)
    db.append_message("s", "assistant", "prior answer")
    replay = db.get_messages_as_conversation("s")
    assert replay[1]["content"] == "prior answer"
    assert not any(p.get("type") == "image_url" for p in replay[0]["content"])
    assert "unavailable" in str(replay[0]["content"]).lower()
    db.close()


def test_image_storage_and_replay_are_profile_isolated_a_b_a(tmp_path, monkeypatch):
    import json
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from agent.session_images import decode_image_content

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / "a", tmp_path / "b"]
    rows, contents = [], []
    secret_scope.set_multiplex_active(True)
    try:
        for home in homes:
            home.mkdir()
            with _profile_runtime_scope(home):
                assert get_hermes_home() == home
                db = SessionDB(home / "state.db")
                db.create_session("same-session-id", source="api_server")
                content = parts()
                row = _db_flush_row(SimpleNamespace(_session_db=db), {"role": "user", "content": content}, True)
                assert len(list((home / "images").iterdir())) == 1
                db.append_message("same-session-id", **row)
                rows.append(row)
                contents.append(content)
                assert db.get_messages_as_conversation("same-session-id")[0]["content"] == content
                db.close()
        with _profile_runtime_scope(homes[0]):
            db = SessionDB(homes[0] / "state.db")
            assert db.get_messages_as_conversation("same-session-id")[0]["content"] == contents[0]
            db.close()
        # A sidecar cannot smuggle an absolute/traversal path into B's history.
        envelope = json.loads(rows[0]["api_content"].partition(":")[2])
        envelope["parts"][1]["image_url"]["url"]["hermes_image"] = str(next((homes[0] / "images").iterdir()))
        foreign = "hermes-multimodal-v1:" + json.dumps(envelope)
        with _profile_runtime_scope(homes[1]):
            restored = decode_image_content(foreign, get_hermes_home())
            assert "unavailable" in str(restored).lower()
            assert not any(p.get("type") == "image_url" for p in restored)
        assert str(homes[0]) not in str(contents[1])
        assert str(homes[1]) not in str(contents[0])
    finally:
        secret_scope.set_multiplex_active(False)


def test_reattachment_repairs_a_changed_managed_file(tmp_path):
    from agent.session_images import store_inline_image
    url = parts()[1]["image_url"]["url"]
    relative, mime = store_inline_image(url, tmp_path)
    target = tmp_path / relative
    target.write_bytes(b"damaged")
    assert store_inline_image(url, tmp_path) == (relative, mime)
    assert target.read_bytes() == base64.b64decode(url.partition(",")[2])


def test_clean_display_override_does_not_change_multimodal_model_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="desktop")
    live, clean = parts(), parts()
    live[0]["text"] += "\nPRIVATE TURN CONTEXT"
    agent = SimpleNamespace(_session_db=db, _persist_user_message_override=clean)
    db.append_message("s", **_db_flush_row(agent, {"role": "user", "content": live}, True))
    model, display = db.get_resume_conversations("s")
    assert model[0]["content"] == live
    assert display[0]["content"] == clean
    assert db.get_messages_as_conversation("s", include_compacted=True)[0]["content"] == clean
    db.create_session("child", source="desktop", parent_session_id="s")
    assert db.get_ancestor_display_prefix("child")[0]["content"] == clean
    from gateway.platforms.api_server import APIServerAdapter
    assert APIServerAdapter._message_response(db.get_messages("s")[0], image_home=tmp_path)["content"] == clean
    db.close()


@pytest.mark.parametrize("override", ["clean user text", "", None])
def test_string_display_override_keeps_ordered_images_and_model_parts(override, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="api_server")
    live = [parts()[1], {"type": "text", "text": "PRIVATE TURN CONTEXT"},
            parts()[1], {"type": "text", "text": "MORE INJECTED TEXT"}]
    from agent.session_images import encode_image_content, decode_image_content
    sidecar = encode_image_content(live, tmp_path, display_content=override)
    direct_expected = live if override is None else [{"type": "text", "text": override}, live[0], live[2]]
    assert decode_image_content(sidecar, tmp_path) == live
    assert decode_image_content(sidecar, tmp_path, display=True) == direct_expected
    agent = SimpleNamespace(_session_db=db, _persist_user_message_override=override)
    db.append_message("s", **_db_flush_row(agent, {"role": "user", "content": live}, True))
    model, display = db.get_resume_conversations("s")
    expected = live if override is None else [
        {"type": "text", "text": override}, live[0], live[2]]
    assert model[0]["content"] == live
    assert display[0]["content"] == expected
    assert db.get_messages_as_conversation("s", include_compacted=True)[0]["content"] == expected
    from gateway.platforms.api_server import APIServerAdapter
    assert APIServerAdapter._message_response(db.get_messages("s")[0], image_home=tmp_path)["content"] == expected
    db.close()


@pytest.mark.platforms("windows")
def test_managed_image_write_refuses_cross_profile_junction(tmp_path):
    import subprocess
    from agent.session_images import store_inline_image
    home, foreign = tmp_path / "home", tmp_path / "foreign"
    home.mkdir()
    foreign.mkdir()
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(home / "images"), str(foreign)], check=True, capture_output=True)
    try:
        with pytest.raises(ValueError, match="profile"):
            store_inline_image(parts()[1]["image_url"]["url"], home)
        assert not list(foreign.iterdir())
    finally:
        (home / "images").rmdir()
