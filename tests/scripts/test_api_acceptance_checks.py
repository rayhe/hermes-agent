"""Regression tests for what the image acceptance harness actually proves."""
import base64
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

from PIL import Image
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "api_acceptance"


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(json.dumps(body).encode())
        self.status = status
        self.headers = headers or {}


def image_harness(tmp_path, monkeypatch, *, phase="after", protocol="chat", retention=False,
                  vision="success", leak=None, new_tool=False):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("API_SERVER_KEY", "synthetic-local-test")
    image = tmp_path / "fixture.png"
    Image.new("RGB", (3, 2), "red").save(image)
    manifest = {"path": str(image), "code": "KITE-TOP", "triangle_color": "purple",
                "lower_left": "L-SECRET", "lower_right": "R-SECRET"}
    (tmp_path / "image-continuity-fixture.json").write_text(json.dumps(manifest))
    results = tmp_path / "image-continuity-results"
    results.mkdir()
    prefix = "" if protocol == "chat" else "responses-"
    (results / (prefix + "after.json")).write_text(json.dumps({"session_id": "s", "response_id": "resp_1"}))
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    handle = "C:/synthetic/images/api_" + "a" * 64 + ".png"
    history = [{"role": "user", "content": [{"type": "text", "text": "inspect\n[Image attached at: " + handle + "]"},
                {"type": "image_url", "image_url": {"url": url}}]},
               {"role": "assistant", "content": "KITE-TOP purple"}]
    if vision != "missing":
        ref = "data:image/png;base64,[omitted]" if vision == "invalid_ref" else handle
        history.insert(1, {"role": "assistant", "tool_calls": [{"id": "v1", "function": {
            "name": "vision_analyze", "arguments": json.dumps({"image_url": ref, "question": "top only"})}}]})
        output = {"success": True, "already_in_context": True}
        if vision == "error":
            output = {"success": False, "error": "invalid image"}
        elif vision == "native":
            output = "Image loaded into your context — you can see it natively now. Use your built-in vision.\n[screenshot]"
        history.insert(2, {"role": "tool", "tool_name": "vision_analyze", "tool_call_id": "wrong" if vision == "unpaired" else "v1",
                           "content": output if isinstance(output, str) else json.dumps(output)})
    if leak:
        target = manifest["lower_right" if phase == "restart" else "lower_left"]
        if leak == "answer":
            history[-1]["content"] += " " + target
        elif leak == "arguments":
            history[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"image_url": handle, "question": target})
        elif leak == "result":
            history[2]["content"] = json.dumps({"success": True, "analysis": target})
    requests = []
    def open_request(request, timeout):
        requests.append(request)
        path = request.full_url.partition("example.test")[2]
        if path.startswith("/api/sessions/"):
            return Response({"messages": history})
        payload = json.loads(request.data)
        if "[omitted]" in json.dumps(payload):
            return Response({"error": "bad image"}, 400)
        current = next(m['content'] for m in reversed(payload.get('messages', payload.get('input'))) if m['role'] == 'user')
        text = str(current)
        if new_tool and ('lower-left' in text or 'lower-right' in text):
            history.append({'role': 'assistant', 'tool_calls': [{'id': 'new', 'function': {
                'name': 'vision_analyze', 'arguments': json.dumps({'image_url': handle})}}]})
        if "lower-left" in text:
            answer = manifest["lower_left"]
        elif "lower-right" in text:
            answer = manifest["lower_right"]
        else:
            answer = manifest["code"] + " purple"
        if protocol == "responses":
            return Response({"id": "resp_2", "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}]},
                            headers={"X-Hermes-Session-Id": "s"})
        return Response({"choices": [{"message": {"content": answer}}]})
    argv = [str(SCRIPTS / "verify_image_continuity.py"), "--home", str(tmp_path), "--base-url", "https://example.test",
            "--output-dir", str(tmp_path), "--phase", phase, "--protocol", protocol]
    if retention:
        argv.append("--retention-only")
    monkeypatch.setattr(sys, "argv", argv)
    opener = type("Opener", (), {"open": staticmethod(open_request)})()
    return requests, patch("acceptance_http.build_api_opener", return_value=opener), results / (prefix + phase + ".json")


@pytest.mark.parametrize("vision", ["missing", "invalid_ref", "error", "unpaired"])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("phase", ["after", "restart"])
def test_full_image_harness_rejects_correct_answer_without_successful_vision(vision, protocol, phase, tmp_path, monkeypatch):
    _, transport, _ = image_harness(tmp_path, monkeypatch, vision=vision, protocol=protocol, phase=phase)
    with transport, pytest.raises(AssertionError):
        runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")


@pytest.mark.parametrize("vision", ["success", "native"])
def test_full_image_harness_accepts_successful_native_inspection(vision, tmp_path, monkeypatch):
    _, transport, report = image_harness(tmp_path, monkeypatch, vision=vision)
    with transport:
        runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
    assert json.loads(report.read_text())["checks"]["authenticated_history_has_successful_vision"]["passed"]


@pytest.mark.parametrize("phase", ["after", "restart"])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("leak", ["answer", "arguments", "result"])
def test_unmentioned_label_is_checked_before_next_question(phase, protocol, leak, tmp_path, monkeypatch):
    requests, transport, _ = image_harness(tmp_path, monkeypatch, phase=phase, protocol=protocol, leak=leak)
    with transport, pytest.raises(AssertionError, match="already disclosed"):
        runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
    for request in requests:
        if request.data:
            assert "lower-left" not in request.data.decode()
            assert "lower-right" not in request.data.decode()


def test_disclosure_scan_ignores_image_bytes_but_checks_tool_json_text(tmp_path, monkeypatch):
    _, transport, _ = image_harness(tmp_path, monkeypatch)
    with transport:
        namespace = runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
    check = namespace["assert_target_unmentioned"]
    history = {"data": [{"role": "tool", "content": json.dumps({"_multimodal": True, "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECRET"}},
        {"type": "text", "text": "not disclosed"}]})}]}
    check(history, "SECRET")
    history["data"][0]["content"] = json.dumps({"analysis": "SECRET"})
    with pytest.raises(AssertionError, match="already disclosed"):
        check(history, "SECRET")


@pytest.mark.parametrize("phase", ["after", "restart"])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("new_tool", [False, True])
def test_retention_only_requires_direct_replay_without_new_tools(phase, protocol, new_tool, tmp_path, monkeypatch):
    requests, transport, report = image_harness(tmp_path, monkeypatch, phase=phase, protocol=protocol,
                                               retention=True, vision="missing", new_tool=new_tool)
    with transport:
        if new_tool:
            with pytest.raises(AssertionError):
                runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
        else:
            runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
    checks = json.loads(report.read_text())["checks"]
    name = "restart_without_new_tools" if phase == "restart" else "followup_without_new_tools"
    assert checks[name]["passed"] is (not new_tool)
    questions = [r for r in requests if r.data and ("lower-left" in r.data.decode() or "lower-right" in r.data.decode())]
    assert len(questions) == 1
    assert "Do not use any tools" in questions[0].data.decode()


@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_native_replay_uses_consistent_test_only_system_instruction(protocol, tmp_path, monkeypatch):
    requests, transport, _ = image_harness(tmp_path, monkeypatch, retention=True, vision="missing", protocol=protocol)
    with transport:
        runpy.run_path(str(SCRIPTS / "verify_image_continuity.py"), run_name="__main__")
    payloads = [json.loads(request.data) for request in requests if request.data]
    policies = []
    for payload in payloads:
        if protocol == "responses":
            policy = payload.get("instructions", "")
        else:
            systems = [m["content"] for m in payload["messages"] if m["role"] == "system"]
            assert len(systems) == 1
            policy = systems[0]
        assert "synthetic native-image replay verification only" in policy
        assert "never call tools" in policy
        policies.append(policy)
    assert len(policies) == 2 and policies[0] == policies[1]
