"""Runtime provenance and exact deployment discovery must fail closed."""
import io
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "api_acceptance"


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode())
        self.status = status


def setup_args(tmp_path, monkeypatch, preflight=False):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("API_SERVER_KEY", "synthetic-local-test")
    argv = [str(SCRIPTS / "verify_model_routes.py"), "--home", str(tmp_path),
            "--base-url", "https://example.test", "--output-dir", str(tmp_path)]
    if preflight:
        argv.append("--preflight")
    monkeypatch.setattr(sys, "argv", argv)


@pytest.mark.parametrize("provider", [None, "wrong-provider", "expected"])
def test_session_metadata_needs_proven_provider(provider, tmp_path, monkeypatch):
    setup_args(tmp_path, monkeypatch, preflight=True)
    submitted = {}
    def open_request(request, timeout):
        path = request.full_url.partition("example.test")[2]
        if path == "/v1/runs":
            submitted.update(json.loads(request.data))
            return Response({"run_id": "r"}, 202)
        marker = next((tmp_path / "cache" / "scratch").glob("*.txt")).read_text()
        if path == "/v1/runs/r":
            return Response({"status": "completed", "session_id": "s", "output": marker})
        if path.endswith("/messages?limit=100"):
            return Response({"data": [{"role": "tool", "tool_name": "terminal", "content": marker}]})
        assert path == "/api/sessions/s"
        metadata = {"model": submitted["model"]}
        if provider:
            metadata["provider"] = submitted["provider"] if provider == "expected" else provider
        return Response({"data": metadata})
    opener = type("Opener", (), {"open": staticmethod(open_request)})()
    with patch("acceptance_http.build_api_opener", return_value=opener):
        if provider == "expected":
            runpy.run_path(str(SCRIPTS / "verify_model_routes.py"), run_name="__main__")
        else:
            with pytest.raises(AssertionError, match="Model route verification failed"):
                runpy.run_path(str(SCRIPTS / "verify_model_routes.py"), run_name="__main__")
    report = json.loads((tmp_path / "model-route-preflight.json").read_text())
    assert report["all_passed"] is (provider == "expected")
    for check in report["checks"].values():
        assert check["runtime"].get("provider") == (check["expected_backend"]["provider"] if provider == "expected" else provider)


def test_discovery_rejects_unexpected_exposed_alias(tmp_path, monkeypatch):
    setup_args(tmp_path, monkeypatch)
    def open_request(request, timeout):
        assert request.full_url.endswith("/v1/models"), "Discovery did not fail closed"
        return Response({"data": [{"id": alias} for alias in ["hermes-agent", "gpt-6.1-sol", "opus-5.5",
                        "swift-1.5-qwen3.8-27b-orcarouter", "unexpected-private-backend"]]})
    opener = type("Opener", (), {"open": staticmethod(open_request)})()
    with patch("acceptance_http.build_api_opener", return_value=opener), \
            pytest.raises(AssertionError, match="exact"):
        runpy.run_path(str(SCRIPTS / "verify_model_routes.py"), run_name="__main__")
    report = json.loads((tmp_path / "model-route-verification.json").read_text())
    assert report["model_discovery"]["passed"] is False
