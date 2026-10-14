"""Acceptance harnesses must not send even synthetic credentials outside HTTPS origin."""
import importlib
import runpy
import ssl
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "api_acceptance"


@pytest.mark.parametrize("script", ["verify_api.py", "verify_image_continuity.py", "verify_model_routes.py"])
@pytest.mark.parametrize("origin", ["http://example.test", "https:///missing", "https://user:secret@example.test",
                                    "https://example.test/v1", "https://example.test?key=x",
                                    "https://example.test#fragment", "https://example.test?",
                                    "https://example.test#", "https://bad host.test", "https://example.test:bad",
                                    "https://bad_.test", "https://-bad.test", "https://example.test:"])
def test_invalid_origin_is_rejected_before_key_read_or_request(script, origin, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    argv = [str(SCRIPTS / script), "--home", str(tmp_path), "--base-url", origin,
            "--output-dir", str(tmp_path / "output")]
    if script == "verify_image_continuity.py":
        argv += ["--phase", "after"]
    monkeypatch.setattr(sys, "argv", argv)
    with patch("dotenv.dotenv_values", side_effect=AssertionError("key read before validation")), \
            patch("urllib.request.urlopen", side_effect=AssertionError("request before validation")):
        with pytest.raises(ValueError, match="HTTPS origin"):
            runpy.run_path(str(SCRIPTS / script), run_name="__main__")


def test_https_origin_accepts_only_optional_trailing_slash():
    helper = importlib.import_module("scripts.api_acceptance.acceptance_http")
    assert helper.validate_api_origin("https://example.test/") == "https://example.test"
    assert helper.validate_api_origin("https://example.test:8443") == "https://example.test:8443"


@pytest.fixture
def tls_contexts(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_file, key_file = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    client = ssl.create_default_context(cafile=str(cert_file))
    return server, client


@contextmanager
def listener(respond, tls=None):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            status, headers, body = respond(self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        do_POST = do_GET
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield ("https" if tls else "http") + "://localhost:" + str(server.server_port), seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("destination", ["same", "other_port", "other_host", "http", "credentials"])
def test_real_https_redirect_policy_never_leaks_test_token(destination, tls_contexts):
    helper = importlib.import_module("scripts.api_acceptance.acceptance_http")
    server, client = tls_contexts
    with listener(lambda path: (200, {}, b"foreign"), server) as (foreign, foreign_seen):
        targets = {}
        def respond(path):
            return (302, {"Location": targets["url"]}, b"") if path == "/start" else (200, {}, b"ok")
        with listener(respond, server) as (origin, seen):
            targets["url"] = {"same": "/end", "other_port": foreign + "/end",
                "other_host": origin.replace("localhost", "127.0.0.1") + "/end",
                "http": origin.replace("https:", "http:") + "/end",
                "credentials": origin.replace("//", "//user:secret@") + "/end"}[destination]
            opener = helper.build_api_opener(origin, context=client)
            req = urllib.request.Request(origin + "/start", headers={"Authorization": "Bearer synthetic-local-test"})
            if destination == "same":
                with opener.open(req, timeout=5) as response:
                    assert response.read() == b"ok"
                assert seen == [("/start", "Bearer synthetic-local-test"), ("/end", "Bearer synthetic-local-test")]
            else:
                with pytest.raises(ValueError, match="same HTTPS origin"):
                    opener.open(req, timeout=5)
                assert seen == [("/start", "Bearer synthetic-local-test")]
            assert not foreign_seen
            # The opener also guards the initial request, before any bytes leave.
            with pytest.raises(ValueError, match="same HTTPS origin"):
                opener.open(urllib.request.Request(foreign + "/direct", headers={"Authorization": "Bearer synthetic-local-test"}), timeout=5)
            assert not foreign_seen


@pytest.mark.parametrize("script", ["verify_api.py", "verify_image_continuity.py", "verify_model_routes.py"])
def test_scripts_enforce_real_cross_origin_redirect(script, tls_contexts, tmp_path, monkeypatch):
    server, client = tls_contexts
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("API_SERVER_KEY", "synthetic-local-test")
    with listener(lambda path: (200, {}, b"{}"), server) as (foreign, foreign_seen):
        with listener(lambda path: (302, {"Location": foreign + "/stolen"}, b""), server) as (origin, seen):
            argv = [str(SCRIPTS / script), "--home", str(tmp_path), "--base-url", origin,
                    "--output-dir", str(tmp_path / "output")]
            if script == "verify_image_continuity.py":
                argv += ["--phase", "after"]
            monkeypatch.setattr(sys, "argv", argv)
            with patch("ssl._create_default_https_context", return_value=client):
                with pytest.raises(ValueError, match="same HTTPS origin"):
                    runpy.run_path(str(SCRIPTS / script), run_name="__main__")
            assert len(seen) == 1
            assert not foreign_seen
