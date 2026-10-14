"""Credential-safe transport shared by the local acceptance scripts."""
from urllib.parse import urlsplit
from urllib.request import BaseHandler, HTTPRedirectHandler, HTTPSHandler, build_opener


def validate_api_origin(value):
    error = "--base-url must be an HTTPS origin with a hostname and optional trailing slash only"
    try:
        url = urlsplit(value)
        port = url.port
        import re
        hostname = (url.hostname or '').encode('idna').decode('ascii').removesuffix('.')
        valid_host = ':' in hostname or (len(hostname) <= 253 and all(
            re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label)
            for label in hostname.split('.')))
        if (url.scheme != "https" or not valid_host or url.netloc.endswith(':') or url.username is not None
                or url.password is not None or url.path not in ("", "/")
                or "?" in value or "#" in value or "\\" in value
                or any(c.isspace() or ord(c) < 32 for c in value)
                or any(c in url.hostname for c in "%/?#@") or port == 0):
            raise ValueError(error)
    except ValueError:
        raise ValueError(error) from None
    return value.removesuffix("/")


def build_api_opener(base_url, *, context=None):
    base = urlsplit(validate_api_origin(base_url))
    origin = (base.scheme, base.hostname, base.port or 443)

    def check(value):
        try:
            url = urlsplit(value)
            allowed = (url.scheme, url.hostname, url.port or 443) == origin
            allowed &= url.username is None and url.password is None and not url.fragment
        except ValueError:
            allowed = False
        if not allowed:
            raise ValueError("API requests and redirects must stay on the same HTTPS origin")

    class OriginGuard(BaseHandler):
        def https_request(self, request):
            check(request.full_url)
            return request
        http_request = https_request

    class SameOriginRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return build_opener(OriginGuard(), SameOriginRedirect(), HTTPSHandler(context=context))
