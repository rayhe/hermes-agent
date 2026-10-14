"""User-image sidecars: searchable text stays in content; pixels live in profile images/.

The existing api_content column carries a versioned envelope, never base64. Restoring
it is anchored to the SessionDB home, not whichever profile imported this module.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

_PREFIX = "hermes-multimodal-v1:"
_MAX_BYTES = 50 * 1024 * 1024


def store_inline_image(url: str, home: Path) -> tuple[str, str]:
    from tools.image_source import _finalize, _resolve_data_url
    import mimetypes

    raw, declared = _resolve_data_url(url)
    image = _finalize(raw, declared, "data", "session image")
    relative = "images/api_" + hashlib.sha256(raw).hexdigest() + (mimetypes.guess_extension(image.mime) or ".img")
    target = home / relative
    if not target.resolve().is_relative_to(home.resolve()):
        raise ValueError("Managed image path escapes its profile home")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.is_file() and target.stat().st_size == len(raw) and target.read_bytes() == raw):
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            temporary.write_bytes(raw)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return relative, image.mime


def _stored_parts(content: list, home: Path) -> list:
    parts = copy.deepcopy(content)
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        ref = part.get("image_url")
        url = ref.get("url") if isinstance(ref, dict) else ref
        if isinstance(url, str) and url.startswith("data:"):
            path, mime = store_inline_image(url, home)
            if not isinstance(ref, dict):
                ref = part["image_url"] = {}
            ref["url"] = {"hermes_image": path, "mime": mime}
    return parts


def encode_image_content(content: Any, home: Path, *, display_content: Any = None) -> str | None:
    if not isinstance(content, list) or not any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content
    ):
        return None
    envelope = {"parts": _stored_parts(content, home)}
    if isinstance(display_content, str):
        # A clean text override replaces ALL live text, not the ordered attachments.
        display_content = [{"type": "text", "text": display_content}] + [
            p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
    if display_content is not None and display_content != content:
        envelope["display_parts"] = _stored_parts(display_content, home)
    return _PREFIX + json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))


def decode_image_content(sidecar: Any, home: Path, *, display: bool = False) -> list | None:
    if not isinstance(sidecar, str) or not sidecar.startswith(_PREFIX):
        return None
    try:
        envelope = json.loads(sidecar[len(_PREFIX):])
        parts = envelope.get("display_parts", envelope["parts"]) if display else envelope["parts"]
        if not isinstance(parts, list):
            raise ValueError("invalid image envelope")
    except (ValueError, KeyError, TypeError, AttributeError):
        return [{"type": "text", "text": "[Image unavailable: invalid stored attachment. Ask the user to reattach it.]"}]
    parts = copy.deepcopy(parts)
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        ref = part.get("image_url")
        source = ref.get("url") if isinstance(ref, dict) else None
        if not isinstance(source, dict) or "hermes_image" not in source:
            continue
        from tools.image_source import ImageResolutionError
        try:
            relative = Path(source["hermes_image"])
            target = (home / relative).resolve()
            images_root = (home / "images").resolve()
            # A sidecar cannot authorize a cross-profile path or symlink escape.
            if (relative.is_absolute() or not target.is_relative_to(images_root)
                    or not images_root.is_relative_to(home.resolve())):
                raise ValueError("invalid managed reference")
            if target.stat().st_size > _MAX_BYTES:
                raise ValueError("oversized image")
            raw = target.read_bytes()
            expected_digest = target.stem.removeprefix("api_")
            if hashlib.sha256(raw).hexdigest() != expected_digest:
                raise ValueError("attachment bytes changed")
            from tools.image_source import _finalize
            image = _finalize(raw, source["mime"], "file", "session image")
            ref["url"] = f"data:{image.mime};base64,{base64.b64encode(raw).decode('ascii')}"
        except (OSError, ValueError, TypeError, KeyError, ImageResolutionError):
            parts[index] = {"type": "text", "text": "[Image unavailable: managed attachment is missing or unreadable. Ask the user to reattach it.]"}
    return parts
