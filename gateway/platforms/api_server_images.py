"""API inline-image validation and handoff using existing profile-local attachments."""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path


def validate_inline_image(url: str) -> tuple[bytes, str]:
    """Reject incomplete payloads before admission; header sniffing is not pixel validation."""
    from PIL import Image
    from tools.image_source import ImageResolutionError, _finalize, _resolve_data_url
    from tools.vision_tools_image_prep import _validate_raster_image_decodable

    try:
        header, separator, payload = url.partition(",")
        if not separator or not re.fullmatch(
            r"data:image/[^;,\s]+(?:;[^;=,\s]+=[^;,\s]+)*;base64", header, re.IGNORECASE
        ):
            raise ValueError("invalid data URL encoding")
        raw, declared = _resolve_data_url(header.rsplit(";", 1)[0] + ";base64," + payload)
        resolved = _finalize(raw, declared, "data", "inline API image")
        # Reuse the shared frame-count and aggregate decoded-pixel limits.
        if _validate_raster_image_decodable(BytesIO(raw)):  # type: ignore[arg-type]
            raise ValueError("incomplete or excessive image")
    except (ImageResolutionError, OSError, ValueError, Image.DecompressionBombError) as exc:
        # No base64, paths, or decoder diagnostics in client-facing errors.
        raise ValueError("invalid_image_url:Image data URL must contain valid base64 and a complete decodable image.") from exc
    return raw, resolved.mime


def prepare_api_image_content(content, agent):
    """Stage only the admitted inline images; leave past context and remote URLs untouched.

    Mutate the normalized current-turn list shared with the Responses-history builder,
    keeping its current-user anchor identical to the model payload.
    """
    if not isinstance(content, list):
        return content
    from agent.session_images import store_inline_image
    from hermes_constants import get_hermes_home

    hints = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        ref = part.get("image_url")
        url = ref.get("url") if isinstance(ref, dict) else ref
        if not isinstance(url, str) or not url.startswith("data:"):
            continue
        db = getattr(agent, "_session_db", None)
        home = (Path(db.db_path).parent if db is not None else get_hermes_home()).resolve()
        relative, _ = store_inline_image(url, home)
        hints.append(f"[Image attached at: {home / relative}]")
    if not hints:
        return content
    text_part = next((p for p in content if isinstance(p, dict) and p.get("type") == "text"), None)
    if text_part is None:
        text_part = {"type": "text", "text": "What do you see in this image?"}
        content.insert(0, text_part)
    text = text_part.get("text", "")
    additions = list(dict.fromkeys(hint for hint in hints if hint not in text.splitlines()))
    if additions:
        text_part["text"] = text + "\n\n" + "\n".join(additions)
    return content
