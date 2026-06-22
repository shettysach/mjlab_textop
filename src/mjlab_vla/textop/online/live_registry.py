from __future__ import annotations

from threading import Lock
from uuid import uuid4

from mjlab_vla.textop.online.source import TextOpOnlineSource

_LIVE_SOURCES: dict[str, TextOpOnlineSource] = {}
_LOCK = Lock()


def register_live_textop_source(source: TextOpOnlineSource) -> str:
    key = uuid4().hex
    with _LOCK:
        _LIVE_SOURCES[key] = source
    return key


def get_live_textop_source(key: str) -> TextOpOnlineSource:
    with _LOCK:
        try:
            return _LIVE_SOURCES[key]
        except KeyError as exc:
            raise KeyError(f"Unknown live TextOp source key: {key}") from exc


def unregister_live_textop_source(key: str) -> None:
    with _LOCK:
        _LIVE_SOURCES.pop(key, None)
