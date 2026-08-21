"""Render payloads into self-contained HTML pages."""
from __future__ import annotations

import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


def _read(name: str) -> str:
    return (ASSETS / name).read_text()


def _embed(payload: dict) -> str:
    # `</script>` inside a JSON string would end the block early.
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_run(payload: dict) -> str:
    return (
        _read("run.html")
        .replace("__TITLE__", payload["run"]["name"])
        .replace("__CSS__", _read("app.css"))
        .replace("__JS__", _read("run.js"))
        .replace("__PAYLOAD__", _embed(payload))
    )


def render_index(entries: list[dict]) -> str:
    return (
        _read("index.html")
        .replace("__CSS__", _read("app.css"))
        .replace("__JS__", _read("index.js"))
        .replace("__PAYLOAD__", _embed({"runs": entries}))
    )
