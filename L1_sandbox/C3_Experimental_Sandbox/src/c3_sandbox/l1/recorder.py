"""Controller-owned, fsync'ed append-only trajectory journal.

For live use this must run outside the untrusted workspace. A verified import
chain authenticates its own ordering only, NOT the historical wall clock.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .schema import validate
from .storage import canonical, digest, write_bytes


class EventJournal:
    def __init__(self, root: Path, episode_id: str):
        self.root = root
        self.episode_id = episode_id
        self.path = write_bytes(root, "trajectory.jsonl", b"")
        self.events: list[dict] = []

    def append(self, kind: str, payload: dict, *, links: dict | None = None,
               source_refs: list[dict] | None = None, observed_at: str | None = None,
               assurance: str = "source_recorded") -> dict:
        seq = len(self.events)
        event = {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id,
                 "event_id": f"{self.episode_id}:e{seq:06d}", "seq": seq, "kind": kind,
                 "observed_at": observed_at, "assurance": assurance,
                 "source_refs": source_refs or [], "links": links or {}, "payload": payload,
                 "prev_event_sha256": self.events[-1]["event_sha256"] if self.events else None}
        event["event_sha256"] = digest(event)
        validate("event", event)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "ab") as f:
            f.write(canonical(event) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(event)
        return event


def scan_journal(path: Path) -> dict:
    """Report a recoverable valid prefix; never drop or rewrite a torn tail."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events, errors = [], []
    previous = None
    for i, line in enumerate(lines):
        if not line.endswith(b"\n"):
            errors.append({"line": i + 1, "code": "unterminated_tail"})
            break
        try:
            event = json.loads(line)
            validate("event", event)
            unhashed = {k: v for k, v in event.items() if k != "event_sha256"}
            if event["seq"] != i or event["prev_event_sha256"] != previous or digest(unhashed) != event["event_sha256"]:
                raise ValueError("broken event sequence or hash chain")
            events.append(event)
            previous = event["event_sha256"]
        except (ValueError, TypeError) as exc:
            errors.append({"line": i + 1, "code": "invalid_event", "detail": str(exc)})
            break
    return {"events": events, "errors": errors, "valid_prefix_events": len(events),
            "complete": not errors, "raw_size_bytes": len(raw)}
