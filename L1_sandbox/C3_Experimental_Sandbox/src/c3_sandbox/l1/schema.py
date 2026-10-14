"""The published JSON Schema is the runtime validator, not a parallel contract."""
from __future__ import annotations
from functools import lru_cache
from importlib.resources import files
import json


@lru_cache(maxsize=1)
def document() -> dict:
    return json.loads(files(__package__).joinpath("evidence.schema.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def validator(kind: str):
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise RuntimeError("Install the L1 extra: pip install -e '.[l1]'") from exc
    doc = document()
    if kind not in doc["$defs"]:
        raise ValueError("unknown schema definition")
    schema = {"$schema": doc["$schema"], "$defs": doc["$defs"], "$ref": "#/$defs/" + kind}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate(kind: str, value) -> None:
    errors = sorted(validator(kind).iter_errors(value), key=lambda e: str(list(e.absolute_path)))
    if errors:
        err = errors[0]
        raise ValueError(f"{kind} schema at {list(err.absolute_path)}: {err.message}")
