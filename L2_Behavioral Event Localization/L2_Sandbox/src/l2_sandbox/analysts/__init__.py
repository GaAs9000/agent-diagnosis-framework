"""L2 v0.3 Analysts (Rules, Reference, and LLM)."""
from __future__ import annotations
from typing import TypedDict


class Values(TypedDict):
    analyst: str
    values: list[float]


class Hypothesis(TypedDict):
    turn: int
    predicate: dict
    effect: float
    rationale: str
