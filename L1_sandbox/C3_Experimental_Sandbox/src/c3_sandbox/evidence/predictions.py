"""Freeze complete pre-confirmation predictions, never inferred result records."""
import json
import math
from pathlib import Path

from ._storage import directory, encode, get, put, local_path


def _validate(record):
    if not isinstance(record, dict):
        raise ValueError("prediction must be an object")
    for key in ("case_id", "method_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if type(record.get("is_synthetic")) is not bool:
        raise ValueError("is_synthetic must be explicit boolean")
    if record.get("pre_confirmation") is not True:
        raise ValueError("pre_confirmation must explicitly be true")
    budget = record.get("budget")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget < 0:
        raise ValueError("budget must be a finite nonnegative number")
    universe = record.get("candidate_universe")
    if not isinstance(universe, list) or not universe or not all(isinstance(key, str) and key.strip() for key in universe) or len(set(universe)) != len(universe):
        raise ValueError("candidate_universe must contain unique nonempty candidate ids")
    probabilities = record.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(universe):
        raise ValueError("probabilities must cover exactly the complete candidate universe")
    for value in probabilities.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("probabilities must be finite numbers in [0,1]")
    encode(record)  # Also reject nonfinite values elsewhere and non-JSON payloads.


class PredictionStore:
    def __init__(self, root):
        self.root = local_path(root)
        directory(self.root)

    def freeze(self, record: dict) -> str:
        _validate(record)
        key = put(self.root, encode(record), ".json")
        if self.read(key) != record:
            raise ValueError("prediction failed readback verification")
        return key

    def read(self, prediction_id: str) -> dict:
        record = json.loads(get(self.root, prediction_id, ".json"))
        _validate(record)
        return record
