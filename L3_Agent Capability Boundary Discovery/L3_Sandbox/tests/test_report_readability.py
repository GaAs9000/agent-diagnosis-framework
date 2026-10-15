"""Offline presentation checks; no execution, model, or grader calls."""
from l3_sandbox import report


def cell(task, ys, expected=3):
    n = len(ys)
    return {
        "task_id": task, "family_id": "family-a", "condition_id": "requests_070",
        "nexpected": expected, "nvalid": n, "nmissing": expected - n,
        "mean": sum(ys) / n if n else None, "nzero": ys.count(0),
        "delivery_rate": sum(bool(y) for y in ys) / n if n else None,
        "dim1_rate": sum(bool(y) for y in ys) / n if n else None,
        "delivery_unknown": 0, "dim1_unknown": 0,
        "terminations": {"tool_budget": n} if n else {},
        "model_requests": {"known_total": 7 * n, "unknown_trials": 0},
        "tool_calls": {"known_total": 14 * n, "unknown_trials": 0},
        "unknown_usage_count": 0,
    }


def test_condition_display_weights_slots_with_unequal_missing_cells():
    pooled = report._pooled([cell("a", [0, 0, 0]), cell("b", [1]), cell("c", [])])
    assert pooled["mean"] == pooled["delivery_rate"] == pooled["dim1_rate"] == .25
    assert pooled["nexpected"] == 9
    assert (pooled["nvalid"], pooled["nmissing"], pooled["nzero"]) == (4, 5, 3)
    assert pooled["terminations"] == {"tool_budget": 4}
    assert pooled["model_requests"]["known_total"] == 28


def test_observed_zero_is_not_missing_and_unknown_rates_stay_unknown():
    zero, missing = cell("zero", [0]), cell("missing", [])
    assert report._pooled([zero])["mean"] == 0
    assert report._pooled([missing])["mean"] is None
    zero.update(delivery_unknown=1, delivery_rate=None)
    assert report._pooled([zero])["delivery_rate"] is None
    assert report._display(0) == "0"
    assert report._display(None) == "unknown"


def test_markdown_shows_operational_family_usage_and_frozen_budget():
    summary = {
        "experiment_id": "pilot", "interpretation": "exploratory_pilot_only",
        "nexpected": 6, "nvalid": 1, "nmissing": 5,
        "nattempts": 2, "ninvalid_attempts": 1, "nreplacement_trials": 0,
        "model_version_drift": False,
        "family_metadata": {"n_tasks": 2, "n_families": 1,
                            "definition": "semantic_operational_group", "manifest_sha256": "a" * 64,
                            "lineage_status": "unknown", "verified_meaning": "semantic membership only"},
        "cells": [cell("zero", [0]), cell("missing", [])],
        "contrasts": [{"name": "primary", "weights": {"requests_070": 1, "requests_035": -1},
                       "n_tasks_paired": 0, "n_tasks_expected": 2, "estimate": None,
                       "minimum_effect": .1, "ci95": None, "ci_reason": "incomplete_predetermined_cohort",
                       "status": "incomplete"}],
        "attempt_usage": {"known_total_tokens": {"known_total": 42, "unknown_trials": 1}},
    }
    plan = {"jobs": [{"config": {"budget": {"max_tool_calls": 140, "wall_time_seconds": 10800}}}]}
    text = report._markdown(summary, plan)
    assert "| requests_070 | 6 | 1 | 5 | 0.0000 |" in text
    assert "| zero | family-a | requests_070 | 1/3 | 2 | 0.0000 |" in text
    assert "| missing | family-a | requests_070 | 0/3 | 3 | unknown |" in text
    assert "tool_budget: 1" in text
    assert "140" in text and "10800" in text
    assert "requests_035: -1; requests_070: 1" in text and "0.1000" in text
    assert "semantic_operational_group" in text and "a" * 64 in text
    assert "statistical independence is not certified" in text
    assert "| known_total_tokens | 42 | 1 |" in text
    assert "Actual monetary cost is unknown" in text
    assert "Missing outcomes are not zero" in text
