"""D4 public observation rendering; no diagnosis, hints, or tool execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Callable

from ..probe_registry import ProbeRegistry, VERSION, _vocabulary

INVENTORY_FIELDS = (
    "file_count", "extension_counts", "total_bytes", "docx_count", "paragraph_count",
    "table_count", "section_count", "header_count", "omml_count", "xml_parse_error_count",
)
FEEDBACK_FIELDS = ("exit_code", "stdout_bytes", "stderr_bytes", "files_changed", "parse_error_count")
# Unknown suffixes must be grouped into 'other' by the inventory producer. A free
# string here would be an otherwise unchecked channel for paths or answer text.
EXTENSIONS = frozenset({".docx", ".pdf", ".txt", ".json", ".xml", ".py", ".md",
                        ".csv", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".svg",
                        ".zip", "other", "none"})
_REGISTRY = ProbeRegistry({"version": VERSION, "probes": list(_vocabulary().values())})
_HEADER = "Runtime-owned advisory: public observation presentation.\n"
_PAD_HEADER = "\nNeutral presentation filler (no workspace facts): "


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _count(value, name, *, signed=False):
    if type(value) is not int or (value < 0 and not signed) or abs(value) > 2**63 - 1:
        raise ValueError(f"invalid public integer: {name}")


def validate_public_evidence(evidence):
    """Return a detached public copy; reject extra/nested unconstrained channels."""
    if type(evidence) is not dict or set(evidence) != {"inventory", "feedback"}:
        raise ValueError("evidence must contain only inventory and feedback")
    inventory = evidence["inventory"]
    if type(inventory) is not dict or set(inventory) != set(INVENTORY_FIELDS):
        raise ValueError("invalid public inventory fields")
    for key, value in inventory.items():
        if key == "extension_counts":
            if type(value) is not dict or not set(value).issubset(EXTENSIONS):
                raise ValueError("invalid public extension categories")
            for suffix, count in value.items():
                _count(count, suffix)
        else:
            _count(value, key)
    feedback = evidence["feedback"]
    if feedback is not None:
        if type(feedback) is not dict or set(feedback) != set(FEEDBACK_FIELDS):
            raise ValueError("invalid public feedback fields")
        for key, value in feedback.items():
            _count(value, key, signed=key == "exit_code")
    return json.loads(_canonical(evidence))


@dataclass(frozen=True)
class RenderedProbe:
    probe_id: str
    public_evidence: dict
    public_digest: str
    payload: dict
    rendered_text: str
    rendered_digest: str
    token_count: int | None
    matching_audit: dict

    def to_dict(self):
        return asdict(self)


def _tokens(text, counter):
    if counter is None:
        return None
    value = counter(text)
    if type(value) is not int or value < 0:
        raise ValueError("token_counter must return a nonnegative integer")
    return value


def _body(probe_id, public):
    if probe_id == "original":
        return ""
    mode = probe_id.split(".")[1]
    sham = probe_id.endswith(".matched")
    inventory, feedback = public["inventory"], public["feedback"]
    neutral = "unreported"
    if mode == "expanded":
        groups = [("Workspace totals", INVENTORY_FIELDS[:3]),
                  ("Document structure totals", INVENTORY_FIELDS[3:-1]),
                  ("XML parser totals", INVENTORY_FIELDS[-1:])]
        return "\n".join(title + "\n" + "\n".join(
            f"{key}: {neutral if sham else _canonical(inventory[key])}" for key in keys)
            for title, keys in groups)
    if mode == "compact":
        # Same public statistics as expanded, with redundant section framing collapsed.
        return "Workspace inventory: " + "; ".join(
            f"{key}={neutral if sham else _canonical(inventory[key])}" for key in INVENTORY_FIELDS)
    if mode == "structured":
        return "Previous execution feedback\n" + "\n".join(
            f"{key}: {neutral if sham or feedback is None else feedback[key]}" for key in FEEDBACK_FIELDS)
    if mode == "aggregate":
        values = [neutral] * 4 if sham or feedback is None else [
            "zero" if feedback["exit_code"] == 0 else "nonzero",
            str(feedback["stdout_bytes"] + feedback["stderr_bytes"]),
            str(feedback["files_changed"]), str(feedback["parse_error_count"])]
        return "Previous execution summary: " + "; ".join(
            f"{key}={value}" for key, value in zip(
                ("exit_status", "output_bytes", "files_changed", "parse_error_count"), values))
    raise ValueError("unsupported rendering mode")


def render_probe(probe_id, public_evidence, *, token_counter: Callable[[str], int] | None = None):
    public = validate_public_evidence(public_evidence)
    try:
        spec = _REGISTRY.get(probe_id)
    except KeyError as exc:
        raise ValueError("unknown probe") from exc
    text = "" if probe_id == "original" else _HEADER + _body(probe_id, public)
    return RenderedProbe(probe_id, public, _digest(_canonical(public)), spec["payload"],
                         text, _digest(text), _tokens(text, token_counter),
                         {"paired": False, "byte_count": len(text.encode()),
                          "tokenizer_verified": False})


def render_pair(target_id, public_evidence, *, token_counter=None):
    """Match actual advisory byte/token costs, or fail closed after bounded search.

    Padding is explicit neutral content, not whitespace-only padding. No text is
    truncated. Matching does not certify that a control is behaviorally inert.
    """
    try:
        spec = _REGISTRY.get(target_id)
    except KeyError as exc:
        raise ValueError("unknown target") from exc
    if spec["kind"] != "target":
        raise ValueError("render_pair requires a target")
    control_id = spec["matched_control_id"]
    arms = [render_probe(key, public_evidence, token_counter=token_counter)
            for key in (target_id, control_id)]
    # ASCII-only templates make character and UTF-8 length identical.
    width = max(len(arm.rendered_text) for arm in arms) + len(_PAD_HEADER) + 1024
    candidates = [{}, {}]
    evaluated = [0, 0]
    # Some tokenizers assign the old " x"/"n" mixtures nearly identical
    # tokens per byte. Digits supply a different density without workspace
    # facts; retain letters as a second family for other tokenizer vocabularies.
    families = ("digits", "letters") if token_counter else ("letters",)
    common = set()
    for family_number, family in enumerate(families):
        tail = "0" if family == "digits" else "n"
        for index in range(513 if token_counter else 1):
            for arm_number, arm in enumerate(arms):
                prefix = arm.rendered_text + _PAD_HEADER
                size = width - len(prefix)
                filler = " x" * index + tail * (size - 2 * index)
                text = prefix + filler
                count = _tokens(text, token_counter)
                evaluated[arm_number] += 1
                candidates[arm_number].setdefault(
                    count, (text, family_number * 513 + index, family))
            common = set(candidates[0]) & set(candidates[1])
            if common:
                break
        if common:
            break
    if not common:
        raise ValueError("unable to match actual tokenizer costs within 1026 candidates per arm")
    chosen = min(common) if token_counter else None
    result = {}
    for arm, choices in zip(arms, candidates):
        text, index, family = choices[chosen]
        audit = {"paired": True, "target_id": target_id, "control_id": control_id,
                 "byte_count": len(text.encode()), "byte_counts_equal": True,
                 "tokenizer_verified": token_counter is not None,
                 "token_counts_equal": True if token_counter else None,
                 "padding_variant": index, "padding_family": family,
                 "padding_search_limit": 1026,
                 "padding_candidates_evaluated": evaluated[arms.index(arm)],
                 "control_inertness_assumed": False}
        result[arm.probe_id] = replace(arm, rendered_text=text, rendered_digest=_digest(text),
                                      token_count=chosen, matching_audit=audit)
    return result
