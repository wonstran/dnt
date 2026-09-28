"""Proposed edits, their decisions, and the JSONL ledger (spec 4.1, 4.2)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from pathlib import Path

import numpy as np


class EventKind(StrEnum):
    """What an event proposes to change."""

    DROP = "DROP"
    RECLASS = "RECLASS"
    SPLIT = "SPLIT"
    LINK = "LINK"
    FILL = "FILL"
    SMOOTH = "SMOOTH"


class Decision(StrEnum):
    """How an event was decided."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    AUTO_REJECT = "AUTO_REJECT"
    VLM_ACCEPT = "VLM_ACCEPT"
    VLM_REJECT = "VLM_REJECT"
    HUMAN_PENDING = "HUMAN_PENDING"
    HUMAN_ACCEPT = "HUMAN_ACCEPT"
    HUMAN_REJECT = "HUMAN_REJECT"


ACCEPTED = frozenset({Decision.AUTO_ACCEPT, Decision.VLM_ACCEPT, Decision.HUMAN_ACCEPT})
REJECTED = frozenset({Decision.AUTO_REJECT, Decision.VLM_REJECT, Decision.HUMAN_REJECT})
DEFINING_PARAMS: dict[EventKind, tuple[str, ...]] = {
    EventKind.SPLIT: ("cut_frame",),
    EventKind.DROP: ("reason", "spans", "of"),
    EventKind.RECLASS: ("new_cls", "spans"),
    EventKind.LINK: ("gap",),
    EventKind.FILL: ("gap",),
    EventKind.SMOOTH: (),
}


def clean_json(obj):
    """Return ``obj`` as strict-JSON-safe data (numpy -> Python, NaN/inf -> None)."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj) if isinstance(obj, set | frozenset) else obj
        return [clean_json(v) for v in items]
    if isinstance(obj, np.ndarray):
        return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _dumps(obj) -> str:
    return json.dumps(clean_json(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_integral_floats(obj):
    """Recursively convert integral floats (e.g., 821.0, np.float64(821.0)) to int."""
    if isinstance(obj, dict):
        return {k: _normalize_integral_floats(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_normalize_integral_floats(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return int(v) if v.is_integer() else v
    if isinstance(obj, float):
        return int(obj) if obj.is_integer() else obj
    return obj


def proposal_key(stage: str, kind, lineage, params: dict) -> str:
    """Return the immutable key of a proposal (spec 4.2)."""
    k = EventKind(kind)
    defining = {name: params.get(name) for name in DEFINING_PARAMS[k]}
    # Normalize integral floats to ints for consistent hashing
    defining = _normalize_integral_floats(defining)
    lineage = _normalize_integral_floats(lineage)
    return hashlib.sha256(_dumps([stage, k.value, lineage, defining]).encode()).hexdigest()


@dataclass
class Event:
    """One proposed edit, its decision, and its evidence (spec 4.1)."""

    id: str
    proposal_key: str
    round: int
    stage: str
    kind: EventKind
    tracks: list[int]
    lineage: list
    frames: tuple[int, int]
    params: dict
    edit: dict | None
    algo_score: float
    signals: dict
    decision: Decision | None
    decision_history: list[dict] = field(default_factory=list)
    vlm: dict | None = None
    applied: bool = False

    @classmethod
    def propose(
        cls,
        *,
        stage: str,
        kind,
        tracks,
        lineage,
        frames,
        params: dict,
        algo_score: float,
        signals: dict | None = None,
        round: int = 0,
    ) -> Event:
        """Create an undecided proposal and compute its key."""
        params = clean_json(dict(params))
        lineage = clean_json(lineage)
        return cls(
            id="",
            proposal_key=proposal_key(stage, kind, lineage, params),
            round=int(round),
            stage=stage,
            kind=EventKind(kind),
            tracks=[int(t) for t in tracks],
            lineage=lineage,
            frames=(int(frames[0]), int(frames[1])),
            params=params,
            edit=None,
            algo_score=float(algo_score),
            signals=clean_json(dict(signals or {})),
            decision=None,
        )

    @property
    def accepted(self) -> bool:
        """Whether the current decision accepts the event."""
        return self.decision in ACCEPTED

    def to_dict(self) -> dict:
        """Return the event as JSON-safe data."""
        return clean_json(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        """Rebuild an event written by ``to_dict``."""
        d = dict(d)
        d["kind"] = EventKind(d["kind"])
        d["decision"] = Decision(d["decision"]) if d.get("decision") else None
        d["frames"] = (int(d["frames"][0]), int(d["frames"][1]))
        return cls(**d)


@dataclass
class Ledger:
    """A header plus the events of one run (spec 4.2)."""

    header: dict
    events: list[Event]

    def write(self, path) -> None:
        """Write the header line and one line per event."""
        lines = [_dumps(self.header), *(_dumps(e.to_dict()) for e in self.events)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path) -> Ledger:
        """Read a ledger written by ``write``."""
        lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        header = json.loads(lines[0])
        return cls(header=header, events=[Event.from_dict(json.loads(ln)) for ln in lines[1:]])


def assign_ids(events: list[Event], stage: str, round: int, start: int = 1) -> int:
    """Give unnumbered events IDs ``{stage}-r{round}-{seq:06d}``; return the next sequence."""
    seq = start
    for ev in events:
        if not ev.id:
            ev.id = f"{stage}-r{round}-{seq:06d}"
            seq += 1
    return seq
