"""Prompt templates and answer options per event kind and target (spec 7.2)."""

from __future__ import annotations

from ..events import Event, EventKind

PERSON_SCREEN = [
    "pedestrian",
    "cyclist",
    "motorcycle_rider",
    "scooter_rider",
    "person_in_vehicle",
    "not_a_person",
    "unsure",
]
VEHICLE_SCREEN = [
    "vehicle",
    "part_or_duplicate_of_another_vehicle",
    "not_a_vehicle",
    "unsure",
]
SAME = ["same_individual", "different", "unsure"]
#: Rider answers and the ``RefineConfig.reclass_map`` key each one selects.
RIDER_OPTIONS = {"cyclist": "cyclist", "motorcycle_rider": "motorcycle", "scooter_rider": "scooter"}

_GLOSS = {
    "pedestrian": "a person on foot",
    "cyclist": "a person riding a bicycle",
    "motorcycle_rider": "a person riding a motorcycle",
    "scooter_rider": "a person riding a scooter",
    "person_in_vehicle": "a person inside a vehicle",
    "not_a_person": "not a person at all (a shadow, a sign, a pole, a bag, ...)",
    "vehicle": "a complete road vehicle",
    "part_or_duplicate_of_another_vehicle": "a part of, or a second box on, another vehicle",
    "not_a_vehicle": "not a vehicle at all",
    "same_individual": "the same individual in both rows",
    "different": "two different individuals",
    "unsure": "you cannot tell",
}


def options_for(event: Event, target: str) -> list[str] | None:
    """Return the answer options for ``event``, or ``None`` if it is never sent to a VLM."""
    if event.stage in ("switch", "link") and event.kind in (EventKind.SPLIT, EventKind.LINK):
        return list(SAME)
    if event.stage == "screen" and event.kind in (EventKind.DROP, EventKind.RECLASS):
        return list(PERSON_SCREEN if target == "person" else VEHICLE_SCREEN)
    return None


def build_prompt(event: Event, target: str, options: list[str], *, fps: float) -> str:
    """Return the question for ``event`` (deterministic text, no event identifiers)."""
    if event.kind is EventKind.SPLIT:
        body = (
            "The tracker kept one ID across a possible change of object. Row A shows crops of "
            "the tracked object before the cut; row B shows crops after it. Is the object in "
            "row B the same individual as in row A?"
        )
    elif event.kind is EventKind.LINK:
        t_e, t_s = event.params["gap"]
        gap = (int(t_s) - int(t_e)) / float(fps)
        body = (
            "The tracker lost an object and later started a new track. Row A shows the last "
            f"crops of the first track; row B shows the first crops of the second track, "
            f"{gap:.1f} s later. Are A and B the same individual?"
        )
    else:
        what = "a pedestrian" if target == "person" else "a road vehicle"
        if event.params.get("spans"):  # a partial edit: show what it would change and what not
            body = (
                f"The tracker reports {what}, but the tracker's cue fires only on part of one "
                "track. Row A shows crops from that part, which is the part that would be "
                "changed. Row B shows crops from the rest of the same track. What is the "
                "tracked object in row A? Use row B only to see what the object looked like "
                "before or after."
            )
        else:
            body = (
                f"The tracker reports {what}. The crops were taken at evenly spaced moments "
                "of one track (a context frame follows when present). What is the tracked "
                "object?"
            )
    lines = [body, "", "Options:"]
    lines += [f"- {o}: {_GLOSS[o]}" for o in options]
    lines += [
        "",
        "Reply with only a JSON object, no other text:",
        '{"answer": "<one of the options>", "confidence": <number from 0 to 1>, '
        '"reason": "<one short sentence>"}',
    ]
    return "\n".join(lines)
