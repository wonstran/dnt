"""Checks shared by the dedup real-data tests and their own tests."""


def check_retained(rows, raw_rows, dropped, group):
    """Assert that every retained raw observation of ``group`` is an observed output row.

    ``rows`` are the output rows of the consolidated identity (columns ``frame`` and
    ``interp``); ``raw_rows`` is the set of ``(raw_id, frame)`` of the input; ``dropped`` is the
    set of ``(raw_id, frame)`` that the applied ``MERGE`` events of the group list as dropped.
    The accounting is done on raw observations first, and projected to frames only afterwards:
    a frame where one observation lost still needs the winner's observed row.
    """
    original = {(r, f) for r, f in raw_rows if r in group}
    assert dropped <= original, f"dropped rows not in the input: {sorted(dropped - original)[:5]}"
    retained = original - dropped
    frames = [f for _, f in retained]
    assert len(frames) == len(set(frames)), "two retained observations compete on one frame"
    observed = set(rows.loc[rows["interp"] == 0, "frame"])
    missing = sorted(set(frames) - observed)
    assert not missing, f"retained observations missing from the output: {missing[:10]}"
