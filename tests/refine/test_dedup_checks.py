import pandas as pd
import pytest

from ._dedup_checks import check_retained

GROUP = (122, 125, 134)
RAW = {(122, 10), (125, 10), (122, 11), (134, 12), (7, 10)}  # (7, 10) is not in the group


def _rows(observed=(), filled=()):
    return pd.DataFrame({
        "frame": [*observed, *filled],
        "interp": [0] * len(observed) + [1] * len(filled),
    })


def test_passes_when_the_winner_is_observed_on_the_shared_frame():
    check_retained(_rows(observed=[10, 11, 12]), RAW, {(125, 10)}, GROUP)


def test_fails_when_the_winners_observed_row_is_missing():
    # (122, 10) won frame 10 and (125, 10) was dropped: frame 10 must still be observed
    with pytest.raises(AssertionError, match="missing"):
        check_retained(_rows(observed=[11, 12]), RAW, {(125, 10)}, GROUP)


def test_fails_when_only_a_filled_row_remains_on_the_winners_frame():
    with pytest.raises(AssertionError, match="missing"):
        check_retained(_rows(observed=[11, 12], filled=[10]), RAW, {(125, 10)}, GROUP)


def test_fails_when_two_retained_observations_compete_on_a_frame():
    with pytest.raises(AssertionError, match="compete"):
        check_retained(_rows(observed=[10, 11, 12]), RAW, set(), GROUP)


def test_fails_when_a_dropped_row_is_not_in_the_input():
    with pytest.raises(AssertionError, match="not in the input"):
        check_retained(_rows(observed=[10, 11, 12]), RAW, {(125, 10), (125, 99)}, GROUP)


def test_ignores_raw_rows_outside_the_group():
    check_retained(_rows(observed=[10, 11, 12]), RAW, {(125, 10)}, GROUP)  # (7, 10) never counts
