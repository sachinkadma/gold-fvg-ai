"""
Pytest unit tests for DST-aware timezone conversion and killzone evaluation.
Validates America/New_York timezone localization and killzone assignment across DST seams.
"""

import os
import sys
import datetime
import pytest
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.signals import is_in_killzone


def test_timezone_is_america_new_york(sample_dst_data):
    """Verifies that loaded dataframe index is localized to America/New_York."""
    assert str(sample_dst_data.index.tz) == "America/New_York"


def test_spring_dst_killzone_consistency(sample_dst_data):
    """
    Tests Spring DST seam (March 14, 2021):
    Clocks jump forward from EST (UTC-5) to EDT (UTC-4).
    London Killzone must remain 02:00-05:00 in America/New_York local time.
    """
    # Friday March 12, 2021 (Winter EST, UTC-5)
    b_fri = sample_dst_data.loc["2021-03-12 02:00":"2021-03-12 02:15"]
    assert not b_fri.empty
    t_fri = b_fri.index[0]
    assert is_in_killzone(t_fri.time()) == "LONDON"
    assert t_fri.utcoffset() == datetime.timedelta(days=-1, seconds=68400)  # UTC-5

    # Monday March 15, 2021 (Summer EDT, UTC-4)
    b_mon = sample_dst_data.loc["2021-03-15 02:00":"2021-03-15 02:15"]
    assert not b_mon.empty
    t_mon = b_mon.index[0]
    assert is_in_killzone(t_mon.time()) == "LONDON"
    assert t_mon.utcoffset() == datetime.timedelta(days=-1, seconds=72000)  # UTC-4


def test_fall_dst_killzone_consistency(sample_dst_data):
    """
    Tests Fall DST seam (November 7, 2021):
    Clocks fall back from EDT (UTC-4) to EST (UTC-5).
    NY AM Killzone must remain 07:00-10:00 in America/New_York local time.
    """
    # Friday November 5, 2021 (Summer EDT, UTC-4)
    b_fri = sample_dst_data.loc["2021-11-05 07:00":"2021-11-05 07:15"]
    assert not b_fri.empty
    t_fri = b_fri.index[0]
    assert is_in_killzone(t_fri.time()) == "NY_AM"
    assert t_fri.utcoffset() == datetime.timedelta(days=-1, seconds=72000)  # UTC-4

    # Monday November 8, 2021 (Winter EST, UTC-5)
    b_mon = sample_dst_data.loc["2021-11-08 07:00":"2021-11-08 07:15"]
    assert not b_mon.empty
    t_mon = b_mon.index[0]
    assert is_in_killzone(t_mon.time()) == "NY_AM"
    assert t_mon.utcoffset() == datetime.timedelta(days=-1, seconds=68400)  # UTC-5


def test_killzone_boundaries():
    """Validates exact edge boundary behavior of is_in_killzone."""
    # London: 02:00 <= t < 05:00
    assert is_in_killzone(datetime.time(1, 59)) is None
    assert is_in_killzone(datetime.time(2, 0)) == "LONDON"
    assert is_in_killzone(datetime.time(4, 59)) == "LONDON"
    assert is_in_killzone(datetime.time(5, 0)) is None

    # NY AM: 07:00 <= t < 10:00
    assert is_in_killzone(datetime.time(6, 59)) is None
    assert is_in_killzone(datetime.time(7, 0)) == "NY_AM"
    assert is_in_killzone(datetime.time(9, 59)) == "NY_AM"
    assert is_in_killzone(datetime.time(10, 0)) is None
