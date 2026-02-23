import numpy as np
import pandas as pd

from wtt_campaign_indexer.lvm_fixture import pick_failed_burst_drop, pick_trigger_and_burst


def test_pick_trigger_and_burst_basic():
    n = 200
    trigger = np.zeros(n)
    trigger[50:] = 5.0

    burst = np.zeros(n)
    burst[120:] = 10.0

    df = pd.DataFrame({"Voltage": trigger, "PLEN-PT": burst})
    result = pick_trigger_and_burst(df, rolling_trigger=3, rolling_burst=5)

    assert 48 <= result.trigger_idx <= 52
    assert 115 <= result.burst_idx <= 125


def test_multiple_triggers_choose_closest_to_burst():
    n = 300
    trigger = np.zeros(n)
    trigger[60:] = 5.0
    trigger[220:] = 10.0

    burst = np.zeros(n)
    burst[230:] = np.linspace(0, 25, n - 230)

    df = pd.DataFrame({"Voltage": trigger, "PLEN-PT": burst})
    result = pick_trigger_and_burst(df, rolling_trigger=3, rolling_burst=11)

    assert 210 <= result.trigger_idx <= 225


def test_pick_failed_burst_drop_accepts_no_nearby_trigger():
    n = 2000
    t = np.arange(n) / 100.0

    plenum = np.zeros(n)
    plenum[200:800] = np.linspace(0, 20, 600)
    plenum[800:1800] = np.linspace(20, 2, 1000)
    plenum[1800:] = 2

    trigger = np.zeros(n)
    trigger[100:120] = 5.0

    df = pd.DataFrame({"Time": t, "Voltage": trigger, "PLEN-PT": plenum})
    result = pick_failed_burst_drop(
        df,
        burst_channel="PLEN-PT",
        rolling_burst=31,
    )

    assert 790 <= result.drop_idx <= 1810
    assert result.rise_to_drop_ms > 500.0
    assert abs(result.drop_grad) / result.rise_grad >= 0.5


def test_pick_failed_burst_drop_rejects_when_fall_not_sharp_enough():
    n = 2000
    t = np.arange(n) / 100.0

    plenum = np.zeros(n)
    plenum[200:800] = np.linspace(0, 20, 600)
    plenum[800:1800] = np.linspace(20, 10, 1000)
    plenum[1800:] = 10

    trigger = np.zeros(n)

    df = pd.DataFrame({"Time": t, "Voltage": trigger, "PLEN-PT": plenum})

    import pytest

    with pytest.raises(ValueError, match="sharper than rise"):
        pick_failed_burst_drop(
            df,
            burst_channel="PLEN-PT",
            rolling_burst=31,
            min_drop_to_rise_grad_ratio=2.0,
        )


def test_pick_failed_burst_drop_rejects_when_rise_to_drop_too_short():
    n = 500
    t = np.arange(n) / 100.0

    plenum = np.zeros(n)
    plenum[100:150] = np.linspace(0, 20, 50)
    plenum[150:220] = np.linspace(20, 0, 70)

    df = pd.DataFrame({"Time": t, "Voltage": np.zeros(n), "PLEN-PT": plenum})

    import pytest

    with pytest.raises(ValueError, match="Rise-to-drop duration is too short"):
        pick_failed_burst_drop(
            df,
            burst_channel="PLEN-PT",
            rolling_burst=11,
            min_rise_to_drop_ms=1000.0,
        )
