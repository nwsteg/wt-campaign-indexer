import numpy as np
import pandas as pd

from wtt_campaign_indexer.lvm_fixture import pick_trigger_and_burst


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
