from pathlib import Path

import pandas as pd

from wtt_campaign_indexer.lvm_fixture import create_lvm_fixture

FIXTURE_IN = Path("tests/fixtures/sample_input.lvm")


def test_fixture_writer_preserves_header_and_row_count(tmp_path: Path):
    out_path = tmp_path / "short.lvm"
    metadata_path = tmp_path / "short.json"

    metadata = create_lvm_fixture(
        input_path=FIXTURE_IN,
        output_path=out_path,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        header_row_index=3,
        pre_ms=2,
        post_ms=3,
        fs_hz=1000.0,
        decimate=2,
        rolling_trigger=3,
        rolling_burst=5,
        metadata_json_path=metadata_path,
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[3] == "Time\tVoltage\tPLEN-PT\tOther"

    df = pd.read_csv(out_path, sep="\t", header=3)

    # window size at 1 kHz: pre=2 samples, post=3 samples => 6 total before decimation
    # decimation=2 keeps 3 samples, plus forced burst sample can add one.
    assert 3 <= len(df) <= 4
    assert metadata["decimation"] == 2
    assert metadata_path.exists()


def test_header_autodetect_when_index_is_wrong(tmp_path: Path):
    out_path = tmp_path / "short_auto.lvm"

    create_lvm_fixture(
        input_path=FIXTURE_IN,
        output_path=out_path,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        header_row_index=23,
        pre_ms=2,
        post_ms=2,
        fs_hz=1000,
    )

    df = pd.read_csv(out_path, sep="\t", header=3)
    assert "Voltage" in df.columns
