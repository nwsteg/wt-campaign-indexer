from pathlib import Path

import pandas as pd

from wtt_campaign_indexer.lvm_fixture import create_lvm_fixture, plot_existing_lvm

FIXTURE_IN = Path("tests/fixtures/sample_input.lvm")


def test_fixture_writer_preserves_header_and_row_count(tmp_path: Path):
    out_path = tmp_path / "short.lvm"
    metadata_path = tmp_path / "short.json"
    plot_path = tmp_path / "short.png"

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
        plot_output_path=plot_path,
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines[3] == "Time\tVoltage\tPLEN-PT\tOther"

    df = pd.read_csv(out_path, sep="\t", header=3)

    # window size at 1 kHz: pre=2 samples, post=3 samples => 6 total before decimation
    # decimation=2 keeps 3 samples, plus forced burst sample can add one.
    assert 3 <= len(df) <= 4
    assert metadata["decimation"] == 2
    assert metadata_path.exists()
    assert plot_path.exists()


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


def test_default_window_is_plus_minus_100ms(tmp_path: Path):
    out_path = tmp_path / "short_default.lvm"

    metadata = create_lvm_fixture(
        input_path=FIXTURE_IN,
        output_path=out_path,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        header_row_index=3,
        fs_hz=1000.0,
    )

    assert metadata["pre_ms"] == 100.0
    assert metadata["post_ms"] == 100.0


def test_plot_existing_shortened_lvm(tmp_path: Path):
    short_path = tmp_path / "short_for_plot.lvm"
    plot_path = tmp_path / "from_short.png"

    create_lvm_fixture(
        input_path=FIXTURE_IN,
        output_path=short_path,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        header_row_index=3,
        fs_hz=1000.0,
        pre_ms=2,
        post_ms=3,
    )

    plot_existing_lvm(
        input_path=short_path,
        plot_output_path=plot_path,
        header_row_index=3,
        trigger_channel="Voltage",
        plenum_channel="PLEN-PT",
        fs_hz=1000.0,
    )

    assert plot_path.exists()


def test_failed_burst_drop_window_anchor(tmp_path: Path):
    src = tmp_path / "failed_case.lvm"
    out_path = tmp_path / "failed_case_fixture.lvm"

    rows = [
        "LabVIEW Measurement\n",
        "Writer_Version\t2\n",
        "Separator\tTab\n",
        "Time\tVoltage\tPLEN-PT\tOther\n",
    ]

    fs_hz = 100.0
    n = 2000
    for i in range(n):
        time_s = i / fs_hz
        if i < 200:
            plenum = 0.0
        elif i < 800:
            plenum = (i - 200) * (20.0 / 600.0)
        elif i < 1800:
            plenum = 20.0 - (i - 800) * (18.0 / 1000.0)
        else:
            plenum = 2.0

        voltage = 0.0
        if 120 <= i < 140:
            voltage = 5.0

        rows.append(f"{time_s:.3f}\t{voltage:.3f}\t{plenum:.3f}\t1\n")

    src.write_text("".join(rows), encoding="utf-8")

    metadata = create_lvm_fixture(
        input_path=src,
        output_path=out_path,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        header_row_index=3,
        pre_ms=200.0,
        post_ms=300.0,
        fs_hz=fs_hz,
        decimate=1,
        window_anchor="failed-burst-drop",
        failed_burst_min_rise_to_drop_ms=500.0,
        failed_burst_min_drop_to_rise_grad_ratio=0.5,
        rolling_trigger=5,
        rolling_burst=31,
    )

    assert metadata["window_anchor"] == "failed-burst-drop"
    assert metadata["failed_burst_drop_input_index"] is not None
    assert metadata["trigger_index_in_fixture"] is None
    assert metadata["window_anchor_index_in_fixture"] >= 0

    df = pd.read_csv(out_path, sep="\t", header=3)
    assert len(df) > 0
