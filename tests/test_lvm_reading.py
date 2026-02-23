from pathlib import Path

from wtt_campaign_indexer.lvm_fixture import read_lvm_data


def test_read_lvm_data_handles_extra_column_rows(tmp_path: Path):
    lvm_path = tmp_path / "ragged.lvm"
    lvm_path.write_text(
        "Meta\t1\n"
        "Another\t2\n"
        "Time\tVoltage\tPLEN-PT\n"
        "0.000\t0.0\t0.0\n"
        "0.001\t5.0\t0.1\textra\n"
        "0.002\t5.0\t0.2\n",
        encoding="utf-8",
    )

    df = read_lvm_data(lvm_path, header_row_index=2, read_every_n=1)

    assert list(df.columns) == ["Time", "Voltage", "PLEN-PT"]
    assert len(df) == 3


def test_read_lvm_data_respects_usecols_subset(tmp_path: Path):
    lvm_path = tmp_path / "subset_cols.lvm"
    lvm_path.write_text(
        "Meta\t1\n"
        "Another\t2\n"
        "Time\tVoltage\tPLEN-PT\tOther\n"
        "0.000\t0.0\t0.0\t11\n"
        "0.001\t5.0\t0.1\t22\n",
        encoding="utf-8",
    )

    df = read_lvm_data(lvm_path, header_row_index=2, read_every_n=1, usecols=[0, 2])

    assert list(df.columns) == ["Time", "PLEN-PT"]
    assert df.shape == (2, 2)


def test_read_lvm_data_respects_read_every_n(tmp_path: Path):
    lvm_path = tmp_path / "every_n.lvm"
    lvm_path.write_text(
        "Meta\t1\n"
        "Another\t2\n"
        "Time\tVoltage\tPLEN-PT\n"
        "0.000\t0.0\t0.0\n"
        "0.001\t1.0\t0.1\n"
        "0.002\t2.0\t0.2\n"
        "0.003\t3.0\t0.3\n",
        encoding="utf-8",
    )

    df = read_lvm_data(lvm_path, header_row_index=2, read_every_n=2)

    assert len(df) == 2
