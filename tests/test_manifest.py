import json
import shutil
from pathlib import Path

import pytest

from wtt_campaign_indexer.discovery import discover_campaign
from wtt_campaign_indexer.manifest import (
    build_campaign_summary_markdown,
    infer_rate_from_cihx,
    infer_rate_from_hcc,
    write_campaign_manifests,
    write_campaign_summary,
    write_fst_manifest,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_manifest_required_keys_always_exist(tmp_path: Path):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1391.lvm")
    _touch(fst_dir / "FST_1391_fixture.lvm")
    _touch(
        fst_dir / "schlieren" / "run_001" / "meta.cihx",
        "AcquisitionFrameRate = 25000",
    )

    fst = discover_campaign(tmp_path).fsts[0]
    manifest_path = write_fst_manifest(fst)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert list(payload) == [
        "fst_id",
        "lvm_path",
        "lvm_fixture_path",
        "tunnel_mach",
        "jet_used",
        "jet_mach",
        "diagnostics",
        "runs",
        "condition_summary",
        "condition_notes",
        "units",
    ]
    assert payload["fst_id"] == "FST_1391"
    assert payload["lvm_path"].endswith("FST_1391.lvm")
    assert payload["lvm_fixture_path"].endswith("FST_1391_fixture.lvm")

    assert payload["runs"][0]["run_id"] == "run_001"
    assert payload["runs"][0]["inferred_rate_hz"] == 25000.0
    assert payload["runs"][0]["errors"] == []
    assert payload["runs"][0]["is_support_run"] is False

    assert payload["condition_summary"] == {
        "p0": None,
        "T0": None,
        "Re_1": None,
        "p0j": None,
        "T0j": None,
        "pinf": None,
        "pinf_ref_jet_mach": None,
        "pj": None,
        "p0j/pinf": None,
        "pj/pinf": None,
        "J": None,
    }
    assert payload["units"]["condition_summary"]["p0"] == "psia"
    assert payload["units"]["runs"]["inferred_rate_hz"] == "Hz"


def test_manifest_deterministic_ordering_and_formatting(tmp_path: Path):
    fst_dir = tmp_path / "FST_1392"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1392.lvm")
    _touch(fst_dir / "tracking" / "run_001" / "meta.cihx", "<FrameRate>1000</FrameRate>")

    fst = discover_campaign(tmp_path).fsts[0]
    manifest_path = write_fst_manifest(fst)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["fst_id"] == "FST_1392"
    assert payload["diagnostics"] == [{"name": "tracking", "is_known": True, "run_count": 1}]
    assert payload["runs"][0]["run_id"] == "run_001"
    assert payload["runs"][0]["is_support_run"] is False
    assert payload["runs"][0]["inferred_rate_hz"] == 1000.0
    assert payload["units"]["condition_summary"]["T0j"] == "K"


def test_manifest_handles_absent_diagnostics_gracefully(tmp_path: Path):
    fst_dir = tmp_path / "FST_1400"
    fst_dir.mkdir()

    fst = discover_campaign(tmp_path).fsts[0]
    manifest_path = write_fst_manifest(fst)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["lvm_path"] is None
    assert payload["lvm_fixture_path"] is None
    assert payload["diagnostics"] == []
    assert payload["runs"] == []
    assert payload["condition_summary"]["p0"] is None


def test_campaign_summary_contains_discovery_overview(tmp_path: Path):
    fst_dir = tmp_path / "FST1388"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1388.lvm")
    _touch(fst_dir / "pls" / "run_S0001" / "meta.cihx", "AcquisitionFrameRate = 10000")

    summary = build_campaign_summary_markdown(
        tmp_path, tunnel_mach=7.2, jet_used=True, jet_mach=3.09
    )

    assert "# Dummy campaign summary" in summary
    assert "## Top-level overview (steady-state, 50-90 ms after burst)" in summary
    assert "| FST_1388 | FST_1388.lvm | 1 | 1 |" in summary
    assert "| pls | yes | 1 |" in summary
    assert "| pls | run_S0001 | no | 10000.000 |" in summary
    assert "Jet enabled" in summary


def test_write_campaign_summary_writes_file(tmp_path: Path):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    shutil.copyfile("tests/fixtures/sample_input.lvm", fst_dir / "FST_1391.lvm")

    output_path = tmp_path / "campaign_summary.md"
    result_path = write_campaign_summary(tmp_path, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert "FST_1391" in output_path.read_text(encoding="utf-8")

    figure_dir = tmp_path / "campaign_summary_figs"
    assert figure_dir.exists()
    assert (figure_dir / "FST_1391_overview.png").exists()


def test_infer_rate_supports_record_rate_pattern(tmp_path: Path):
    cihx = tmp_path / "meta.cihx"
    cihx.write_text("recordRate=20000", encoding="utf-8")

    assert infer_rate_from_cihx(cihx) == 20000.0


def test_infer_rate_from_hcc_prefers_telops_reader(tmp_path: Path, monkeypatch):
    hcc = tmp_path / "meta.hcc"
    hcc.write_bytes(b"\x00\x01\x02")

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_telops_hcc", lambda _path: 180.0
    )
    monkeypatch.setattr("wtt_campaign_indexer.manifest._infer_rate_from_av", lambda _path: None)
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_imageio", lambda _path: None
    )

    assert infer_rate_from_hcc(hcc) == 180.0


def test_infer_rate_from_hcc_falls_back_to_av(tmp_path: Path, monkeypatch):
    hcc = tmp_path / "meta.hcc"
    hcc.write_bytes(b"\x00\x01\x02")

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_telops_hcc", lambda _path: None
    )
    monkeypatch.setattr("wtt_campaign_indexer.manifest._infer_rate_from_av", lambda _path: 355.0)
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_imageio", lambda _path: None
    )

    assert infer_rate_from_hcc(hcc) == 355.0


def test_manifest_prefers_hcc_when_cihx_missing(tmp_path: Path, monkeypatch):
    fst_dir = tmp_path / "FST1402"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1402.lvm")
    _touch(fst_dir / "ir" / "run_S0001" / "run_S0001.hcc")

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_telops_hcc", lambda _path: 355.0
    )
    monkeypatch.setattr("wtt_campaign_indexer.manifest._infer_rate_from_av", lambda _path: None)
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest._infer_rate_from_imageio", lambda _path: None
    )

    fst = discover_campaign(tmp_path).fsts[0]
    manifest_path = write_fst_manifest(fst)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["runs"][0]["cihx_path"] is None
    assert payload["runs"][0]["hcc_path"].endswith("run_S0001.hcc")
    assert payload["runs"][0]["inferred_rate_hz"] == 355.0


def test_infer_rate_from_hcc_supports_telops_header_fallback(tmp_path: Path):
    hcc = tmp_path / "meta.hcc"
    header = bytearray(80)
    header[:4] = b"TC\x02\r"
    header[44:48] = (3_000_000).to_bytes(4, "little")
    header[76:80] = (1500).to_bytes(4, "little")
    hcc.write_bytes(bytes(header) + b"\x00" * 16)

    assert infer_rate_from_hcc(hcc) == 3000.0


def test_campaign_summary_skips_fst_with_skip_marker(tmp_path: Path):
    fst_keep = tmp_path / "FST1388"
    fst_skip = tmp_path / "FST1389"
    fst_keep.mkdir()
    fst_skip.mkdir()
    _touch(fst_keep / "FST_1388.lvm")
    _touch(fst_keep / "pls" / "run_S0001" / "meta.cihx", "AcquisitionFrameRate = 10000")
    _touch(fst_skip / "FST_1389.lvm")
    _touch(fst_skip / "skip.txt")

    summary = build_campaign_summary_markdown(
        tmp_path, tunnel_mach=7.2, jet_used=True, jet_mach=3.09
    )

    assert "FST_1388" in summary
    assert "FST_1389" not in summary


def test_write_campaign_manifests_reuses_existing_when_unchanged(tmp_path: Path, monkeypatch):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1391.lvm")

    write_campaign_manifests(tmp_path)

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest.build_fst_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not rebuild")),
    )

    # Reuse path should not call build_fst_manifest again.
    write_campaign_manifests(tmp_path, reprocess_all=False)


def test_write_campaign_manifests_reprocess_all_forces_rebuild(tmp_path: Path, monkeypatch):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1391.lvm")

    write_campaign_manifests(tmp_path)

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest.build_fst_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced rebuild")),
    )

    with pytest.raises(RuntimeError, match="forced rebuild"):
        write_campaign_manifests(tmp_path, reprocess_all=True)
