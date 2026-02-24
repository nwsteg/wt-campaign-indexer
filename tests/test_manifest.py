import inspect
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
    write_campaign_summary_figures,
    write_fst_manifest,
)


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_write_campaign_summary_signature_matches_figures_contract() -> None:
    summary_params = inspect.signature(write_campaign_summary).parameters
    figures_params = inspect.signature(write_campaign_summary_figures).parameters

    assert "campaign_root" in summary_params
    assert "output_path" in summary_params
    assert "progress_callback" in summary_params
    assert "reprocess_all" in summary_params
    assert "jet_used" in summary_params

    assert "campaign_root" in figures_params
    assert "summary_output_path" in figures_params
    assert "progress_callback" in figures_params
    assert "reprocess_all" in figures_params
    assert "jet_used" in figures_params

    assert summary_params["campaign_root"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert summary_params["output_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert summary_params["progress_callback"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert summary_params["reprocess_all"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert summary_params["jet_used"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    assert figures_params["campaign_root"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert figures_params["summary_output_path"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert figures_params["progress_callback"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert figures_params["reprocess_all"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert figures_params["jet_used"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


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
    assert "| FST_1388 | pls |" in summary
    assert "## FST traces" in summary
    assert "### FST_1388" in summary
    assert "_Plot directory not provided._" in summary


def test_write_campaign_summary_writes_file(tmp_path: Path):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _write_long_lvm(fst_dir / "FST_1391.lvm")

    output_path = tmp_path / "campaign_summary.md"
    result_path = write_campaign_summary(tmp_path, output_path)

    assert result_path == output_path
    assert output_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "FST_1391" in markdown

    figure_dir = tmp_path / "campaign_summary_figs"
    assert figure_dir.exists()
    assert (figure_dir / "FST_1391_overview.png").exists()
    assert "![FST_1391 trace](campaign_summary_figs/FST_1391_overview.png)" in markdown

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


def test_cache_reuses_unchanged_fsts_across_runs(tmp_path: Path, monkeypatch):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1391.lvm")

    write_campaign_manifests(tmp_path)

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest.build_fst_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not rebuild")),
    )
    write_campaign_manifests(tmp_path, reprocess_all=False)


def test_cache_processes_only_new_fst(tmp_path: Path, monkeypatch):
    fst_a = tmp_path / "FST1391"
    fst_a.mkdir()
    _touch(fst_a / "FST_1391.lvm")
    write_campaign_manifests(tmp_path)

    fst_b = tmp_path / "FST1392"
    fst_b.mkdir()
    _touch(fst_b / "FST_1392.lvm")

    rebuilt: list[str] = []
    original = __import__(
        "wtt_campaign_indexer.manifest", fromlist=["build_fst_manifest"]
    ).build_fst_manifest

    def _tracking_build(fst, **kwargs):
        rebuilt.append(fst.normalized_name)
        return original(fst, **kwargs)

    monkeypatch.setattr("wtt_campaign_indexer.manifest.build_fst_manifest", _tracking_build)
    write_campaign_manifests(tmp_path)

    assert rebuilt == ["FST_1392"]


def test_cache_invalidates_only_entries_with_setting_change(tmp_path: Path, monkeypatch):
    fst_a = tmp_path / "FST1391"
    fst_b = tmp_path / "FST1392"
    fst_a.mkdir()
    fst_b.mkdir()
    _touch(fst_a / "FST_1391.lvm")
    _touch(fst_b / "FST_1392.lvm")

    write_campaign_manifests(tmp_path, tunnel_mach=7.2, jet_used=False, jet_mach=None)

    cache_path = tmp_path / ".wtt_campaign_cache.json"
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_payload["fsts"]["FST_1392"]["settings"]["tunnel_mach"] = 8.0
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    rebuilt: list[str] = []
    original = __import__(
        "wtt_campaign_indexer.manifest", fromlist=["build_fst_manifest"]
    ).build_fst_manifest

    def _tracking_build(fst, **kwargs):
        rebuilt.append(fst.normalized_name)
        return original(fst, **kwargs)

    monkeypatch.setattr("wtt_campaign_indexer.manifest.build_fst_manifest", _tracking_build)
    write_campaign_manifests(tmp_path, tunnel_mach=7.2, jet_used=False, jet_mach=None)

    assert rebuilt == ["FST_1392"]


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


def _write_long_lvm(path: Path, n: int = 800, fs_hz: float = 1000.0) -> None:
    lines = ["LabVIEW Measurement\n", "meta\n", "Time\tVoltage\tPLEN-PT\tTC 8\tLVDT\n"]
    dt = 1.0 / fs_hz
    for idx in range(n):
        t = idx * dt
        if idx < 180:
            voltage = 0.0
        elif idx > 200:
            voltage = 5.0
        else:
            voltage = (idx - 180) * 0.25
        if idx < 260:
            plenum = 20.0
            tc8 = 300.0
            lvdt = 30.0
        elif idx < 300:
            plenum = 20.0 + (idx - 260) * 0.5
            tc8 = 300.0 + (idx - 260) * 0.05
            lvdt = 30.0 + (idx - 260) * 0.2
        else:
            plenum = 40.0
            tc8 = 302.0
            lvdt = 38.0
        lines.append(f"{t:.6f}	{voltage:.3f}	{plenum:.6f}	{tc8:.6f}	{lvdt:.6f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _baseline_steady_state_summary(lvm_path: Path):
    from wtt_campaign_indexer.lvm_fixture import (
        detect_header_row_from_file,
        infer_sample_rate_hz,
        pick_trigger_and_burst,
        read_lvm_data,
    )

    header_row_index = detect_header_row_from_file(lvm_path, "Voltage", "PLEN-PT")
    df = read_lvm_data(lvm_path, header_row_index=header_row_index)
    detection = pick_trigger_and_burst(df, trigger_channel="Voltage", burst_channel="PLEN-PT")
    fs_hz = infer_sample_rate_hz(df)
    start_idx = detection.burst_idx + int(round(50.0 * fs_hz / 1000.0))
    end_idx = detection.burst_idx + int(round(90.0 * fs_hz / 1000.0))
    window = df.iloc[start_idx : end_idx + 1]
    return {
        "p0": float(window["PLEN-PT"].mean()),
        "T0": float(window["TC 8"].mean()),
        "p0j": float(window["LVDT"].mean()),
    }


def test_two_pass_steady_state_matches_full_read_within_tolerance(tmp_path: Path):
    from wtt_campaign_indexer.manifest import compute_steady_state_conditions

    lvm_path = tmp_path / "FST_1391.lvm"
    _write_long_lvm(lvm_path)

    baseline = _baseline_steady_state_summary(lvm_path)
    summary, _notes = compute_steady_state_conditions(lvm_path)

    assert summary["p0"] == pytest.approx(baseline["p0"], rel=1e-3, abs=1e-3)
    assert summary["T0"] == pytest.approx(baseline["T0"], rel=1e-3, abs=1e-3)
    assert summary["p0j"] == pytest.approx(baseline["p0j"], rel=1e-3, abs=1e-3)


def test_summary_figure_uses_expected_window_bounds(tmp_path: Path, monkeypatch):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _write_long_lvm(fst_dir / "FST_1391.lvm")

    captured = {}

    def _capture(
        df_window, anchor_idx, anchor_label, start_idx, fs_hz, output_path, fst_id, jet_used
    ):
        captured["len"] = len(df_window)
        captured["start_ms"] = (start_idx - anchor_idx) * 1000.0 / fs_hz
        captured["end_ms"] = (start_idx + len(df_window) - 1 - anchor_idx) * 1000.0 / fs_hz

    monkeypatch.setattr("wtt_campaign_indexer.manifest._write_fst_summary_plot", _capture)

    out = tmp_path / "campaign_summary.md"
    write_campaign_summary(tmp_path, out)

    assert captured["start_ms"] == pytest.approx(-20.0, abs=2.0)
    assert captured["end_ms"] == pytest.approx(120.0, abs=2.0)
    assert captured["len"] > 10


def test_fallback_used_when_coarse_detection_confidence_insufficient(tmp_path: Path, monkeypatch):
    from wtt_campaign_indexer.lvm_fixture import CoarseEventEstimate
    from wtt_campaign_indexer.manifest import compute_steady_state_conditions

    lvm_path = tmp_path / "FST_1391.lvm"
    _write_long_lvm(lvm_path)

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest.detect_event_index_coarse",
        lambda *args, **kwargs: CoarseEventEstimate(
            trigger_idx=None,
            burst_idx=None,
            anchor_idx=None,
            sample_rate_hz=None,
            confidence_ok=False,
            reason="simulated low confidence",
        ),
    )

    summary, notes = compute_steady_state_conditions(lvm_path)
    assert summary["p0"] is not None
    assert any("coarse detection confidence insufficient" in note.lower() for note in notes)
