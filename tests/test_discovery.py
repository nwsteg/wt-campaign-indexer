from pathlib import Path

from wtt_campaign_indexer.discovery import discover_campaign


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_missing_lvm(tmp_path: Path):
    fst_dir = tmp_path / "FST1391"
    (fst_dir / "kulite").mkdir(parents=True)

    result = discover_campaign(tmp_path)

    assert len(result.fsts) == 1
    fst = result.fsts[0]
    assert fst.number == 1391
    assert fst.normalized_name == "FST_1391"
    assert fst.primary_lvm is None


def test_multiple_lvm_candidates_prefers_exact_name(tmp_path: Path):
    fst_dir = tmp_path / "FST_1392"
    fst_dir.mkdir()
    _touch(fst_dir / "A_first.lvm")
    _touch(fst_dir / "FST_1392.lvm")
    _touch(fst_dir / "Z_last.lvm")

    result = discover_campaign(tmp_path)

    assert len(result.fsts) == 1
    assert result.fsts[0].primary_lvm == fst_dir / "FST_1392.lvm"


def test_camera_and_non_camera_diagnostics(tmp_path: Path):
    fst_dir = tmp_path / "FST1393"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1393.lvm")

    schlieren = fst_dir / "schlieren"
    (schlieren / "run_002").mkdir(parents=True)
    (schlieren / "run_001").mkdir(parents=True)
    _touch(schlieren / "run_001" / "b.cihx")
    _touch(schlieren / "run_001" / "a.cihx")
    _touch(schlieren / "run_002" / "meta.cihx")

    (fst_dir / "kulite").mkdir()

    result = discover_campaign(tmp_path)
    fst = result.fsts[0]

    diagnostics = {diag.name: diag for diag in fst.diagnostics}
    assert [diag.name for diag in fst.diagnostics] == ["kulite", "schlieren"]

    schlieren_diag = diagnostics["schlieren"]
    assert [run.name for run in schlieren_diag.runs] == ["run_001", "run_002"]
    assert [path.name for path in schlieren_diag.runs[0].cihx_files] == ["a.cihx", "b.cihx"]

    kulite_diag = diagnostics["kulite"]
    assert kulite_diag.runs == ()


def test_unknown_diagnostics_and_fst_sorting(tmp_path: Path):
    fst_b = tmp_path / "FST2000"
    fst_a = tmp_path / "FST_1399"
    fst_b.mkdir()
    fst_a.mkdir()
    (fst_b / "zzz_special").mkdir()
    (fst_b / "alpha_misc").mkdir()

    result = discover_campaign(tmp_path)

    assert [fst.number for fst in result.fsts] == [1399, 2000]

    unknown = result.fsts[1].diagnostics
    assert [diag.name for diag in unknown] == ["alpha_misc", "zzz_special"]
    assert all(diag.is_known is False for diag in unknown)


def test_pls_is_known_and_nested_runs_are_discovered(tmp_path: Path):
    fst_dir = tmp_path / "FST1388"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1388.lvm")

    _touch(fst_dir / "pls" / "run_S0001" / "run_S0001.cihx")
    _touch(fst_dir / "shift" / "shift_S0001" / "shift_S0001.cihx")

    result = discover_campaign(tmp_path)
    fst = result.fsts[0]

    diagnostics = {diag.name: diag for diag in fst.diagnostics}
    assert diagnostics["pls"].is_known is True
    assert diagnostics["shift"].is_known is False

    assert [run.name for run in diagnostics["pls"].runs] == ["run_S0001"]
    assert [run.name for run in diagnostics["shift"].runs] == ["shift_S0001"]


def test_ir_is_known_and_hcc_runs_are_discovered(tmp_path: Path):
    fst_dir = tmp_path / "FST1401"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1401.lvm")

    _touch(fst_dir / "ir" / "run_S0002" / "run_S0002.hcc")

    result = discover_campaign(tmp_path)
    fst = result.fsts[0]

    diagnostics = {diag.name: diag for diag in fst.diagnostics}
    assert diagnostics["ir"].is_known is True

    run = diagnostics["ir"].runs[0]
    assert run.name == "run_S0002"
    assert [path.name for path in run.hcc_files] == ["run_S0002.hcc"]
    assert run.cihx_files == ()


def test_skip_txt_excludes_fst_from_discovery(tmp_path: Path):
    fst_keep = tmp_path / "FST1401"
    fst_skip = tmp_path / "FST1402"
    fst_keep.mkdir()
    fst_skip.mkdir()
    _touch(fst_keep / "FST_1401.lvm")
    _touch(fst_skip / "FST_1402.lvm")
    _touch(fst_skip / "skip.txt")

    result = discover_campaign(tmp_path)

    assert [fst.normalized_name for fst in result.fsts] == ["FST_1401"]


def test_ignores_common_cache_directories(tmp_path: Path):
    fst_dir = tmp_path / "FST1394"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1394.lvm")

    (fst_dir / ".ipynb_checkpoints").mkdir()
    (fst_dir / "pycache").mkdir()
    (fst_dir / "__pycache__").mkdir()
    (fst_dir / "piv").mkdir()

    result = discover_campaign(tmp_path)

    assert [diag.name for diag in result.fsts[0].diagnostics] == ["piv"]
