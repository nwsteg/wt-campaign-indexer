import json
from pathlib import Path

from wtt_campaign_indexer.discovery import discover_campaign
from wtt_campaign_indexer.manifest import (
    build_campaign_summary_markdown,
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
        "diagnostics",
        "runs",
        "condition_summary",
        "units",
    ]
    assert payload["fst_id"] == "FST_1391"
    assert payload["lvm_path"].endswith("FST_1391.lvm")
    assert payload["lvm_fixture_path"].endswith("FST_1391_fixture.lvm")

    assert payload["runs"][0]["run_id"] == "run_001"
    assert payload["runs"][0]["inferred_rate_hz"] == 25000.0
    assert payload["runs"][0]["errors"] == []

    assert payload["condition_summary"] == {
        "p0": None,
        "T0": None,
        "Re_1": None,
        "p0j": None,
        "p0j/p0": None,
        "p0j/pinf": None,
        "J": None,
    }
    assert payload["units"]["condition_summary"]["p0"] == "Pa"
    assert payload["units"]["runs"]["inferred_rate_hz"] == "Hz"


def test_manifest_deterministic_ordering_and_formatting(tmp_path: Path):
    fst_dir = tmp_path / "FST_1392"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1392.lvm")
    _touch(fst_dir / "tracking" / "run_001" / "meta.cihx", "<FrameRate>1000</FrameRate>")

    fst = discover_campaign(tmp_path).fsts[0]
    manifest_path = write_fst_manifest(fst)

    expected = """{
  \"fst_id\": \"FST_1392\",
  \"lvm_path\": \"__LVM_PATH__\",
  \"lvm_fixture_path\": null,
  \"diagnostics\": [
    {
      \"name\": \"tracking\",
      \"is_known\": true,
      \"run_count\": 1
    }
  ],
  \"runs\": [
    {
      \"run_id\": \"run_001\",
      \"diagnostic\": \"tracking\",
      \"cihx_path\": \"__CIHX_PATH__\",
      \"inferred_rate_hz\": 1000.0,
      \"notes\": [],
      \"errors\": []
    }
  ],
  \"condition_summary\": {
    \"p0\": null,
    \"T0\": null,
    \"Re_1\": null,
    \"p0j\": null,
    \"p0j/p0\": null,
    \"p0j/pinf\": null,
    \"J\": null
  },
  \"units\": {
    \"condition_summary\": {
      \"p0\": \"Pa\",
      \"T0\": \"K\",
      \"Re_1\": \"1/m\",
      \"p0j\": \"Pa\",
      \"p0j/p0\": \"1\",
      \"p0j/pinf\": \"1\",
      \"J\": \"1\"
    },
    \"runs\": {
      \"inferred_rate_hz\": \"Hz\",
      \"run_id\": null,
      \"cihx_path\": null
    }
  }
}
"""
    expected = expected.replace("__LVM_PATH__", str(fst_dir / "FST_1392.lvm"))
    expected = expected.replace(
        "__CIHX_PATH__",
        str(fst_dir / "tracking" / "run_001" / "meta.cihx"),
    )

    assert manifest_path.read_text(encoding="utf-8") == expected


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

    summary = build_campaign_summary_markdown(tmp_path)

    assert "# Dummy campaign summary" in summary
    assert "| FST_1388 | FST_1388.lvm | 1 | 1 |" in summary
    assert "| pls | yes | 1 |" in summary
    assert "| pls | run_S0001 | 10000.000 |" in summary


def test_write_campaign_summary_writes_file(tmp_path: Path):
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    _touch(fst_dir / "FST_1391.lvm")

    output_path = tmp_path / "campaign_summary.md"
    result_path = write_campaign_summary(tmp_path, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert "FST_1391" in output_path.read_text(encoding="utf-8")
