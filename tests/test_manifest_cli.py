import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import wtt_campaign_indexer.manifest_cli as manifest_cli_module

from wtt_campaign_indexer.manifest_cli import (
    _resolve_jet_options,
    _resolve_reprocess_all,
    _resolve_summary_output,
    _resolve_tunnel_mach,
    build_parser,
    main,
)


def test_parser_accepts_campaign_root_as_positional() -> None:
    args = build_parser().parse_args(["examples/dummy_campaign"])

    assert str(args.campaign_root).endswith("examples/dummy_campaign")


def test_parser_defaults_campaign_root_to_current_directory() -> None:
    args = build_parser().parse_args([])

    assert str(args.campaign_root) == "."


def test_resolve_tunnel_mach_uses_explicit_value() -> None:
    assert _resolve_tunnel_mach(6.5) == 6.5


def test_resolve_tunnel_mach_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli._prompt_text", lambda _prompt: "8.1")

    assert _resolve_tunnel_mach(None) == 8.1


def test_resolve_tunnel_mach_uses_default_when_prompt_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli._prompt_text", lambda _prompt: "")

    assert _resolve_tunnel_mach(None) == 7.2


def test_resolve_tunnel_mach_uses_default_when_input_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli._prompt_text", lambda _prompt: None)

    assert _resolve_tunnel_mach(None) == 7.2


def test_resolve_jet_options_prompts_when_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = iter(["y", "3.09"])
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli._prompt_text",
        lambda _prompt: next(prompts),
    )

    jet_used, jet_mach = _resolve_jet_options(jet_used=None, jet_mach=None)

    assert jet_used is True
    assert jet_mach == 3.09


def test_resolve_summary_output_defaults_to_campaign_root() -> None:
    campaign_root = Path("/tmp/campaign")

    assert _resolve_summary_output(campaign_root, None) == campaign_root / "campaign_summary.md"


def test_resolve_summary_output_respects_explicit_path() -> None:
    campaign_root = Path("/tmp/campaign")
    explicit = Path("custom/output.md")

    assert _resolve_summary_output(campaign_root, explicit) == explicit


def test_resolve_reprocess_all_obeys_explicit_flags(tmp_path: Path) -> None:
    summary_path = tmp_path / "campaign_summary.md"

    assert (
        _resolve_reprocess_all(
            tmp_path,
            summary_path,
            reprocess_all=True,
            reuse_existing=False,
            tunnel_mach=7.2,
            jet_used=False,
            jet_mach=None,
        )
        is True
    )
    assert (
        _resolve_reprocess_all(
            tmp_path,
            summary_path,
            reprocess_all=False,
            reuse_existing=True,
            tunnel_mach=7.2,
            jet_used=False,
            jet_mach=None,
        )
        is False
    )


def test_resolve_reprocess_all_prompts_when_outputs_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    (fst_dir / "FST_1391.lvm").write_text("", encoding="utf-8")
    (fst_dir / "FST_1391_manifest.json").write_text("{}", encoding="utf-8")

    summary_path = tmp_path / "campaign_summary.md"
    summary_path.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli._prompt_bool",
        lambda _prompt, default: True,
    )

    assert (
        _resolve_reprocess_all(
            tmp_path,
            summary_path,
            reprocess_all=False,
            reuse_existing=False,
            tunnel_mach=7.2,
            jet_used=False,
            jet_mach=None,
        )
        is True
    )


def test_resolve_reprocess_all_prompt_includes_predicted_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    (fst_dir / "FST_1391.lvm").write_text("", encoding="utf-8")
    (fst_dir / "FST_1391_manifest.json").write_text("{}", encoding="utf-8")

    summary_path = tmp_path / "campaign_summary.md"
    summary_path.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli.predict_campaign_reuse_counts",
        lambda *_args, **_kwargs: (8, 2),
    )

    captured = {}

    def _capture_prompt(prompt: str, default: bool) -> bool:
        captured["prompt"] = prompt
        return False

    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli._prompt_bool", _capture_prompt)

    _resolve_reprocess_all(
        tmp_path,
        summary_path,
        reprocess_all=False,
        reuse_existing=False,
        tunnel_mach=7.2,
        jet_used=False,
        jet_mach=None,
    )

    assert "predicted: 8 reuse, 2 rebuild" in captured["prompt"]


def test_main_passes_expected_kwargs_to_manifest_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fst_dir = tmp_path / "FST1391"
    fst_dir.mkdir()
    (fst_dir / "FST_1391_manifest.json").write_text("{}", encoding="utf-8")

    summary_output = tmp_path / "campaign_summary.md"
    summary_output.write_text("existing", encoding="utf-8")
    figure_dir = tmp_path / "campaign_summary_figs"
    figure_dir.mkdir()
    (figure_dir / "FST_1391_overview.png").write_text("", encoding="utf-8")

    class _Parser:
        @staticmethod
        def parse_args():
            return argparse.Namespace(
                campaign_root=tmp_path,
                summary_output=summary_output,
                tunnel_mach=None,
                jet_used=None,
                jet_mach=None,
                reprocess_all=False,
                reuse_existing=False,
            )

    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli.build_parser", lambda: _Parser())

    prompts = iter(["8.1", "3.3"])
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli._prompt_text",
        lambda _prompt: next(prompts),
    )

    prompt_calls: list[str] = []

    def _prompt_bool(prompt: str, default: bool) -> bool:
        del default
        prompt_calls.append(prompt)
        if len(prompt_calls) == 1:
            return True
        return False

    monkeypatch.setattr("wtt_campaign_indexer.manifest_cli._prompt_bool", _prompt_bool)
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli.discover_campaign",
        lambda _root: SimpleNamespace(
            fsts=[SimpleNamespace(path=fst_dir, normalized_name="FST_1391")]
        ),
    )
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli.predict_campaign_reuse_counts",
        lambda *_args, **_kwargs: (1, 0),
    )

    captured: dict[str, object] = {}

    def _write_campaign_manifests(
        campaign_root: Path,
        *,
        tunnel_mach: float,
        jet_used: bool,
        jet_mach: float | None,
        progress_callback,
        reprocess_all: bool,
    ) -> list[Path]:
        captured["manifest"] = {
            "campaign_root": campaign_root,
            "tunnel_mach": tunnel_mach,
            "jet_used": jet_used,
            "jet_mach": jet_mach,
            "progress_callback": progress_callback,
            "reprocess_all": reprocess_all,
        }
        return [fst_dir / "FST_1391_manifest.json"]

    def _write_campaign_summary(
        campaign_root: Path,
        output_path: Path,
        *,
        tunnel_mach: float,
        jet_used: bool,
        jet_mach: float | None,
        progress_callback,
        reprocess_all: bool,
    ) -> Path:
        captured["summary"] = {
            "campaign_root": campaign_root,
            "output_path": output_path,
            "tunnel_mach": tunnel_mach,
            "jet_used": jet_used,
            "jet_mach": jet_mach,
            "progress_callback": progress_callback,
            "reprocess_all": reprocess_all,
        }
        return output_path

    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli.write_campaign_manifests",
        _write_campaign_manifests,
    )
    monkeypatch.setattr(
        "wtt_campaign_indexer.manifest_cli.write_campaign_summary",
        _write_campaign_summary,
    )

    main()

    assert prompt_calls[0].startswith("Was a jet used")
    assert "Reprocess all FSTs?" in prompt_calls[1]
    assert captured["manifest"] == {
        "campaign_root": tmp_path,
        "tunnel_mach": 8.1,
        "jet_used": True,
        "jet_mach": 3.3,
        "progress_callback": manifest_cli_module._print_progress,
        "reprocess_all": False,
    }
    assert captured["summary"] == {
        "campaign_root": tmp_path,
        "output_path": summary_output,
        "tunnel_mach": 8.1,
        "jet_used": True,
        "jet_mach": 3.3,
        "progress_callback": manifest_cli_module._print_progress,
        "reprocess_all": False,
    }
