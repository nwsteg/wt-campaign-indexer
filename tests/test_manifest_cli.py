from pathlib import Path

import pytest

from wtt_campaign_indexer.manifest_cli import (
    _resolve_jet_options,
    _resolve_reprocess_all,
    _resolve_summary_output,
    _resolve_tunnel_mach,
    build_parser,
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
