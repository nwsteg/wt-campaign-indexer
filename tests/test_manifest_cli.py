import pytest

from wtt_campaign_indexer.manifest_cli import (
    _resolve_jet_options,
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
