from __future__ import annotations

import argparse
from pathlib import Path

from wtt_campaign_indexer.manifest import (
    write_campaign_manifests,
    write_campaign_summary,
)


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _prompt_text(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = _prompt_text(f"{prompt} {suffix}: ")
    if raw is None:
        print("Input unavailable; using default selection.", flush=True)
        return default

    normalized = raw.strip().lower()
    if not normalized:
        return default
    return normalized in {"y", "yes", "true", "1"}


def _resolve_tunnel_mach(tunnel_mach: float | None) -> float:
    if tunnel_mach is not None:
        return tunnel_mach

    raw = _prompt_text("Enter tunnel Mach number for this campaign summary [7.2]: ")
    if raw is None:
        print("Input unavailable; defaulting tunnel Mach to 7.2.", flush=True)
        return 7.2

    normalized = raw.strip()
    if not normalized:
        return 7.2
    return float(normalized)


def _resolve_jet_options(
    *,
    jet_used: bool | None,
    jet_mach: float | None,
) -> tuple[bool, float | None]:
    if jet_used is not None:
        if not jet_used:
            return False, None
        if jet_mach is None:
            raise ValueError("--jet-mach is required when --jet-used is provided.")
        return True, jet_mach

    prompted_jet_used = _prompt_bool("Was a jet used for this campaign summary?", default=False)
    if not prompted_jet_used:
        return False, None

    raw_mach = _prompt_text("Enter jet Mach number (e.g. 3.09): ")
    if raw_mach is None:
        raise ValueError("Jet Mach number is required when jet is used.")

    normalized = raw_mach.strip()
    if not normalized:
        raise ValueError("Jet Mach number is required when jet is used.")
    return True, float(normalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write campaign manifest files and a campaign summary markdown report"
    )
    parser.add_argument(
        "campaign_root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Campaign root folder (defaults to current directory)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("campaign_summary.md"),
        help="Path to markdown summary output file",
    )
    parser.add_argument(
        "--tunnel-mach",
        type=float,
        default=None,
        help="Tunnel Mach number used for isentropic pinf calculations (prompted if omitted)",
    )
    parser.add_argument(
        "--jet-used",
        action="store_true",
        default=None,
        help="Set when a jet was used (if omitted, you are prompted)",
    )
    parser.add_argument(
        "--no-jet-used",
        action="store_false",
        dest="jet_used",
        help="Explicitly disable jet terms in summary",
    )
    parser.add_argument(
        "--jet-mach",
        type=float,
        default=None,
        help="Jet Mach number (required when jet is used)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tunnel_mach = _resolve_tunnel_mach(args.tunnel_mach)
    jet_used, jet_mach = _resolve_jet_options(jet_used=args.jet_used, jet_mach=args.jet_mach)

    print("Starting campaign manifest generation...", flush=True)
    manifest_paths = write_campaign_manifests(
        args.campaign_root,
        tunnel_mach=tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
        progress_callback=_print_progress,
    )
    print("Starting campaign summary generation...", flush=True)
    summary_path = write_campaign_summary(
        args.campaign_root,
        args.summary_output,
        tunnel_mach=tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
        progress_callback=_print_progress,
    )
    print(f"Wrote {len(manifest_paths)} manifest files")
    print(f"Wrote summary to {summary_path}")
    print(f"Tunnel Mach={tunnel_mach}; jet_used={jet_used}; jet_mach={jet_mach}")


if __name__ == "__main__":
    main()
