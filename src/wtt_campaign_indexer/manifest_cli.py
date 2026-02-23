from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wtt_campaign_indexer.manifest import (
    write_campaign_manifests,
    write_campaign_summary,
)


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


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

    if not sys.stdin.isatty():
        return False, None

    prompted_jet_used = _prompt_bool("Was a jet used for this campaign summary?", default=False)
    if not prompted_jet_used:
        return False, None

    raw_mach = input("Enter jet Mach number (e.g. 3.09): ").strip()
    if not raw_mach:
        raise ValueError("Jet Mach number is required when jet is used.")
    return True, float(raw_mach)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write campaign manifest files and a campaign summary markdown report"
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("campaign_summary.md"),
        help="Path to markdown summary output file",
    )
    parser.add_argument(
        "--tunnel-mach",
        type=float,
        default=7.2,
        help="Tunnel Mach number used for isentropic pinf calculations",
    )
    parser.add_argument(
        "--jet-used",
        action="store_true",
        default=None,
        help="Set when a jet was used (if omitted, interactive prompt is shown in TTY mode)",
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
    jet_used, jet_mach = _resolve_jet_options(jet_used=args.jet_used, jet_mach=args.jet_mach)

    manifest_paths = write_campaign_manifests(
        args.campaign_root,
        tunnel_mach=args.tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
    )
    summary_path = write_campaign_summary(
        args.campaign_root,
        args.summary_output,
        tunnel_mach=args.tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
    )
    print(f"Wrote {len(manifest_paths)} manifest files")
    print(f"Wrote summary to {summary_path}")
    print(f"Tunnel Mach={args.tunnel_mach}; jet_used={jet_used}; jet_mach={jet_mach}")


if __name__ == "__main__":
    main()
