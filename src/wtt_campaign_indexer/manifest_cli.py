from __future__ import annotations

import argparse
from pathlib import Path

from wtt_campaign_indexer.manifest import write_campaign_manifests, write_campaign_summary


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_paths = write_campaign_manifests(args.campaign_root)
    summary_path = write_campaign_summary(args.campaign_root, args.summary_output)
    print(f"Wrote {len(manifest_paths)} manifest files")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
