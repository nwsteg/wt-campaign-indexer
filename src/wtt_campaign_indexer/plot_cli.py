from __future__ import annotations

import argparse
from pathlib import Path

from wtt_campaign_indexer.lvm_fixture import plot_existing_lvm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot plenum pressure and trigger signal from an existing LVM file"
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to existing (shortened) .lvm"
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to output .png")
    parser.add_argument("--header-row-index", type=int, default=23)
    parser.add_argument("--trigger-channel", default="Voltage")
    parser.add_argument("--plenum-channel", default="PLEN-PT")
    parser.add_argument("--fs-hz", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_existing_lvm(
        input_path=args.input,
        plot_output_path=args.output,
        header_row_index=args.header_row_index,
        trigger_channel=args.trigger_channel,
        plenum_channel=args.plenum_channel,
        fs_hz=args.fs_hz,
    )
    print(f"Wrote verification plot to {args.output}")


if __name__ == "__main__":
    main()
