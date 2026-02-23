from __future__ import annotations

import argparse
from pathlib import Path

from wtt_campaign_indexer.lvm_fixture import create_lvm_fixture


def _none_or_string(value: str) -> str | None:
    lowered = value.strip().lower()
    return None if lowered in {"none", "null", ""} else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a shortened .lvm fixture around trigger/burst"
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to source .lvm file")
    parser.add_argument("--output", type=Path, required=True, help="Path to output fixture .lvm")
    parser.add_argument("--trigger-channel", default="Voltage")
    parser.add_argument("--burst-channel", type=_none_or_string, default="PLEN-PT")
    parser.add_argument("--header-row-index", type=int, default=23)
    parser.add_argument("--pre-ms", type=float, default=100.0)
    parser.add_argument("--post-ms", type=float, default=100.0)
    parser.add_argument("--fs-hz", type=float, default=None)
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--read-every-n", type=int, default=1)
    parser.add_argument("--rolling-trigger", type=int, default=6)
    parser.add_argument("--rolling-burst", type=int, default=100)
    parser.add_argument(
        "--window-anchor",
        choices=["trigger", "failed-burst-drop"],
        default="trigger",
        help="Anchor window on detected trigger (default) or plenum drop for failed-burst snippets",
    )
    parser.add_argument(
        "--failed-burst-min-rise-to-drop-ms",
        type=float,
        default=500.0,
        help="For failed-burst mode, minimum time between rise and drop events (ms)",
    )
    parser.add_argument(
        "--failed-burst-min-drop-to-rise-grad-ratio",
        type=float,
        default=0.5,
        help="For failed-burst mode, minimum |drop_grad| / rise_grad ratio",
    )
    parser.add_argument("--write-metadata-json", type=Path, default=None)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional path to save verification plot (plenum + trigger)",
    )
    parser.add_argument(
        "--plot-plenum-channel",
        default="PLEN-PT",
        help="Channel used for plenum trace in verification plot",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = create_lvm_fixture(
        input_path=args.input,
        output_path=args.output,
        trigger_channel=args.trigger_channel,
        burst_channel=args.burst_channel,
        header_row_index=args.header_row_index,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        fs_hz=args.fs_hz,
        decimate=args.decimate,
        read_every_n=args.read_every_n,
        rolling_trigger=args.rolling_trigger,
        rolling_burst=args.rolling_burst,
        window_anchor=args.window_anchor,
        failed_burst_min_rise_to_drop_ms=args.failed_burst_min_rise_to_drop_ms,
        failed_burst_min_drop_to_rise_grad_ratio=args.failed_burst_min_drop_to_rise_grad_ratio,
        metadata_json_path=args.write_metadata_json,
        plot_output_path=args.plot_output,
        plot_plenum_channel=args.plot_plenum_channel,
    )
    print(
        f"Wrote fixture to {args.output} | fs_hz={metadata['fs_hz']:.3f} | "
        f"trigger_index={metadata['trigger_index_in_fixture']}"
    )


if __name__ == "__main__":
    main()
