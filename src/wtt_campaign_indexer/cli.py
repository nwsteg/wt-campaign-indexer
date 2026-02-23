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
    parser.add_argument("--pre-ms", type=float, default=200.0)
    parser.add_argument("--post-ms", type=float, default=300.0)
    parser.add_argument("--fs-hz", type=float, default=None)
    parser.add_argument("--decimate", type=int, default=1)
    parser.add_argument("--rolling-trigger", type=int, default=6)
    parser.add_argument("--rolling-burst", type=int, default=100)
    parser.add_argument("--write-metadata-json", type=Path, default=None)
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
        rolling_trigger=args.rolling_trigger,
        rolling_burst=args.rolling_burst,
        metadata_json_path=args.write_metadata_json,
    )
    print(
        f"Wrote fixture to {args.output} | fs_hz={metadata['fs_hz']:.3f} | "
        f"trigger_index={metadata['trigger_index_in_fixture']}"
    )


if __name__ == "__main__":
    main()
