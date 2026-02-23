from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DetectionResult:
    trigger_idx: int
    burst_idx: int | None
    trigger_candidates: list[int]


def read_lvm_data(
    file_path: Path,
    header_row_index: int = 23,
    read_every_n: int = 1,
    usecols: list[int] | None = None,
) -> pd.DataFrame:
    """Read a typical FST LVM file using the same pattern as campaign scripts."""
    if read_every_n < 1:
        raise ValueError("read_every_n must be >= 1")

    def keep_row(row_index: int) -> bool:
        if row_index < header_row_index + 1:
            return False
        return (row_index - header_row_index) % read_every_n == 0

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        engine="c",
        na_values=[""],
        low_memory=True,
        on_bad_lines="warn",
        skiprows=lambda x: not keep_row(x),
        usecols=usecols,
    )

    headers = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        nrows=1,
        skiprows=header_row_index,
        usecols=usecols,
    )

    df.columns = [str(col).strip() for col in headers.iloc[0]]
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.interpolate(method="linear", inplace=True, limit_direction="both", axis=0)
    return df


def _rolling_gradient(values: pd.Series, window: int) -> np.ndarray:
    smoothed = values.rolling(window=max(window, 1), center=True, min_periods=1).mean()
    return np.gradient(smoothed.to_numpy(dtype=float))


def _top_peaks(
    gradient: np.ndarray, k: int, min_distance: int, positive_only: bool = False
) -> list[int]:
    scores = gradient.copy()
    if not positive_only:
        scores = np.abs(scores)
    else:
        scores[scores < 0] = 0

    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    peaks: list[int] = []
    for idx in order:
        if len(peaks) >= k:
            break
        if scores[idx] <= 0:
            break
        if any(abs(idx - chosen) < min_distance for chosen in peaks):
            continue
        peaks.append(idx)
    return sorted(peaks)


def pick_trigger_and_burst(
    df: pd.DataFrame,
    trigger_channel: str = "Voltage",
    burst_channel: str | None = "PLEN-PT",
    rolling_trigger: int = 6,
    rolling_burst: int = 100,
    top_k: int = 5,
) -> DetectionResult:
    if trigger_channel not in df.columns:
        raise ValueError(
            f"Trigger channel '{trigger_channel}' not found in columns: {list(df.columns)}"
        )

    trigger_grad = _rolling_gradient(df[trigger_channel], rolling_trigger)
    candidates = _top_peaks(trigger_grad, k=top_k, min_distance=max(rolling_trigger, 1))
    if not candidates:
        raise ValueError("No trigger candidates were found from trigger channel gradient.")

    burst_idx: int | None = None
    if burst_channel and burst_channel in df.columns:
        burst_grad = _rolling_gradient(df[burst_channel], rolling_burst)
        burst_idx = int(np.argmax(burst_grad))

    candidate_scores = sorted(candidates, key=lambda i: float(abs(trigger_grad[i])), reverse=True)

    if len(candidate_scores) == 1:
        chosen = candidate_scores[0]
    else:
        strongest = float(abs(trigger_grad[candidate_scores[0]]))
        second = float(abs(trigger_grad[candidate_scores[1]]))
        one_strong = strongest > 0 and second / strongest < 0.5

        if one_strong or burst_idx is None:
            chosen = candidate_scores[0]
        else:
            chosen = min(
                candidate_scores, key=lambda i: (abs(i - burst_idx), -abs(trigger_grad[i]), i)
            )

    return DetectionResult(
        trigger_idx=int(chosen), burst_idx=burst_idx, trigger_candidates=candidates
    )


def detect_header_row(lines: list[str], trigger_channel: str, burst_channel: str | None) -> int:
    wanted = {trigger_channel.strip().lower()}
    if burst_channel:
        wanted.add(burst_channel.strip().lower())

    for idx, line in enumerate(lines):
        if "\t" not in line:
            continue
        cols = [c.strip().lower() for c in line.rstrip("\n").split("\t")]
        if wanted.issubset(set(cols)):
            return idx
    raise ValueError("Unable to auto-detect the LVM header row from channel names.")


def infer_sample_rate_hz(df: pd.DataFrame) -> float:
    time_candidates = ["Time", "time", "X_Value", "x_value", "Seconds", "seconds"]
    for col in time_candidates:
        if col in df.columns:
            diffs = np.diff(df[col].to_numpy(dtype=float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if len(diffs) == 0:
                continue
            dt = float(np.median(diffs))
            if dt <= 0:
                continue
            return 1.0 / dt
    raise ValueError("Unable to infer sample rate from time column; provide --fs-hz explicitly.")


def _select_indices(
    start: int, end_exclusive: int, decimate: int, must_keep: set[int]
) -> list[int]:
    selected = list(range(start, end_exclusive, decimate))
    selected.extend(idx for idx in sorted(must_keep) if start <= idx < end_exclusive)
    return sorted(set(selected))


def write_verification_plot(
    fixture_df: pd.DataFrame,
    trigger_channel: str,
    plen_channel: str,
    trigger_idx_in_fixture: int,
    fs_hz: float,
    plot_path: Path,
) -> None:
    if trigger_channel not in fixture_df.columns:
        raise ValueError(f"Trigger channel '{trigger_channel}' not available in fixture data")
    if plen_channel not in fixture_df.columns:
        raise ValueError(f"Plenum channel '{plen_channel}' not available in fixture data")

    ms = (np.arange(len(fixture_df)) - trigger_idx_in_fixture) * 1000.0 / fs_hz

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ms, fixture_df[plen_channel], color="tab:blue", label=plen_channel)
    ax1.set_xlabel("Time from trigger (ms)")
    ax1.set_ylabel(f"{plen_channel}", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ms, fixture_df[trigger_channel], color="tab:red", label=trigger_channel)
    ax2.set_ylabel(f"{trigger_channel}", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax1.set_title("Fixture verification: plenum + trigger")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def create_lvm_fixture(
    input_path: Path,
    output_path: Path,
    trigger_channel: str = "Voltage",
    burst_channel: str | None = "PLEN-PT",
    header_row_index: int = 23,
    pre_ms: float = 100.0,
    post_ms: float = 100.0,
    fs_hz: float | None = None,
    decimate: int = 1,
    rolling_trigger: int = 6,
    rolling_burst: int = 100,
    read_every_n: int = 1,
    metadata_json_path: Path | None = None,
    plot_output_path: Path | None = None,
    plot_plenum_channel: str = "PLEN-PT",
) -> dict:
    if decimate < 1:
        raise ValueError("--decimate must be >= 1")

    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    if header_row_index >= len(lines):
        header_row_index = detect_header_row(
            lines, trigger_channel=trigger_channel, burst_channel=burst_channel
        )

    try:
        df = read_lvm_data(
            input_path,
            header_row_index=header_row_index,
            read_every_n=read_every_n,
        )
    except Exception:
        auto_idx = detect_header_row(
            lines, trigger_channel=trigger_channel, burst_channel=burst_channel
        )
        df = read_lvm_data(input_path, header_row_index=auto_idx, read_every_n=read_every_n)
        header_row_index = auto_idx
    else:
        if trigger_channel not in df.columns:
            auto_idx = detect_header_row(
                lines, trigger_channel=trigger_channel, burst_channel=burst_channel
            )
            if auto_idx != header_row_index:
                df = read_lvm_data(input_path, header_row_index=auto_idx, read_every_n=read_every_n)
                header_row_index = auto_idx

    detection = pick_trigger_and_burst(
        df,
        trigger_channel=trigger_channel,
        burst_channel=burst_channel,
        rolling_trigger=rolling_trigger,
        rolling_burst=rolling_burst,
    )

    inferred_fs_hz = infer_sample_rate_hz(df) * read_every_n
    fs_hz = fs_hz if fs_hz is not None else inferred_fs_hz
    effective_fs_hz = fs_hz / read_every_n

    pre_samples = int(round(pre_ms * effective_fs_hz / 1000.0))
    post_samples = int(round(post_ms * effective_fs_hz / 1000.0))

    start = max(0, detection.trigger_idx - pre_samples)
    end_exclusive = min(len(df), detection.trigger_idx + post_samples + 1)

    must_keep = {detection.trigger_idx}
    if detection.burst_idx is not None:
        must_keep.add(detection.burst_idx)
    indices = _select_indices(start, end_exclusive, decimate=decimate, must_keep=must_keep)

    fixture_df = df.iloc[indices].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        for line in lines[: header_row_index + 1]:
            file_obj.write(line if line.endswith("\n") else f"{line}\n")
        fixture_df.to_csv(file_obj, sep="\t", index=False, header=False, lineterminator="\n")

    trigger_fixture_idx = indices.index(detection.trigger_idx)
    burst_fixture_idx = (
        indices.index(detection.burst_idx) if detection.burst_idx in indices else None
    )

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "fs_hz": float(fs_hz),
        "effective_fs_hz": float(effective_fs_hz),
        "read_every_n": int(read_every_n),
        "trigger_index_in_fixture": int(trigger_fixture_idx),
        "burst_index_in_fixture": int(burst_fixture_idx) if burst_fixture_idx is not None else None,
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
        "decimation": int(decimate),
        "trigger_channel": trigger_channel,
        "burst_channel": burst_channel,
        "plot_plenum_channel": plot_plenum_channel,
        "trigger_candidates_input_indices": detection.trigger_candidates,
    }

    if metadata_json_path:
        metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if plot_output_path:
        write_verification_plot(
            fixture_df=fixture_df,
            trigger_channel=trigger_channel,
            plen_channel=plot_plenum_channel,
            trigger_idx_in_fixture=trigger_fixture_idx,
            fs_hz=effective_fs_hz,
            plot_path=plot_output_path,
        )

    return metadata
