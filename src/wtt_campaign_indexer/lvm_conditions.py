from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

from pygasflow.atd.viscosity import viscosity_air_southerland
from pygasflow.isentropic import pressure_ratio, temperature_ratio

from wtt_campaign_indexer.lvm_fixture import (
    detect_event_index_coarse,
    detect_header_row_from_file,
    infer_sample_rate_hz,
    pick_trigger_and_burst,
    read_lvm_data,
    read_lvm_data_window,
)

CONDITION_UNITS = OrderedDict(
    [
        ("p0", "psia"),
        ("T0", "K"),
        ("Re_1", "1/m"),
        ("p0j", "psia"),
        ("T0j", "K"),
        ("pinf", "psia"),
        ("pinf_ref_jet_mach", "psia"),
        ("pj", "psia"),
        ("p0j/pinf", "1"),
        ("pj/pinf", "1"),
        ("J", "1"),
    ]
)

BASIC_SCHLIEREN_METADATA_UNITS = OrderedDict(
    [
        ("p0", "psia"),
        ("T0", "K"),
        ("Reinf", "1/m"),
    ]
)

STEADY_STATE_START_MS = 50.0
STEADY_STATE_END_MS = 90.0
ASSUMED_T0J_K = 300.0
DEFAULT_TUNNEL_MACH = 7.2


def read_lvm_event_window(
    lvm_path: Path,
    header_row_index: int,
    start_ms: float,
    end_ms: float,
    require_burst_confidence: bool,
    fallback_notes: list[str],
) -> tuple[object, int, int, float, str]:
    """Read a bounded event-relative LVM window around the detected burst/trigger."""
    coarse = detect_event_index_coarse(
        lvm_path,
        header_row_index=header_row_index,
        trigger_channel="Voltage",
        burst_channel="PLEN-PT",
        read_every_n=10,
    )

    if coarse.confidence_ok and coarse.anchor_idx is not None and coarse.sample_rate_hz is not None:
        fs_hz = coarse.sample_rate_hz
        start_idx = max(0, coarse.anchor_idx + int(round(start_ms * fs_hz / 1000.0)))
        end_idx = coarse.anchor_idx + int(round(end_ms * fs_hz / 1000.0))
        if end_idx <= start_idx:
            raise ValueError("Event window was empty after coarse detection.")

        pad_ms = 30.0
        pad_samples = int(round(pad_ms * fs_hz / 1000.0))
        bounded_start = max(0, start_idx - pad_samples)
        bounded_end = end_idx + pad_samples

        window_df = read_lvm_data_window(
            lvm_path,
            header_row_index=header_row_index,
            start_data_row=bounded_start,
            end_data_row=bounded_end,
        )
        if window_df.empty:
            fallback_notes.append(
                "Coarse event window read returned empty; using full read fallback."
            )
        else:
            rel_start = max(0, start_idx - bounded_start)
            rel_end = min(len(window_df) - 1, end_idx - bounded_start)
            if rel_end > rel_start:
                return (
                    window_df.iloc[rel_start : rel_end + 1].copy(),
                    start_idx,
                    coarse.anchor_idx,
                    fs_hz,
                    "burst" if coarse.burst_idx is not None else "trigger",
                )
    if require_burst_confidence:
        fallback_notes.append(
            f"Coarse detection confidence insufficient ({coarse.reason}); using full read fallback."
        )

    data = read_lvm_data(lvm_path, header_row_index=header_row_index)
    detection = pick_trigger_and_burst(data, trigger_channel="Voltage", burst_channel="PLEN-PT")
    anchor = detection.burst_idx
    label = "burst"
    if anchor is None:
        if require_burst_confidence:
            raise ValueError("Burst index could not be detected.")
        anchor = detection.trigger_idx
        label = "trigger"
    fs_hz = infer_sample_rate_hz(data)
    start_idx = max(0, anchor + int(round(start_ms * fs_hz / 1000.0)))
    end_idx = min(len(data) - 1, anchor + int(round(end_ms * fs_hz / 1000.0)))
    if end_idx <= start_idx:
        raise ValueError("Event window was empty after anchor detection.")
    return data.iloc[start_idx : end_idx + 1].copy(), start_idx, anchor, fs_hz, label


def _safe_mean(df, channel: str) -> float | None:
    if channel not in df.columns:
        return None
    value = float(df[channel].mean())
    if math.isnan(value):
        return None
    return value


def _select_channel(df, *candidates: str) -> float | None:
    for candidate in candidates:
        value = _safe_mean(df, candidate)
        if value is not None:
            return value
    return None


def _compute_reynolds_per_meter(p0_psia: float, t0_k: float, mach: float) -> float:
    p0_pa = p0_psia * 6894.757293168
    gamma = 1.4
    gas_constant = 287.05

    p_static = p0_pa * float(pressure_ratio(mach))
    t_static = t0_k * float(temperature_ratio(mach))

    rho_static = p_static / (gas_constant * t_static)
    a_static = math.sqrt(gamma * gas_constant * t_static)
    u_static = mach * a_static
    mu_static = float(viscosity_air_southerland(t_static))

    return rho_static * u_static / mu_static


def _isentropic_static_pressure(p0_psia: float, mach: float) -> float:
    return p0_psia * float(pressure_ratio(mach))


def compute_steady_state_conditions(
    lvm_path: Path,
    tunnel_mach: float = DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
) -> tuple[OrderedDict[str, float | None], list[str]]:
    notes: list[str] = []
    try:
        try:
            header_row_index = detect_header_row_from_file(
                lvm_path,
                trigger_channel="Voltage",
                burst_channel="PLEN-PT",
            )
        except ValueError:
            header_row_index = 23

        window, start_idx, anchor_idx, _fs_hz, anchor_label = read_lvm_event_window(
            lvm_path,
            header_row_index=header_row_index,
            start_ms=STEADY_STATE_START_MS,
            end_ms=STEADY_STATE_END_MS,
            require_burst_confidence=True,
            fallback_notes=notes,
        )
    except Exception as exc:
        notes.append(f"Unable to compute steady-state summary from LVM: {exc}")
        return OrderedDict((key, None) for key in CONDITION_UNITS), notes

    p0 = _select_channel(window, "PLEN-PT")
    t0 = _select_channel(window, "TC 8")
    p0j = _select_channel(window, "LVDT")
    t0j = ASSUMED_T0J_K

    if p0 is None:
        notes.append("Channel 'PLEN-PT' not found; p0 unavailable.")
    if t0 is None:
        notes.append("Channel 'TC 8' not found; T0 unavailable.")
    if p0j is None:
        notes.append("Channel 'LVDT' not found; p0j unavailable.")

    reynolds = None
    pinf = None
    pinf_ref_jet_mach = None
    if p0 is not None and t0 is not None and t0 > 0:
        reynolds = _compute_reynolds_per_meter(p0_psia=p0, t0_k=t0, mach=tunnel_mach)
        pinf = _isentropic_static_pressure(p0_psia=p0, mach=tunnel_mach)
        if jet_mach is not None:
            pinf_ref_jet_mach = _isentropic_static_pressure(p0_psia=p0, mach=jet_mach)

    pj = None
    p0j_over_pinf = None
    pj_over_pinf = None
    j_value = None

    if pinf is not None and p0j is not None and pinf > 0:
        p0j_over_pinf = p0j / pinf

    if jet_used:
        if jet_mach is None:
            notes.append("Jet marked as used but jet_mach missing; pj/pinf and J unavailable.")
        elif p0j is not None and pinf is not None and pinf > 0:
            pj = _isentropic_static_pressure(p0_psia=p0j, mach=jet_mach)
            pj_over_pinf = pj / pinf
            j_value = pj_over_pinf * (jet_mach / tunnel_mach) ** 2

    notes.append(
        "Steady-state window: "
        f"{STEADY_STATE_START_MS:.0f}-{STEADY_STATE_END_MS:.0f} ms after {anchor_label} "
        f"(indices {start_idx}-{start_idx + len(window) - 1}, anchor {anchor_idx})."
    )
    notes.append(f"T0j assumed constant at {ASSUMED_T0J_K:.0f} K.")
    notes.append(f"pinf computed from p0 using isentropic relation at M={tunnel_mach:.3g}.")
    if jet_mach is not None:
        notes.append(f"Reference pinf at jet Mach from p0 uses M={jet_mach:.3g}.")
    if jet_used and jet_mach is not None:
        notes.append(f"pj computed from p0j using isentropic relation at jet M={jet_mach:.3g}.")

    return (
        OrderedDict(
            [
                ("p0", p0),
                ("T0", t0),
                ("Re_1", reynolds),
                ("p0j", p0j),
                ("T0j", t0j),
                ("pinf", pinf),
                ("pinf_ref_jet_mach", pinf_ref_jet_mach),
                ("pj", pj),
                ("p0j/pinf", p0j_over_pinf),
                ("pj/pinf", pj_over_pinf),
                ("J", j_value),
            ]
        ),
        notes,
    )


def compute_basic_schlieren_metadata(
    lvm_path: Path,
    tunnel_mach: float = DEFAULT_TUNNEL_MACH,
) -> tuple[OrderedDict[str, float | None], list[str]]:
    """Return the subset most likely to travel with schlieren data products."""
    summary, notes = compute_steady_state_conditions(
        lvm_path,
        tunnel_mach=tunnel_mach,
    )
    return (
        OrderedDict(
            [
                ("p0", summary["p0"]),
                ("T0", summary["T0"]),
                ("Reinf", summary["Re_1"]),
            ]
        ),
        notes,
    )


__all__ = [
    "ASSUMED_T0J_K",
    "BASIC_SCHLIEREN_METADATA_UNITS",
    "CONDITION_UNITS",
    "DEFAULT_TUNNEL_MACH",
    "STEADY_STATE_END_MS",
    "STEADY_STATE_START_MS",
    "compute_basic_schlieren_metadata",
    "compute_steady_state_conditions",
    "read_lvm_event_window",
]
