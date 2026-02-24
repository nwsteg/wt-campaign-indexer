from __future__ import annotations

import importlib
import json
import math
import re
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
from pygasflow.atd.viscosity import viscosity_air_southerland

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygasflow.isentropic import pressure_ratio, temperature_ratio

from wtt_campaign_indexer.discovery import FSTDiscovery, discover_campaign
from wtt_campaign_indexer.lvm_fixture import (
    detect_event_index_coarse,
    detect_header_row_from_file,
    infer_sample_rate_hz,
    pick_trigger_and_burst,
    read_lvm_data,
    read_lvm_data_window,
)

_CONDITION_UNITS = {
    "p0": "psia",
    "T0": "K",
    "Re_1": "1/m",
    "p0j": "psia",
    "T0j": "K",
    "pinf": "psia",
    "pinf_ref_jet_mach": "psia",
    "pj": "psia",
    "p0j/pinf": "1",
    "pj/pinf": "1",
    "J": "1",
}

_CIHX_RATE_PATTERNS = (
    re.compile(r"AcquisitionFrameRate[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"<FrameRate[^>]*>([0-9]+(?:\.[0-9]+)?)</FrameRate>", re.IGNORECASE),
    re.compile(r"recordRate[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)

_HCC_RATE_PATTERNS = (
    re.compile(r"(?:frame(?:_|\s*)rate|framerate|fps)[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)

_STEADY_STATE_START_MS = 50.0
_STEADY_STATE_END_MS = 90.0
_ASSUMED_T0J_K = 300.0
_DEFAULT_TUNNEL_MACH = 7.2
_SUMMARY_PLOT_START_MS = -20.0
_SUMMARY_PLOT_END_MS = 120.0


def _read_lvm_event_window(
    lvm_path: Path,
    header_row_index: int,
    start_ms: float,
    end_ms: float,
    require_burst_confidence: bool,
    fallback_notes: list[str],
) -> tuple[object, int, int, float, str]:
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
            fallback_notes.append("Coarse event window read returned empty; using full read fallback.")
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
        fallback_notes.append(f"Coarse detection confidence insufficient ({coarse.reason}); using full read fallback.")

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


def infer_rate_from_cihx(cihx_path: Path) -> float | None:
    text = cihx_path.read_text(encoding="utf-8", errors="replace")
    for pattern in _CIHX_RATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _to_positive_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(rate) or rate <= 0:
        return None
    return rate


def _infer_rate_from_telops_hcc(hcc_path: Path) -> float | None:
    """Try Telops-native readers first when available."""
    for module_name in ("telops_hcc", "telops_hccpy", "telops"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue

        # Common style: module.open(path) -> object with a frame-rate attribute.
        open_fn = getattr(module, "open", None)
        if callable(open_fn):
            try:
                opened = open_fn(str(hcc_path))
            except Exception:
                opened = None

            if opened is not None:
                for attr in (
                    "frame_rate",
                    "framerate",
                    "fps",
                    "acquisition_frame_rate",
                    "record_rate",
                ):
                    rate = _to_positive_float(getattr(opened, attr, None))
                    if rate is not None:
                        close_fn = getattr(opened, "close", None)
                        if callable(close_fn):
                            close_fn()
                        return rate

                close_fn = getattr(opened, "close", None)
                if callable(close_fn):
                    close_fn()

        # Common style: module.Reader(path) / module.HccFile(path).
        for cls_name in ("Reader", "HccFile", "HCCFile", "HccReader"):
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            try:
                reader = cls(str(hcc_path))
            except Exception:
                continue

            for attr in (
                "frame_rate",
                "framerate",
                "fps",
                "acquisition_frame_rate",
                "record_rate",
            ):
                rate = _to_positive_float(getattr(reader, attr, None))
                if rate is not None:
                    close_fn = getattr(reader, "close", None)
                    if callable(close_fn):
                        close_fn()
                    return rate

            close_fn = getattr(reader, "close", None)
            if callable(close_fn):
                close_fn()

    return None


def _infer_rate_from_av(hcc_path: Path) -> float | None:
    try:
        av = importlib.import_module("av")
    except ImportError:
        return None

    try:
        container = av.open(str(hcc_path))
    except Exception:
        return None

    try:
        for stream in container.streams:
            for attr in ("average_rate", "base_rate", "guessed_rate", "rate"):
                rate = _to_positive_float(getattr(stream, attr, None))
                if rate is not None:
                    return rate
    finally:
        container.close()

    return None


def _infer_rate_from_imageio(hcc_path: Path) -> float | None:
    try:
        imageio_v3 = importlib.import_module("imageio.v3")
    except ImportError:
        return None

    try:
        metadata = imageio_v3.immeta(str(hcc_path))
    except Exception:
        return None

    if not isinstance(metadata, dict):
        return None

    for key in ("fps", "frame_rate", "framerate", "record_rate"):
        rate = _to_positive_float(metadata.get(key))
        if rate is not None:
            return rate

    return None


def infer_rate_from_hcc(hcc_path: Path) -> float | None:
    """Infer HCC frame rate from available readers, then a Telops header fallback."""
    for parser in (
        _infer_rate_from_telops_hcc,
        _infer_rate_from_av,
        _infer_rate_from_imageio,
        _infer_rate_from_telops_header,
    ):
        rate = parser(hcc_path)
        if rate is not None:
            return rate
    return None


def _infer_rate_from_telops_header(hcc_path: Path) -> float | None:
    """Fallback for Telops HCC files when optional reader deps are unavailable.

    The dummy campaign Telops fixtures store acquisition frequency in milli-Hz at
    offset 44. Some files also include an integer at offset 76, but in practice
    that value may represent another field, so the milli-Hz header value is
    preferred when available.
    """
    try:
        with hcc_path.open("rb") as handle:
            header = handle.read(80)
    except OSError:
        return None

    if len(header) < 80 or not header.startswith(b"TC\x02\r"):
        return None

    milli_hz_rate = _to_positive_float(struct.unpack_from("<I", header, 44)[0])
    if milli_hz_rate is not None:
        rate_hz = milli_hz_rate / 1000.0
        if 0 < rate_hz <= 1_000_000:
            return rate_hz

    fallback_rate = _to_positive_float(struct.unpack_from("<I", header, 76)[0])
    if fallback_rate is None or fallback_rate > 1_000_000:
        return None
    return fallback_rate


def _looks_like_support_run(run_id: str) -> bool:
    lowered = run_id.lower()
    return "scale" in lowered or "cal" in lowered


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
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
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

        window, start_idx, anchor_idx, _fs_hz, anchor_label = _read_lvm_event_window(
            lvm_path,
            header_row_index=header_row_index,
            start_ms=_STEADY_STATE_START_MS,
            end_ms=_STEADY_STATE_END_MS,
            require_burst_confidence=True,
            fallback_notes=notes,
        )
    except Exception as exc:
        notes.append(f"Unable to compute steady-state summary from LVM: {exc}")
        return OrderedDict((key, None) for key in _CONDITION_UNITS), notes

    p0 = _select_channel(window, "PLEN-PT")
    t0 = _select_channel(window, "TC 8")
    p0j = _select_channel(window, "LVDT")
    t0j = _ASSUMED_T0J_K

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
        f"{_STEADY_STATE_START_MS:.0f}-{_STEADY_STATE_END_MS:.0f} ms after {anchor_label} "
        f"(indices {start_idx}-{start_idx + len(window) - 1}, anchor {anchor_idx})."
    )
    notes.append(f"T0j assumed constant at {_ASSUMED_T0J_K:.0f} K.")
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


def find_lvm_fixture_path(fst: FSTDiscovery) -> Path | None:
    canonical = fst.path / f"{fst.normalized_name}_fixture.lvm"
    if canonical.exists():
        return canonical

    fixture_candidates = sorted(
        candidate
        for candidate in fst.path.iterdir()
        if (
            candidate.is_file()
            and candidate.suffix.lower() == ".lvm"
            and "fixture" in candidate.stem.lower()
        )
    )
    if len(fixture_candidates) == 1:
        return fixture_candidates[0]

    return None


def build_fst_manifest(
    fst: FSTDiscovery,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
) -> OrderedDict[str, object]:
    diagnostics = [
        OrderedDict(
            [
                ("name", diagnostic.name),
                ("is_known", diagnostic.is_known),
                ("run_count", len(diagnostic.runs)),
            ]
        )
        for diagnostic in fst.diagnostics
    ]

    run_entries: list[OrderedDict[str, object]] = []
    for diagnostic in fst.diagnostics:
        for run in diagnostic.runs:
            cihx_path = run.cihx_files[0] if run.cihx_files else None
            hcc_path = run.hcc_files[0] if run.hcc_files else None
            notes: list[str] = []
            errors: list[str] = []
            inferred_rate: float | None = None

            if cihx_path is not None:
                inferred_rate = infer_rate_from_cihx(cihx_path)
                if inferred_rate is None:
                    notes.append("Unable to infer frame rate from .cihx metadata.")
            elif hcc_path is not None:
                inferred_rate = infer_rate_from_hcc(hcc_path)
                if inferred_rate is None:
                    notes.append("Unable to infer frame rate from .hcc metadata.")
            else:
                errors.append("No .cihx or .hcc file discovered for run.")
            if _looks_like_support_run(run.name):
                notes.append("Support run (scale/cal); not treated as primary flow data.")

            run_entries.append(
                OrderedDict(
                    [
                        ("run_id", run.name),
                        ("diagnostic", diagnostic.name),
                        ("cihx_path", str(cihx_path) if cihx_path else None),
                        ("hcc_path", str(hcc_path) if hcc_path else None),
                        ("inferred_rate_hz", inferred_rate),
                        ("is_support_run", _looks_like_support_run(run.name)),
                        ("notes", notes),
                        ("errors", errors),
                    ]
                )
            )

    fixture_path = find_lvm_fixture_path(fst)
    condition_summary = OrderedDict((key, None) for key in _CONDITION_UNITS)
    condition_notes: list[str] = []
    if fst.primary_lvm:
        condition_summary, condition_notes = compute_steady_state_conditions(
            fst.primary_lvm,
            tunnel_mach=tunnel_mach,
            jet_used=jet_used,
            jet_mach=jet_mach,
        )

    manifest = OrderedDict(
        [
            ("fst_id", fst.normalized_name),
            ("lvm_path", str(fst.primary_lvm) if fst.primary_lvm else None),
            ("lvm_fixture_path", str(fixture_path) if fixture_path else None),
            ("tunnel_mach", tunnel_mach),
            ("jet_used", jet_used),
            ("jet_mach", jet_mach),
            ("diagnostics", diagnostics),
            ("runs", run_entries),
            ("condition_summary", condition_summary),
            ("condition_notes", condition_notes),
            (
                "units",
                OrderedDict(
                    [
                        (
                            "condition_summary",
                            OrderedDict((key, unit) for key, unit in _CONDITION_UNITS.items()),
                        ),
                        (
                            "runs",
                            OrderedDict(
                                [
                                    ("inferred_rate_hz", "Hz"),
                                    ("run_id", None),
                                    ("cihx_path", None),
                                    ("hcc_path", None),
                                    ("is_support_run", None),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    return manifest


def write_fst_manifest(
    fst: FSTDiscovery,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
) -> Path:
    manifest_path = fst.path / f"{fst.normalized_name}_manifest.json"
    manifest = build_fst_manifest(
        fst,
        tunnel_mach=tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _latest_fst_input_mtime(fst: FSTDiscovery) -> float:
    mtimes = [fst.path.stat().st_mtime]
    if fst.primary_lvm is not None and fst.primary_lvm.exists():
        mtimes.append(fst.primary_lvm.stat().st_mtime)

    for diagnostic in fst.diagnostics:
        if diagnostic.path.exists():
            mtimes.append(diagnostic.path.stat().st_mtime)
        for run in diagnostic.runs:
            if run.path.exists():
                mtimes.append(run.path.stat().st_mtime)
            for cihx_path in run.cihx_files:
                if cihx_path.exists():
                    mtimes.append(cihx_path.stat().st_mtime)
            for hcc_path in run.hcc_files:
                if hcc_path.exists():
                    mtimes.append(hcc_path.stat().st_mtime)

    return max(mtimes)


def _load_reusable_manifest(
    fst: FSTDiscovery,
    tunnel_mach: float,
    jet_used: bool,
    jet_mach: float | None,
) -> OrderedDict[str, object] | None:
    manifest_path = fst.path / f"{fst.normalized_name}_manifest.json"
    if not manifest_path.exists():
        return None

    if manifest_path.stat().st_mtime < _latest_fst_input_mtime(fst):
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if payload.get("tunnel_mach") != tunnel_mach:
        return None
    if payload.get("jet_used") != jet_used:
        return None
    if payload.get("jet_mach") != jet_mach:
        return None

    return payload


def write_campaign_manifests(
    campaign_root: Path,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
    reprocess_all: bool = False,
) -> tuple[Path, ...]:
    if progress_callback is not None:
        progress_callback("Discovering campaign folders...")
    discovery = discover_campaign(campaign_root)
    manifest_paths: list[Path] = []
    for fst in discovery.fsts:
        manifest_path = fst.path / f"{fst.normalized_name}_manifest.json"
        reusable_manifest = (
            None
            if reprocess_all
            else _load_reusable_manifest(
                fst,
                tunnel_mach=tunnel_mach,
                jet_used=jet_used,
                jet_mach=jet_mach,
            )
        )
        if reusable_manifest is not None:
            if progress_callback is not None:
                progress_callback(f"Reusing manifest for {fst.normalized_name}...")
            manifest_paths.append(manifest_path)
            continue

        if progress_callback is not None:
            progress_callback(f"Processing {fst.normalized_name}...")
        manifest_paths.append(
            write_fst_manifest(
                fst,
                tunnel_mach=tunnel_mach,
                jet_used=jet_used,
                jet_mach=jet_mach,
            )
        )
    return tuple(manifest_paths)


def _format_rate_khz(rate_hz: float | None) -> str:
    if rate_hz is None:
        return ""
    return f"{rate_hz / 1000.0:.3f}"


def _format_numeric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _pick_primary_actual_run(
    manifest: OrderedDict[str, object], diagnostic_name: str
) -> dict | None:
    for run in manifest["runs"]:
        if run["diagnostic"] != diagnostic_name:
            continue
        if run["is_support_run"]:
            continue
        return run
    return None


def build_campaign_summary_markdown(
    campaign_root: Path,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
    reprocess_all: bool = False,
    figure_dir: Path | None = None,
) -> str:
    if progress_callback is not None:
        progress_callback("Discovering campaign folders...")
    discovery = discover_campaign(campaign_root)
    lines: list[str] = []
    lines.append("# Dummy campaign summary")
    lines.append("")
    lines.append(f"Campaign root: `{Path(campaign_root)}`")
    lines.append("")
    lines.append("## Top-level overview (steady-state, 50-90 ms after burst)")
    lines.append("")
    lines.append(
        "| FST | Diagnostic | Rate (kHz) | p0 (psia) | T0 (K) | Re/m x 10^-6 (1/m) | "
        "p0j (psia) | T0j (K) | p0j/pinf | pj/pinf | J |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    manifests_by_fst: dict[str, OrderedDict[str, object]] = {}
    for fst in discovery.fsts:
        manifest = (
            None
            if reprocess_all
            else _load_reusable_manifest(
                fst,
                tunnel_mach=tunnel_mach,
                jet_used=jet_used,
                jet_mach=jet_mach,
            )
        )
        if manifest is not None:
            if progress_callback is not None:
                progress_callback(f"Reusing manifest for {fst.normalized_name} in summary...")
        else:
            if progress_callback is not None:
                progress_callback(f"Processing {fst.normalized_name}...")
            manifest = build_fst_manifest(
                fst,
                tunnel_mach=tunnel_mach,
                jet_used=jet_used,
                jet_mach=jet_mach,
            )
        manifests_by_fst[fst.normalized_name] = manifest
        summary = manifest["condition_summary"]

        if manifest["diagnostics"]:
            for diagnostic in manifest["diagnostics"]:
                primary_run = _pick_primary_actual_run(manifest, diagnostic["name"])
                rate_khz = _format_rate_khz(
                    primary_run["inferred_rate_hz"] if primary_run is not None else None
                )
                reynolds_millions = (
                    summary["Re_1"] / 1_000_000.0 if summary["Re_1"] is not None else None
                )
                row = [
                    manifest["fst_id"],
                    diagnostic["name"],
                    rate_khz,
                    _format_numeric(summary["p0"], 1),
                    _format_numeric(summary["T0"], 1),
                    _format_numeric(reynolds_millions, 2),
                    _format_numeric(summary["p0j"], 1),
                    _format_numeric(summary["T0j"], 1),
                    _format_numeric(summary["p0j/pinf"], 2),
                    _format_numeric(summary["pj/pinf"], 2),
                    _format_numeric(summary["J"], 2),
                ]
                lines.append("| " + " | ".join(row) + " |")
        else:
            lines.append(f"| {manifest['fst_id']} | (none) |  |  |  |  |  |  |  |  |  |")

    lines.append("")
    lines.append("## FST traces")
    lines.append("")
    lines.append(
        "Plots show trigger voltage and plenum pressure from -20 ms to +120 ms "
        "relative to burst (or trigger fallback when burst is unavailable)."
    )
    if jet_used:
        lines.append("When available, jet pressure (`LVDT`) is included on the same plot.")
    lines.append("")

    for fst in discovery.fsts:
        lines.append(f"### {fst.normalized_name}")
        lines.append("")
        if figure_dir is None:
            lines.append("_Plot directory not provided._")
        else:
            plot_path = figure_dir / f"{fst.normalized_name}_overview.png"
            if plot_path.exists():
                rel = plot_path.relative_to(figure_dir.parent)
                lines.append(f"![{fst.normalized_name} trace]({rel.as_posix()})")
            else:
                lines.append("_Plot unavailable for this FST._")
        lines.append("")

    return "\n".join(lines)


def _write_fst_summary_plot(
    df_window,
    anchor_idx: int,
    anchor_label: str,
    start_idx: int,
    fs_hz: float,
    output_path: Path,
    fst_id: str,
    jet_used: bool,
) -> None:
    if "Voltage" not in df_window.columns:
        raise ValueError("Trigger channel 'Voltage' not found in LVM data.")
    if "PLEN-PT" not in df_window.columns:
        raise ValueError("Plenum channel 'PLEN-PT' not found in LVM data.")

    indices = np.arange(start_idx, start_idx + len(df_window))
    ms = (indices - anchor_idx) * 1000.0 / fs_hz

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ms, df_window["PLEN-PT"], color="tab:blue", label="PLEN-PT")
    if jet_used and "LVDT" in df_window.columns:
        ax1.plot(ms, df_window["LVDT"], color="tab:green", label="LVDT (jet)")
    ax1.set_xlabel(f"Time from {anchor_label} (ms)")
    ax1.set_ylabel("PLEN-PT", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ms, df_window["Voltage"], color="tab:red", label="Voltage")
    ax2.set_ylabel("Voltage", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.axvline(0.0, color="k", linestyle="--", linewidth=1)
    ax1.set_title(f"{fst_id}: trigger + plenum around {anchor_label}")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_campaign_summary_figures(
    campaign_root: Path,
    summary_output_path: Path,
    progress_callback: Callable[[str], None] | None = None,
    reprocess_all: bool = False,
    jet_used: bool = False,
) -> Path:
    figure_dir = Path(summary_output_path).parent / "campaign_summary_figs"
    figure_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_campaign(campaign_root)
    for fst in discovery.fsts:
        if fst.primary_lvm is None:
            if progress_callback is not None:
                progress_callback(f"Skipping plot for {fst.normalized_name}: missing LVM.")
            continue

        output_path = figure_dir / f"{fst.normalized_name}_overview.png"
        if not reprocess_all and output_path.exists() and fst.primary_lvm is not None:
            if output_path.stat().st_mtime >= fst.primary_lvm.stat().st_mtime:
                if progress_callback is not None:
                    progress_callback(f"Reusing plot for {fst.normalized_name}...")
                continue

        try:
            try:
                header_row_index = detect_header_row_from_file(
                    fst.primary_lvm,
                    trigger_channel="Voltage",
                    burst_channel="PLEN-PT",
                )
            except ValueError:
                header_row_index = 23

            local_notes: list[str] = []
            window, start_idx, anchor_idx, fs_hz, anchor_label = _read_lvm_event_window(
                fst.primary_lvm,
                header_row_index=header_row_index,
                start_ms=_SUMMARY_PLOT_START_MS,
                end_ms=_SUMMARY_PLOT_END_MS,
                require_burst_confidence=False,
                fallback_notes=local_notes,
            )
            _write_fst_summary_plot(
                window,
                anchor_idx=anchor_idx,
                anchor_label=anchor_label,
                start_idx=start_idx,
                fs_hz=fs_hz,
                output_path=output_path,
                fst_id=fst.normalized_name,
                jet_used=jet_used,
            )
            for note in local_notes:
                if progress_callback is not None:
                    progress_callback(f"{fst.normalized_name}: {note}")
        except Exception as exc:
            if progress_callback is not None:
                progress_callback(f"Skipping plot for {fst.normalized_name}: {exc}")

    return figure_dir


def write_campaign_summary(
    campaign_root: Path,
    output_path: Path,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
    reprocess_all: bool = False,
) -> Path:
    output_path = Path(output_path)
    figure_dir = write_campaign_summary_figures(
        campaign_root,
        output_path,
        progress_callback=progress_callback,
        reprocess_all=reprocess_all,
        jet_used=jet_used,
    )
    markdown = build_campaign_summary_markdown(
        campaign_root,
        tunnel_mach=tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
        progress_callback=progress_callback,
        reprocess_all=reprocess_all,
        figure_dir=figure_dir,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
