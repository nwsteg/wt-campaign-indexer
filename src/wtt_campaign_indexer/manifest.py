from __future__ import annotations

import importlib
import json
import math
import re
import struct
from collections import OrderedDict
from pathlib import Path

from pygasflow.atd.viscosity import viscosity_air_southerland
from pygasflow.isentropic import pressure_ratio, temperature_ratio

from wtt_campaign_indexer.discovery import FSTDiscovery, discover_campaign
from wtt_campaign_indexer.lvm_fixture import (
    infer_sample_rate_hz,
    pick_trigger_and_burst,
    read_lvm_data,
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

    The dummy campaign HCC fixtures use a `TC\x02\r` header where offset 76 stores
    acquisition frequency as a little-endian unsigned integer.
    """
    try:
        with hcc_path.open("rb") as handle:
            header = handle.read(80)
    except OSError:
        return None

    if len(header) < 80 or not header.startswith(b"TC\x02\r"):
        return None

    rate = _to_positive_float(struct.unpack_from("<I", header, 76)[0])
    if rate is None or rate > 1_000_000:
        return None
    return rate


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
        df = read_lvm_data(lvm_path, header_row_index=23)
        detection = pick_trigger_and_burst(df, trigger_channel="Voltage", burst_channel="PLEN-PT")
        if detection.burst_idx is None:
            raise ValueError("Burst index could not be detected.")
        fs_hz = infer_sample_rate_hz(df)
    except Exception as exc:
        notes.append(f"Unable to compute steady-state summary from LVM: {exc}")
        return OrderedDict((key, None) for key in _CONDITION_UNITS), notes

    start_idx = detection.burst_idx + int(round(_STEADY_STATE_START_MS * fs_hz / 1000.0))
    end_idx = detection.burst_idx + int(round(_STEADY_STATE_END_MS * fs_hz / 1000.0))
    start_idx = max(0, start_idx)
    end_idx = min(len(df) - 1, end_idx)

    if end_idx <= start_idx:
        notes.append("Steady-state window was empty after burst detection.")
        return OrderedDict((key, None) for key in _CONDITION_UNITS), notes

    window = df.iloc[start_idx : end_idx + 1]

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
        f"{_STEADY_STATE_START_MS:.0f}-{_STEADY_STATE_END_MS:.0f} ms after burst "
        f"(indices {start_idx}-{end_idx})."
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


def write_campaign_manifests(
    campaign_root: Path,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
) -> tuple[Path, ...]:
    discovery = discover_campaign(campaign_root)
    return tuple(
        write_fst_manifest(
            fst,
            tunnel_mach=tunnel_mach,
            jet_used=jet_used,
            jet_mach=jet_mach,
        )
        for fst in discovery.fsts
    )


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
) -> str:
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

    for fst in discovery.fsts:
        manifest = build_fst_manifest(
            fst,
            tunnel_mach=tunnel_mach,
            jet_used=jet_used,
            jet_mach=jet_mach,
        )
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
    lines.append("Notes:")
    lines.append(
        "- Top-level values are computed from each FST LVM in the 50-90 ms " "post-burst window."
    )
    lines.append(
        f"- pinf is computed from p0 using isentropic relations with tunnel M={tunnel_mach:.3g}."
    )
    lines.append(
        "- Re/m is computed from static freestream state derived from p0/T0 "
        f"at tunnel M={tunnel_mach:.3g}."
    )
    if jet_used and jet_mach is not None:
        lines.append(
            "- Jet enabled: pj and J are computed using isentropic relations "
            f"with jet M={jet_mach:.3g}."
        )
    else:
        lines.append("- Jet disabled: pj/pinf and J are omitted.")
    lines.append(
        "- Runs containing `scale` or `cal` in their IDs are marked as support runs "
        "in detailed sections and excluded when picking a primary rate for the "
        "overview table."
    )
    lines.append("")
    lines.append(f"FST count: **{len(discovery.fsts)}**")
    lines.append("")
    lines.append("## FST overview")
    lines.append("")
    lines.append("| FST | LVM | Diagnostic count | Run count |")
    lines.append("| --- | --- | ---: | ---: |")

    for fst in discovery.fsts:
        run_count = sum(len(diag.runs) for diag in fst.diagnostics)
        lvm_name = fst.primary_lvm.name if fst.primary_lvm else "(missing)"
        lines.append(
            f"| {fst.normalized_name} | {lvm_name} | {len(fst.diagnostics)} | {run_count} |"
        )

    for fst in discovery.fsts:
        manifest = build_fst_manifest(
            fst,
            tunnel_mach=tunnel_mach,
            jet_used=jet_used,
            jet_mach=jet_mach,
        )
        lines.append("")
        lines.append(f"## {fst.normalized_name}")
        lines.append("")

        if manifest["condition_notes"]:
            lines.append("### LVM condition notes")
            lines.append("")
            for note in manifest["condition_notes"]:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("### Diagnostics")
        lines.append("")
        lines.append("| Diagnostic | Known | Runs |")
        lines.append("| --- | :---: | ---: |")
        for diagnostic in manifest["diagnostics"]:
            known = "yes" if diagnostic["is_known"] else "no"
            lines.append(f"| {diagnostic['name']} | {known} | {diagnostic['run_count']} |")

        lines.append("")
        lines.append("### Runs")
        lines.append("")
        lines.append("| Diagnostic | Run ID | Support run | Inferred rate (Hz) | Notes | Errors |")
        lines.append("| --- | --- | :---: | ---: | --- | --- |")
        if manifest["runs"]:
            for run in manifest["runs"]:
                rate = "" if run["inferred_rate_hz"] is None else f"{run['inferred_rate_hz']:.3f}"
                notes = "; ".join(run["notes"]) if run["notes"] else ""
                errors = "; ".join(run["errors"]) if run["errors"] else ""
                support = "yes" if run["is_support_run"] else "no"
                run_row = [run["diagnostic"], run["run_id"], support, rate, notes, errors]
                lines.append("| " + " | ".join(run_row) + " |")
        else:
            lines.append("| (none) | (none) |  |  |  |  |")

    lines.append("")
    return "\n".join(lines)


def write_campaign_summary(
    campaign_root: Path,
    output_path: Path,
    tunnel_mach: float = _DEFAULT_TUNNEL_MACH,
    jet_used: bool = False,
    jet_mach: float | None = None,
) -> Path:
    output_path = Path(output_path)
    markdown = build_campaign_summary_markdown(
        campaign_root,
        tunnel_mach=tunnel_mach,
        jet_used=jet_used,
        jet_mach=jet_mach,
    )
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
