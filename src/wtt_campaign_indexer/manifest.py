from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

from wtt_campaign_indexer.discovery import FSTDiscovery, discover_campaign

_CONDITION_UNITS = {
    "p0": "Pa",
    "T0": "K",
    "Re_1": "1/m",
    "p0j": "Pa",
    "p0j/p0": "1",
    "p0j/pinf": "1",
    "J": "1",
}

_RATE_PATTERNS = (
    re.compile(r"AcquisitionFrameRate[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"<FrameRate[^>]*>([0-9]+(?:\.[0-9]+)?)</FrameRate>", re.IGNORECASE),
)


def infer_rate_from_cihx(cihx_path: Path) -> float | None:
    text = cihx_path.read_text(encoding="utf-8", errors="replace")
    for pattern in _RATE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def find_lvm_fixture_path(fst: FSTDiscovery) -> Path | None:
    canonical = fst.path / f"{fst.normalized_name}_fixture.lvm"
    if canonical.exists():
        return canonical

    fixture_candidates = sorted(
        candidate
        for candidate in fst.path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() == ".lvm" and "fixture" in candidate.stem.lower()
    )
    if len(fixture_candidates) == 1:
        return fixture_candidates[0]

    return None


def build_fst_manifest(fst: FSTDiscovery) -> OrderedDict[str, object]:
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
            notes: list[str] = []
            errors: list[str] = []
            inferred_rate: float | None = None

            if cihx_path is None:
                errors.append("No .cihx file discovered for run.")
            else:
                inferred_rate = infer_rate_from_cihx(cihx_path)
                if inferred_rate is None:
                    notes.append("Unable to infer frame rate from .cihx metadata.")

            run_entries.append(
                OrderedDict(
                    [
                        ("run_id", run.name),
                        ("diagnostic", diagnostic.name),
                        ("cihx_path", str(cihx_path) if cihx_path else None),
                        ("inferred_rate_hz", inferred_rate),
                        ("notes", notes),
                        ("errors", errors),
                    ]
                )
            )

    fixture_path = find_lvm_fixture_path(fst)

    manifest = OrderedDict(
        [
            ("fst_id", fst.normalized_name),
            ("lvm_path", str(fst.primary_lvm) if fst.primary_lvm else None),
            ("lvm_fixture_path", str(fixture_path) if fixture_path else None),
            ("diagnostics", diagnostics),
            ("runs", run_entries),
            (
                "condition_summary",
                OrderedDict((key, None) for key in _CONDITION_UNITS),
            ),
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
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    return manifest


def write_fst_manifest(fst: FSTDiscovery) -> Path:
    manifest_path = fst.path / f"{fst.normalized_name}_manifest.json"
    manifest = build_fst_manifest(fst)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def write_campaign_manifests(campaign_root: Path) -> tuple[Path, ...]:
    discovery = discover_campaign(campaign_root)
    return tuple(write_fst_manifest(fst) for fst in discovery.fsts)
