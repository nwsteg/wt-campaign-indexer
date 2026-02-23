from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KNOWN_DIAGNOSTICS = ("schlieren", "plif", "pls", "tracking", "piv", "kulite", "ir")
_FST_PATTERN = re.compile(r"^FST_?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RunDiscovery:
    name: str
    path: Path
    cihx_files: tuple[Path, ...]
    hcc_files: tuple[Path, ...]


@dataclass(frozen=True)
class DiagnosticDiscovery:
    name: str
    path: Path
    is_known: bool
    runs: tuple[RunDiscovery, ...]


@dataclass(frozen=True)
class FSTDiscovery:
    number: int
    normalized_name: str
    path: Path
    primary_lvm: Path | None
    diagnostics: tuple[DiagnosticDiscovery, ...]


@dataclass(frozen=True)
class CampaignDiscovery:
    fsts: tuple[FSTDiscovery, ...]


def normalize_fst_name(name: str) -> str | None:
    match = _FST_PATTERN.match(name)
    if not match:
        return None
    fst_number = int(match.group(1))
    return f"FST_{fst_number:04d}"


def discover_campaign(campaign_root: Path) -> CampaignDiscovery:
    campaign_root = Path(campaign_root)
    fst_entries: list[FSTDiscovery] = []

    for child in campaign_root.iterdir():
        if not child.is_dir():
            continue
        normalized = normalize_fst_name(child.name)
        if normalized is None:
            continue

        fst_number = int(normalized.split("_")[1])
        fst_entries.append(
            FSTDiscovery(
                number=fst_number,
                normalized_name=normalized,
                path=child,
                primary_lvm=_pick_primary_lvm(child, fst_number),
                diagnostics=_discover_diagnostics(child),
            )
        )

    fst_entries.sort(key=lambda entry: entry.number)
    return CampaignDiscovery(fsts=tuple(fst_entries))


def _pick_primary_lvm(fst_dir: Path, fst_number: int) -> Path | None:
    lvm_candidates = sorted(
        candidate
        for candidate in fst_dir.iterdir()
        if candidate.is_file() and candidate.suffix == ".lvm"
    )
    if not lvm_candidates:
        return None

    preferred_name = f"FST_{fst_number:04d}.lvm"
    for candidate in lvm_candidates:
        if candidate.name == preferred_name:
            return candidate

    return lvm_candidates[0]


def _discover_diagnostics(fst_dir: Path) -> tuple[DiagnosticDiscovery, ...]:
    diagnostics: list[DiagnosticDiscovery] = []
    known = set(KNOWN_DIAGNOSTICS)

    for child in sorted(entry for entry in fst_dir.iterdir() if entry.is_dir()):
        diagnostic_name = child.name.lower()
        is_known = diagnostic_name in known
        runs = _discover_runs(child)

        diagnostics.append(
            DiagnosticDiscovery(
                name=diagnostic_name,
                path=child,
                is_known=is_known,
                runs=runs,
            )
        )

    diagnostics.sort(key=lambda diagnostic: diagnostic.name)
    return tuple(diagnostics)


def _discover_runs(diagnostic_dir: Path) -> tuple[RunDiscovery, ...]:
    runs: list[RunDiscovery] = []

    for child in sorted(entry for entry in diagnostic_dir.rglob("*") if entry.is_dir()):
        cihx_files = tuple(
            sorted(
                candidate
                for candidate in child.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".cihx"
            )
        )
        hcc_files = tuple(
            sorted(
                candidate
                for candidate in child.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".hcc"
            )
        )
        if not cihx_files and not hcc_files:
            continue

        run_name = str(child.relative_to(diagnostic_dir))
        runs.append(
            RunDiscovery(name=run_name, path=child, cihx_files=cihx_files, hcc_files=hcc_files)
        )

    runs.sort(key=lambda run: run.name)
    return tuple(runs)
