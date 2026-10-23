"""Loader for the bundled Kubernetes Serverless Workload Dataset (KSWD).

The KSWD ships with this repository under ``dataset/`` and contains full
14-day traces (20,160 minutes) per workload, so the classification and
simulation pipelines can run without downloading the Azure Functions 2019
dataset from Kaggle.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ARCHETYPES = ["spike", "periodic", "ramp", "stationary_noisy"]


class KSWDLoader:
    """Load workload traces from the bundled KSWD dataset."""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.workload_dir = self.dataset_dir / "workloads"
        self.archetype_dir = self.dataset_dir / "archetypes"

        if not self.workload_dir.exists():
            raise FileNotFoundError(
                f"KSWD workloads not found at {self.workload_dir}. "
                "Run this script from the repository root."
            )

    def list_workload_ids(self) -> List[str]:
        """Return every workload id present on disk, sorted."""
        return sorted(p.stem for p in self.workload_dir.glob("w_*.json"))

    def load_workload(self, workload_id: str) -> Dict:
        """Load a single workload record."""
        path = self.workload_dir / f"{workload_id}.json"
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def load_invocation_data(
        self,
        max_workloads: Optional[int] = None,
        days: Optional[List[int]] = None,
        balanced: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Load traces as ``{workload_id: invocations_per_minute}``.

        Args:
            max_workloads: Cap on how many workloads to load (None = all).
            days: 1-based day numbers to keep (1-14). None keeps all 14 days.
            balanced: Draw evenly from each archetype so every class is
                represented even for small ``max_workloads``.

        Returns:
            Dictionary mapping workload id to a 1-D float array of
            invocations per minute.
        """
        ids = self._select_ids(max_workloads, balanced)

        traces: Dict[str, np.ndarray] = {}
        for workload_id in ids:
            record = self.load_workload(workload_id)
            series = np.asarray(
                record["request_trace"]["requests_per_minute"], dtype=float
            )
            if days is not None:
                series = self._slice_days(series, days)
            traces[workload_id] = series

        logger.info(f"Loaded {len(traces)} KSWD workloads")
        return traces

    def load_archetype_labels(self, workload_ids: Optional[List[str]] = None) -> Dict[str, str]:
        """Return ground-truth archetype per workload, upper-cased."""
        ids = workload_ids if workload_ids is not None else self.list_workload_ids()
        return {
            workload_id: self.load_workload(workload_id)["archetype"].upper()
            for workload_id in ids
        }

    def load_metadata(self, workload_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Return resource requirements and SLO targets per workload."""
        ids = workload_ids if workload_ids is not None else self.list_workload_ids()
        metadata = {}
        for workload_id in ids:
            record = self.load_workload(workload_id)
            metadata[workload_id] = {
                "archetype": record["archetype"].upper(),
                "archetype_confidence": record["archetype_confidence"],
                "resource_requirements": record["resource_requirements"],
                "slo_targets": record["slo_targets"],
                "features": record.get("features", {}),
            }
        return metadata

    def _select_ids(self, max_workloads: Optional[int], balanced: bool) -> List[str]:
        all_ids = self.list_workload_ids()
        if max_workloads is None or max_workloads >= len(all_ids):
            return all_ids
        if not balanced:
            return all_ids[:max_workloads]

        by_archetype = self._archetype_index()
        if not by_archetype:
            return all_ids[:max_workloads]

        selected: List[str] = []
        per_class = max(1, max_workloads // len(by_archetype))
        for archetype in sorted(by_archetype):
            selected.extend(by_archetype[archetype][:per_class])

        # Top up from the front of the full list if rounding left us short.
        for workload_id in all_ids:
            if len(selected) >= max_workloads:
                break
            if workload_id not in selected:
                selected.append(workload_id)

        return sorted(selected)[:max_workloads]

    def _archetype_index(self) -> Dict[str, List[str]]:
        """Read ``dataset/archetypes/*.json`` groupings if present."""
        index: Dict[str, List[str]] = {}
        for archetype in ARCHETYPES:
            path = self.archetype_dir / f"{archetype}.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            ids = payload.get("workload_ids", payload) if isinstance(payload, dict) else payload
            if isinstance(ids, list) and ids:
                index[archetype] = [
                    entry if isinstance(entry, str) else entry.get("workload_id")
                    for entry in ids
                ]
        return {k: [i for i in v if i] for k, v in index.items() if v}

    @staticmethod
    def _slice_days(series: np.ndarray, days: List[int]) -> np.ndarray:
        """Keep only the requested 1-based days from a per-minute trace."""
        chunks = []
        for day in days:
            start = (day - 1) * 1440
            end = start + 1440
            if start < len(series):
                chunks.append(series[start:end])
        return np.concatenate(chunks) if chunks else series[:0]
