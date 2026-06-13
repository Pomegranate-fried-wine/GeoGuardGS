"""Dry-run repair candidate schema."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RepairCandidate:
    stable_gaussian_id: int
    operation: str
    group_id: str
    confidence: float
    support_pixels: int
    dryrun_only: bool = True
    will_modify_parameters: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self):
        payload = dict(self.__dict__)
        if payload["metadata"] is None:
            payload["metadata"] = {}
        return payload
