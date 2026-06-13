"""Unified group evidence schema for LiDAR and DA3 branches."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GroupEvidence:
    group_id: str
    stable_gaussian_ids: List[int]
    view_ids: List[str]
    region_ids: List[str]
    risk_source: str
    supervision_mode: str
    counterfactual_objective: str
    group_label: str
    future_action: str
    confidence: float
    risk_weighted_contribution: float = 0.0
    mean_talpha: float = 0.0
    max_talpha: float = 0.0
    support_pixels: int = 0
    multiview_support: int = 1
    rgb_safety_score: float = 0.0
    rgb_delta: float = 0.0
    lidar_error_delta: Optional[float] = None
    da3_structure_delta: Optional[float] = None
    edge_delta: Optional[float] = None
    ranking_delta: Optional[float] = None
    side_delta: Optional[float] = None
    is_protected: bool = False
    is_low_evidence: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
