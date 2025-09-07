import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
from enum import Enum, auto


class AnnotationType(Enum):
    POINT = auto()
    POINT_AND_LABEL = auto()
    POINTS = auto()
    POINTS_AND_LABELS = auto()
    CIRCLE = auto()
    CIRCLE_AND_LABEL = auto()
    RECTANGLE = auto()
    PITCH_YAW_ROLL = auto()
    LINE = auto()  # New line annotation type
    LINE_AND_LABEL = auto()  # New line with label annotation type
    MASK = auto()
    INFO_PANEL = auto()  # Info panel for dashboard-style information


@dataclass
class Annotation:
    type: AnnotationType
    coordinates: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None
    # color: Tuple[int, int, int] = (255, 0, 0)
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 255, 0))  # Default color to green
    labels: Optional[Union[str, List[str]]] = None
    radius: Optional[float] = None
    thickness: Optional[float] = None
    orientation: Optional[Tuple[float, float, float]] = None
    mask: Optional[np.ndarray] = None  # Adding mask field
    info_panel: Optional['InfoPanel'] = None  # Adding info panel field

    def __post_init__(self):
        if self.type in {AnnotationType.POINTS_AND_LABELS, AnnotationType.POINT_AND_LABEL} and isinstance(self.labels, list):
            if len(self.coordinates) != len(self.labels):
                raise ValueError("The number of labels must match the number of coordinates.")
        if self.type == AnnotationType.PITCH_YAW_ROLL and self.orientation is None:
            raise ValueError("Orientation (pitch, yaw, roll) must be provided for 'pitch_yaw_roll' type.")