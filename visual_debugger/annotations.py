"""
Type-specific annotation classes for Visual Debugger.
Each annotation type has its own class with only relevant parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .info_panel import InfoPanel


# Base class for all annotations
class BaseAnnotation(ABC):
    """Base class for all annotation types"""
    
    @abstractmethod
    def accept(self, processor):
        """Visitor pattern for processing"""
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Returns the bounding box of this annotation as (x_min, y_min, x_max, y_max).
        This includes all visual elements the annotation will draw.
        """
        pass


# =============================================================================
# Point Annotations
# =============================================================================

@dataclass
class PointAnnotation(BaseAnnotation):
    """A single point on the image"""
    position: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    
    def accept(self, processor):
        return processor.render_point(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        x, y = self.position
        radius = self.size // 2
        return (x - radius, y - radius, x + radius, y + radius)


@dataclass
class LabeledPointAnnotation(BaseAnnotation):
    """A point with a text label"""
    position: Tuple[int, int]
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
    
    def accept(self, processor):
        return processor.render_labeled_point(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        import cv2
        x, y = self.position
        radius = self.size // 2
        
        # Get text size to include label in bounding box
        (text_w, text_h), baseline = cv2.getTextSize(
            self.label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        # Text is typically drawn to the right of the point
        x_min = x - radius
        y_min = y - radius - text_h  # Text might be above
        x_max = x + radius + text_w + 10  # 10px spacing
        y_max = y + radius + baseline
        
        return (x_min, y_min, x_max, y_max)


@dataclass
class PointsAnnotation(BaseAnnotation):
    """Multiple points with the same style"""
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    
    def accept(self, processor):
        return processor.render_points(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        if not self.points:
            return (0, 0, 0, 0)
        
        radius = self.size // 2
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        return (
            min(xs) - radius,
            min(ys) - radius,
            max(xs) + radius,
            max(ys) + radius
        )


@dataclass
class LabeledPointsAnnotation(BaseAnnotation):
    """Multiple points with individual labels"""
    points: List[Tuple[int, int]]
    labels: List[str]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
    
    def __post_init__(self):
        if len(self.points) != len(self.labels):
            raise ValueError(f"Number of points ({len(self.points)}) must match number of labels ({len(self.labels)})")
    
    def accept(self, processor):
        return processor.render_labeled_points(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        if not self.points:
            return (0, 0, 0, 0)
        
        import cv2
        radius = self.size // 2
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        x_min = min(xs) - radius
        y_min = min(ys) - radius
        x_max = max(xs) + radius
        y_max = max(ys) + radius
        
        # Account for labels
        for label in self.labels:
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            x_max = max(x_max, max(xs) + text_w + 10)
            y_max = max(y_max, max(ys) + text_h + baseline)
        
        return (x_min, y_min, x_max, y_max)


# =============================================================================
# Shape Annotations
# =============================================================================

@dataclass
class CircleAnnotation(BaseAnnotation):
    """A circle outline or filled circle"""
    center: Tuple[int, int]
    radius: int
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
    
    def accept(self, processor):
        return processor.render_circle(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        x, y = self.center
        # Account for thickness on the outside of the circle
        effective_radius = self.radius + (self.thickness // 2 if not self.filled else 0)
        return (
            x - effective_radius,
            y - effective_radius,
            x + effective_radius,
            y + effective_radius
        )


@dataclass
class LabeledCircleAnnotation(BaseAnnotation):
    """A circle with a text label"""
    center: Tuple[int, int]
    radius: int
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
    font_scale: float = 0.5
    font_thickness: int = 1
    
    def accept(self, processor):
        return processor.render_labeled_circle(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        import cv2
        x, y = self.center
        effective_radius = self.radius + (self.thickness // 2 if not self.filled else 0)
        
        # Circle bounds
        x_min = x - effective_radius
        y_min = y - effective_radius
        x_max = x + effective_radius
        y_max = y + effective_radius
        
        # Account for label (typically below circle)
        (text_w, text_h), baseline = cv2.getTextSize(
            self.label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        text_x = x - text_w // 2
        text_y = y + self.radius + text_h + 5
        
        x_min = min(x_min, text_x)
        x_max = max(x_max, text_x + text_w)
        y_max = max(y_max, text_y)
        
        return (x_min, y_min, x_max, y_max)


@dataclass
class RectangleAnnotation(BaseAnnotation):
    """A rectangle defined by two corners"""
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
    
    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int, **kwargs):
        """Create rectangle from x, y, width, height format"""
        return cls(
            top_left=(x, y),
            bottom_right=(x + width, y + height),
            **kwargs
        )
    
    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]
    
    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]
    
    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.top_left[0] + self.bottom_right[0]) // 2,
            (self.top_left[1] + self.bottom_right[1]) // 2
        )
    
    def accept(self, processor):
        return processor.render_rectangle(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # Account for thickness
        half_thickness = self.thickness // 2 if not self.filled else 0
        return (
            self.top_left[0] - half_thickness,
            self.top_left[1] - half_thickness,
            self.bottom_right[0] + half_thickness,
            self.bottom_right[1] + half_thickness
        )


@dataclass
class AlphaRectangleAnnotation(BaseAnnotation):
    """A rectangle with alpha transparency support"""
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.7  # Transparency: 0.0 (transparent) to 1.0 (opaque)
    
    def accept(self, processor):
        return processor.render_alpha_rectangle(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        return (
            self.top_left[0],
            self.top_left[1],
            self.bottom_right[0],
            self.bottom_right[1]
        )


# =============================================================================
# Line Annotations
# =============================================================================

@dataclass
class LineAnnotation(BaseAnnotation):
    """A line between two points"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    arrow: bool = False  # If True, add arrowhead at end
    
    def accept(self, processor):
        return processor.render_line(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        x1, y1 = self.start
        x2, y2 = self.end
        
        # Account for line thickness
        half_thickness = self.thickness // 2
        
        # If arrow, add extra space for arrowhead (typically ~10 pixels)
        arrow_buffer = 10 if self.arrow else 0
        
        return (
            min(x1, x2) - half_thickness - arrow_buffer,
            min(y1, y2) - half_thickness - arrow_buffer,
            max(x1, x2) + half_thickness + arrow_buffer,
            max(y1, y2) + half_thickness + arrow_buffer
        )


@dataclass
class LabeledLineAnnotation(BaseAnnotation):
    """A line with a text label at midpoint"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    arrow: bool = False
    font_scale: float = 0.5
    font_thickness: int = 1
    
    @property
    def midpoint(self) -> Tuple[int, int]:
        return (
            (self.start[0] + self.end[0]) // 2,
            (self.start[1] + self.end[1]) // 2
        )
    
    def accept(self, processor):
        return processor.render_labeled_line(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        import cv2
        x1, y1 = self.start
        x2, y2 = self.end
        
        # Line bounds
        half_thickness = self.thickness // 2
        arrow_buffer = 10 if self.arrow else 0
        
        x_min = min(x1, x2) - half_thickness - arrow_buffer
        y_min = min(y1, y2) - half_thickness - arrow_buffer
        x_max = max(x1, x2) + half_thickness + arrow_buffer
        y_max = max(y1, y2) + half_thickness + arrow_buffer
        
        # Account for label at midpoint
        (text_w, text_h), baseline = cv2.getTextSize(
            self.label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
        )
        
        mid_x, mid_y = self.midpoint
        text_x = mid_x - text_w // 2
        text_y = mid_y - 10  # Label typically above line
        
        x_min = min(x_min, text_x)
        y_min = min(y_min, text_y - text_h)
        x_max = max(x_max, text_x + text_w)
        y_max = max(y_max, text_y + baseline)
        
        return (x_min, y_min, x_max, y_max)


@dataclass
class PolylineAnnotation(BaseAnnotation):
    """Multiple connected lines"""
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    closed: bool = False  # If True, connect last point to first
    
    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError("Polyline must have at least 2 points")
    
    def accept(self, processor):
        return processor.render_polyline(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        if not self.points:
            return (0, 0, 0, 0)
        
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        
        half_thickness = self.thickness // 2
        
        return (
            min(xs) - half_thickness,
            min(ys) - half_thickness,
            max(xs) + half_thickness,
            max(ys) + half_thickness
        )


# =============================================================================
# Text Annotations
# =============================================================================

@dataclass
class TextAnnotation(BaseAnnotation):
    """Text at a specific position"""
    text: str
    position: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    font_scale: float = 0.5
    font_thickness: int = 1
    background_color: Optional[Tuple[int, int, int]] = None
    background_padding: int = 5
    
    def accept(self, processor):
        return processor.render_text(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        import cv2
        x, y = self.position
        
        # Calculate text size
        lines = self.text.split('\n')
        max_width = 0
        total_height = 0
        
        for line in lines:
            (w, h), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            max_width = max(max_width, w)
            total_height += h + baseline
        
        # Add background padding if applicable
        if self.background_color:
            padding = self.background_padding
            return (
                x - padding,
                y - total_height - padding,
                x + max_width + padding,
                y + padding
            )
        else:
            return (x, y - total_height, x + max_width, y)


# =============================================================================
# Complex Annotations
# =============================================================================

@dataclass
class BoundingBoxAnnotation(BaseAnnotation):
    """A bounding box with optional label"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    color: Tuple[int, int, int] = (0, 255, 0)
    label: Optional[str] = None
    thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    show_label_background: bool = True
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.bbox[0], self.bbox[1])
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3])
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.bbox[2] // 2, self.bbox[1] + self.bbox[3] // 2)
    
    def get_label_text(self) -> str:
        """Get formatted label text"""
        return self.label if self.label else ""
    
    def accept(self, processor):
        return processor.render_bounding_box(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        import cv2
        x, y, w, h = self.bbox
        
        # Basic box bounds
        half_thickness = self.thickness // 2
        x_min = x - half_thickness
        y_min = y - half_thickness
        x_max = x + w + half_thickness
        y_max = y + h + half_thickness
        
        # Account for label if present
        if self.label:
            label_text = self.get_label_text()
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            
            # Label is typically above the box
            label_y = y - 5
            if self.show_label_background:
                label_y -= 5  # Extra padding for background
            
            y_min = min(y_min, label_y - text_h)
            x_max = max(x_max, x + text_w)
        
        return (x_min, y_min, x_max, y_max)


@dataclass
class MaskAnnotation(BaseAnnotation):
    """Segmentation mask overlay"""
    mask: np.ndarray
    color: Tuple[int, int, int] = (0, 255, 0)
    alpha: float = 0.5
    colormap: Optional[str] = None  # 'random', 'jet', 'hot', or None for class colors
    
    def __post_init__(self):
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        if self.mask.ndim != 2:
            raise ValueError("Mask must be a 2D array")
        if not 0 <= self.alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
    
    def accept(self, processor):
        return processor.render_mask(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # Mask covers the entire image area where it's applied
        # The actual bounds depend on where the mask has non-zero values
        h, w = self.mask.shape
        
        # Find non-zero regions
        non_zero = np.where(self.mask > 0)
        if len(non_zero[0]) == 0:
            return (0, 0, 0, 0)
        
        return (
            int(np.min(non_zero[1])),  # min x
            int(np.min(non_zero[0])),  # min y
            int(np.max(non_zero[1])),  # max x
            int(np.max(non_zero[0]))   # max y
        )


@dataclass
class OrientationAnnotation(BaseAnnotation):
    """3D orientation visualization (pitch, yaw, roll)"""
    position: Tuple[int, int]
    pitch: float  # degrees
    yaw: float    # degrees
    roll: float   # degrees
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 100
    x_color: Tuple[int, int, int] = (255, 0, 0)  # Red for X axis
    y_color: Tuple[int, int, int] = (0, 255, 0)  # Green for Y axis
    z_color: Tuple[int, int, int] = (0, 0, 255)  # Blue for Z axis
    
    def accept(self, processor):
        return processor.render_orientation(self)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # Orientation axes extend from center position
        x, y = self.position
        # The size determines how far the axes extend
        extent = self.size
        
        return (
            x - extent,
            y - extent,
            x + extent,
            y + extent
        )


@dataclass
class InfoPanelAnnotation(BaseAnnotation):
    """
    Info panel for dashboard-style information display.
    This is a composite annotation that decomposes into basic annotations.
    """
    panel: 'InfoPanel'
    
    def get_component_annotations(self, image_shape: Tuple[int, int]) -> List[BaseAnnotation]:
        """
        Decompose the info panel into basic annotations (rectangles and text).
        
        Args:
            image_shape: (height, width) of the target image
            
        Returns:
            List of basic annotations that make up the panel
        """
        annotations = []
        
        # Get panel dimensions and position
        panel_w, panel_h = self.panel.calculate_dimensions()
        x, y = self.panel.calculate_position(image_shape, panel_w, panel_h)
        
        # Ensure panel fits within image bounds
        x = max(0, min(x, image_shape[1] - panel_w))
        y = max(0, min(y, image_shape[0] - panel_h))
        
        # Add semi-transparent background rectangle
        if self.panel.style.show_background:
            # For alpha transparency, we create a special AlphaRectangleAnnotation
            annotations.append(
                AlphaRectangleAnnotation(
                    top_left=(x, y),
                    bottom_right=(x + panel_w, y + panel_h),
                    color=self.panel.style.background_color,
                    alpha=self.panel.style.background_alpha
                )
            )
        
        # Add border if specified
        if self.panel.style.border_color:
            annotations.append(
                RectangleAnnotation(
                    top_left=(x, y),
                    bottom_right=(x + panel_w, y + panel_h),
                    color=self.panel.style.border_color,
                    filled=False,
                    thickness=self.panel.style.border_thickness
                )
            )
        
        # Current Y position for text
        import cv2
        font = self.panel.style.font
        current_y = y + self.panel.style.padding
        
        # Add title if present
        if self.panel.title:
            title_scale = self.panel.style.font_scale * 1.2
            # Calculate title height
            (_, title_h), _ = cv2.getTextSize(
                self.panel.title, font, title_scale, self.panel.style.font_thickness
            )
            current_y += title_h  # Move down by title height
            
            annotations.append(
                TextAnnotation(
                    text=self.panel.title,
                    position=(x + self.panel.style.padding, current_y),
                    color=self.panel.style.title_color,
                    font_scale=title_scale,
                    font_thickness=self.panel.style.font_thickness
                )
            )
            current_y += self.panel.style.line_spacing * 2  # Extra space after title
        
        # Add each entry
        for key, value in self.panel._entries:
            # Format text based on entry type
            if isinstance(value, str):
                text = value
            else:
                text = str(value)
            
            # Check if this is a separator
            if key is None and all(c in '─━═' for c in text):
                # Add a line (using a thin rectangle)
                separator_y = current_y + self.panel.style.line_spacing // 2
                annotations.append(
                    RectangleAnnotation(
                        top_left=(x + self.panel.style.padding, separator_y),
                        bottom_right=(x + panel_w - self.panel.style.padding, separator_y + 1),
                        color=self.panel.style.text_color,
                        filled=True,
                        thickness=0
                    )
                )
                current_y += self.panel.style.line_spacing * 2
            else:
                # Regular text entry
                if key:
                    text = f"{key}: {text}"
                
                # Calculate text height
                (_, text_h), _ = cv2.getTextSize(
                    text, font, self.panel.style.font_scale, self.panel.style.font_thickness
                )
                current_y += text_h  # Move down by text height
                
                annotations.append(
                    TextAnnotation(
                        text=text,
                        position=(x + self.panel.style.padding, current_y),
                        color=self.panel.style.text_color,
                        font_scale=self.panel.style.font_scale,
                        font_thickness=self.panel.style.font_thickness
                    )
                )
                
                current_y += self.panel.style.line_spacing  # Add line spacing
        
        return annotations
    
    def accept(self, processor):
        """
        Render the panel by decomposing it into basic annotations.
        We need the image shape, so we'll get it from the processor's image.
        """
        if hasattr(processor, 'image'):
            image_shape = processor.image.shape[:2]
            for ann in self.get_component_annotations(image_shape):
                ann.accept(processor)
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        # We need image dimensions to calculate exact position
        # For now, return panel dimensions at origin
        if hasattr(self.panel, 'calculate_dimensions'):
            width, height = self.panel.calculate_dimensions()
            return (0, 0, width, height)
        return (0, 0, 200, 100)  # Default size


# =============================================================================
# Convenience Factory Functions
# =============================================================================

def point(x: int, y: int, **kwargs) -> PointAnnotation:
    """Create a point annotation"""
    return PointAnnotation(position=(x, y), **kwargs)


def labeled_point(x: int, y: int, label: str, **kwargs) -> LabeledPointAnnotation:
    """Create a labeled point annotation"""
    return LabeledPointAnnotation(position=(x, y), label=label, **kwargs)


def circle(x: int, y: int, radius: int, **kwargs) -> CircleAnnotation:
    """Create a circle annotation"""
    return CircleAnnotation(center=(x, y), radius=radius, **kwargs)


def rectangle(x: int, y: int, width: int, height: int, **kwargs) -> RectangleAnnotation:
    """Create a rectangle annotation from x, y, width, height"""
    return RectangleAnnotation.from_xywh(x, y, width, height, **kwargs)


def line(x1: int, y1: int, x2: int, y2: int, **kwargs) -> LineAnnotation:
    """Create a line annotation"""
    return LineAnnotation(start=(x1, y1), end=(x2, y2), **kwargs)


def text(msg: str, x: int, y: int, **kwargs) -> TextAnnotation:
    """Create a text annotation"""
    return TextAnnotation(text=msg, position=(x, y), **kwargs)


def bbox(x: int, y: int, w: int, h: int, label: Optional[str] = None, 
         **kwargs) -> BoundingBoxAnnotation:
    """Create a bounding box annotation"""
    return BoundingBoxAnnotation(bbox=(x, y, w, h), label=label, **kwargs)


# =============================================================================
# Migration Support
# =============================================================================

def from_old_annotation(old_ann) -> BaseAnnotation:
    """
    Convert old-style Annotation to new type-specific annotation.
    This helps with migration from the old system.
    """
    from .annotations import AnnotationType
    
    # Map old types to new classes
    type_map = {
        AnnotationType.POINT: lambda a: PointAnnotation(
            position=a.coordinates,
            color=a.color,
            size=a.radius if a.radius else 5
        ),
        AnnotationType.POINT_AND_LABEL: lambda a: LabeledPointAnnotation(
            position=a.coordinates,
            label=a.labels,
            color=a.color,
            size=a.radius if a.radius else 5
        ),
        AnnotationType.POINTS: lambda a: PointsAnnotation(
            points=a.coordinates if isinstance(a.coordinates, list) else [a.coordinates],
            color=a.color,
            size=a.radius if a.radius else 5
        ),
        AnnotationType.POINTS_AND_LABELS: lambda a: LabeledPointsAnnotation(
            points=a.coordinates,
            labels=a.labels if isinstance(a.labels, list) else [a.labels],
            color=a.color
        ),
        AnnotationType.CIRCLE: lambda a: CircleAnnotation(
            center=a.coordinates,
            radius=a.radius if a.radius else 5,
            thickness=a.thickness if a.thickness else 2,
            color=a.color
        ),
        AnnotationType.CIRCLE_AND_LABEL: lambda a: LabeledCircleAnnotation(
            center=a.coordinates,
            radius=a.radius if a.radius else 5,
            label=a.labels,
            thickness=a.thickness if a.thickness else 2,
            color=a.color
        ),
        AnnotationType.RECTANGLE: lambda a: RectangleAnnotation.from_xywh(
            *a.coordinates,  # Assumes x, y, w, h format
            color=a.color,
            thickness=a.thickness if a.thickness else 2
        ),
        AnnotationType.LINE: lambda a: LineAnnotation(
            start=a.coordinates[0],
            end=a.coordinates[1],
            color=a.color,
            thickness=a.thickness if a.thickness else 2
        ),
        AnnotationType.LINE_AND_LABEL: lambda a: LabeledLineAnnotation(
            start=a.coordinates[0],
            end=a.coordinates[1],
            label=a.labels,
            color=a.color,
            thickness=a.thickness if a.thickness else 2
        ),
        AnnotationType.MASK: lambda a: MaskAnnotation(
            mask=a.mask,
            color=a.color
        ),
        AnnotationType.PITCH_YAW_ROLL: lambda a: OrientationAnnotation(
            position=a.coordinates if a.coordinates else (0, 0),
            pitch=a.orientation[0],
            yaw=a.orientation[1],
            roll=a.orientation[2],
            color=a.color
        ),
        AnnotationType.INFO_PANEL: lambda a: InfoPanelAnnotation(
            panel=a.info_panel
        )
    }
    
    converter = type_map.get(old_ann.type)
    if converter:
        return converter(old_ann)
    else:
        raise ValueError(f"Unknown annotation type: {old_ann.type}")