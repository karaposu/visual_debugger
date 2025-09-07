from .visual_debugger import VisualDebugger

# Import new type-specific annotations
from .annotations import (
    # Base class
    BaseAnnotation,
    
    # Point annotations
    PointAnnotation, LabeledPointAnnotation, PointsAnnotation, LabeledPointsAnnotation,
    
    # Shape annotations
    CircleAnnotation, LabeledCircleAnnotation, RectangleAnnotation, AlphaRectangleAnnotation,
    
    # Line annotations
    LineAnnotation, LabeledLineAnnotation, PolylineAnnotation,
    
    # Text and complex annotations
    TextAnnotation, BoundingBoxAnnotation, MaskAnnotation, OrientationAnnotation, InfoPanelAnnotation,
    
    # Factory functions
    point, labeled_point, circle, rectangle, line, text, bbox
)

# Import image processor and compositor
from .image_processor import ImageProcessor
from .composition import ImageCompositor, CompositionStyle, ImageEntry, LayoutDirection
from .info_panel import InfoPanel, PanelPosition, PanelStyle

# For backward compatibility (if needed)
try:
    from .old_annotations import Annotation, AnnotationType
except ImportError:
    pass

__all__ = [
    'VisualDebugger',
    'ImageProcessor',
    'ImageCompositor', 'CompositionStyle', 'ImageEntry', 'LayoutDirection',
    'InfoPanel', 'PanelPosition', 'PanelStyle',
    
    # Annotations
    'BaseAnnotation',
    'PointAnnotation', 'LabeledPointAnnotation', 'PointsAnnotation', 'LabeledPointsAnnotation',
    'CircleAnnotation', 'LabeledCircleAnnotation', 'RectangleAnnotation',
    'LineAnnotation', 'LabeledLineAnnotation', 'PolylineAnnotation',
    'TextAnnotation', 'BoundingBoxAnnotation', 'MaskAnnotation', 'OrientationAnnotation', 'InfoPanelAnnotation',
    
    # Factory functions
    'point', 'labeled_point', 'circle', 'rectangle', 'line', 'text', 'bbox'
]