# Annotation Type Classes Refactor

## The Problem with Current Design

Currently, all annotations use a single `Annotation` class with optional fields:

```python
# Current problematic design
@dataclass
class Annotation:
    type: AnnotationType
    coordinates: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None
    color: Tuple[int, int, int] = (0, 255, 0)
    labels: Optional[Union[str, List[str]]] = None
    radius: Optional[float] = None
    thickness: Optional[float] = None
    orientation: Optional[Tuple[float, float, float]] = None
    mask: Optional[np.ndarray] = None
    info_panel: Optional['InfoPanel'] = None
```

### Problems:

1. **Unclear Requirements**: Which fields are required for which type?
2. **Invalid States Possible**: Can create a CIRCLE with orientation or MASK with radius
3. **Poor IDE Support**: Autocomplete shows all fields even when irrelevant
4. **Runtime Errors**: Mistakes only discovered when rendering fails
5. **Complex Validation**: Need post_init checks for each type combination

## The Solution: Type-Specific Classes

Each annotation type gets its own class with only the relevant fields:

```python
# annotations_v2.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

# Base class for all annotations
@dataclass
class BaseAnnotation(ABC):
    """Base class for all annotation types"""
    color: Tuple[int, int, int] = (0, 255, 0)
    
    @abstractmethod
    def accept(self, processor):
        """Visitor pattern for processing"""
        pass

# Simple Annotations

@dataclass
class PointAnnotation(BaseAnnotation):
    """A single point on the image"""
    position: Tuple[int, int]
    size: int = 5
    
    def accept(self, processor):
        return processor.render_point(self)

@dataclass
class LabeledPointAnnotation(BaseAnnotation):
    """A point with a text label"""
    position: Tuple[int, int]
    label: str
    size: int = 5
    font_scale: float = 0.5
    
    def accept(self, processor):
        return processor.render_labeled_point(self)

@dataclass
class CircleAnnotation(BaseAnnotation):
    """A circle outline"""
    center: Tuple[int, int]
    radius: int
    thickness: int = 2
    filled: bool = False
    
    def accept(self, processor):
        return processor.render_circle(self)

@dataclass
class RectangleAnnotation(BaseAnnotation):
    """A rectangle"""
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    thickness: int = 2
    filled: bool = False
    
    # Alternative constructor for x,y,w,h format
    @classmethod
    def from_xywh(cls, x: int, y: int, width: int, height: int, **kwargs):
        return cls(
            top_left=(x, y),
            bottom_right=(x + width, y + height),
            **kwargs
        )
    
    def accept(self, processor):
        return processor.render_rectangle(self)

@dataclass
class LineAnnotation(BaseAnnotation):
    """A line between two points"""
    start: Tuple[int, int]
    end: Tuple[int, int]
    thickness: int = 2
    arrow: bool = False  # If True, add arrowhead at end
    
    def accept(self, processor):
        return processor.render_line(self)

@dataclass
class PolylineAnnotation(BaseAnnotation):
    """Multiple connected lines"""
    points: List[Tuple[int, int]]
    thickness: int = 2
    closed: bool = False  # If True, connect last point to first
    
    def accept(self, processor):
        return processor.render_polyline(self)

# Complex Annotations

@dataclass
class TextAnnotation(BaseAnnotation):
    """Text at a specific position"""
    text: str
    position: Tuple[int, int]
    font_scale: float = 0.5
    thickness: int = 1
    background: Optional[Tuple[int, int, int]] = None
    padding: int = 5
    
    def accept(self, processor):
        return processor.render_text(self)

@dataclass
class MaskAnnotation(BaseAnnotation):
    """Segmentation mask overlay"""
    mask: np.ndarray
    alpha: float = 0.5
    colormap: Optional[str] = None  # 'random', 'jet', 'hot', etc.
    
    def accept(self, processor):
        return processor.render_mask(self)

@dataclass
class OrientationAnnotation(BaseAnnotation):
    """3D orientation visualization"""
    position: Tuple[int, int]
    pitch: float  # degrees
    yaw: float    # degrees
    roll: float   # degrees
    size: int = 100
    
    def accept(self, processor):
        return processor.render_orientation(self)

@dataclass
class BoundingBoxAnnotation(BaseAnnotation):
    """A bounding box with optional label"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    label: Optional[str] = None
    confidence: Optional[float] = None
    thickness: int = 2
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.bbox[0], self.bbox[1])
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3])
    
    def accept(self, processor):
        return processor.render_bounding_box(self)

# Collection Annotations

@dataclass
class PointsAnnotation(BaseAnnotation):
    """Multiple points with the same style"""
    points: List[Tuple[int, int]]
    size: int = 5
    
    def accept(self, processor):
        return processor.render_points(self)

@dataclass
class LabeledPointsAnnotation(BaseAnnotation):
    """Multiple points with individual labels"""
    points: List[Tuple[int, int]]
    labels: List[str]
    size: int = 5
    font_scale: float = 0.5
    
    def __post_init__(self):
        if len(self.points) != len(self.labels):
            raise ValueError("Number of points must match number of labels")
    
    def accept(self, processor):
        return processor.render_labeled_points(self)
```

## Usage Comparison

### Before (Current System):
```python
# Confusing - which fields are needed?
ann1 = Annotation(
    type=AnnotationType.CIRCLE,
    coordinates=(100, 100),  # Is this center? top-left?
    radius=50,  # Will this be used?
    thickness=2,  # Will this be used?
    labels="test",  # Will this be ignored?
    orientation=(0, 0, 0)  # Definitely ignored but no error!
)

# IDE shows ALL fields even though most are irrelevant
ann2 = Annotation(
    type=AnnotationType.RECTANGLE,
    coordinates=(10, 10, 50, 50),  # x,y,w,h? or two points?
    # color=... radius=... mask=... all shown in autocomplete!
)

# Runtime surprise!
ann3 = Annotation(
    type=AnnotationType.MASK,
    coordinates=(0, 0),  # Ignored at runtime
    mask=my_mask  # This is what's actually needed
)
```

### After (Type-Specific Classes):
```python
# Crystal clear what's needed
ann1 = CircleAnnotation(
    center=(100, 100),  # Clear parameter name!
    radius=50,
    thickness=2,
    color=(255, 0, 0)
)

# IDE only shows relevant fields
ann2 = RectangleAnnotation(
    top_left=(10, 10),
    bottom_right=(60, 60),
    thickness=2
)
# Or use convenient factory method
ann2_alt = RectangleAnnotation.from_xywh(10, 10, 50, 50)

# Type-specific, no confusion
ann3 = MaskAnnotation(
    mask=my_mask,
    alpha=0.5,
    colormap='jet'
)

# Impossible to create invalid states
# ann4 = CircleAnnotation(mask=...)  # TYPE ERROR! Won't even run
```

## IDE Autocomplete Benefits

### Current System:
```python
ann = Annotation(
    type=AnnotationType.POINT_AND_LABEL,
    # IDE shows: coordinates, color, labels, radius, thickness, 
    #            orientation, mask, info_panel...
    # Which ones do I need??? 🤔
)
```

### New System:
```python
ann = LabeledPointAnnotation(
    # IDE shows ONLY: position, label, size, font_scale, color
    # Exactly what you need! ✅
    position=(100, 100),
    label="Detection"
)
```

## Processor Implementation

```python
class AnnotationProcessor:
    """Processes type-specific annotations"""
    
    def render(self, image: np.ndarray, annotation: BaseAnnotation):
        """Main entry point - uses visitor pattern"""
        return annotation.accept(self)
    
    def render_point(self, ann: PointAnnotation):
        cv2.circle(image, ann.position, ann.size, ann.color, -1)
    
    def render_circle(self, ann: CircleAnnotation):
        thickness = -1 if ann.filled else ann.thickness
        cv2.circle(image, ann.center, ann.radius, ann.color, thickness)
    
    def render_rectangle(self, ann: RectangleAnnotation):
        thickness = -1 if ann.filled else ann.thickness
        cv2.rectangle(image, ann.top_left, ann.bottom_right, 
                     ann.color, thickness)
    
    def render_bounding_box(self, ann: BoundingBoxAnnotation):
        # Draw box
        cv2.rectangle(image, ann.top_left, ann.bottom_right, 
                     ann.color, ann.thickness)
        
        # Add label if present
        if ann.label:
            text = ann.label
            if ann.confidence:
                text += f" {ann.confidence:.2%}"
            
            # Add background for text
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, 
                         (ann.top_left[0], ann.top_left[1] - h - 4),
                         (ann.top_left[0] + w, ann.top_left[1]),
                         ann.color, -1)
            cv2.putText(image, text,
                       (ann.top_left[0], ann.top_left[1] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

## Convenience Functions

```python
# Helper functions for common patterns
def point(x: int, y: int, **kwargs) -> PointAnnotation:
    """Quick point creation"""
    return PointAnnotation(position=(x, y), **kwargs)

def box(x: int, y: int, w: int, h: int, **kwargs) -> RectangleAnnotation:
    """Quick box creation"""
    return RectangleAnnotation.from_xywh(x, y, w, h, **kwargs)

def text(msg: str, x: int, y: int, **kwargs) -> TextAnnotation:
    """Quick text creation"""
    return TextAnnotation(text=msg, position=(x, y), **kwargs)

def circle(x: int, y: int, r: int, **kwargs) -> CircleAnnotation:
    """Quick circle creation"""
    return CircleAnnotation(center=(x, y), radius=r, **kwargs)

# Usage
annotations = [
    point(100, 100, color=(255, 0, 0)),
    box(50, 50, 200, 150, thickness=3),
    text("Hello", 10, 10, font_scale=1.0),
    circle(200, 200, 50, filled=True)
]
```

## Migration Strategy

```python
# Compatibility layer during migration
def from_old_annotation(old_ann: Annotation) -> BaseAnnotation:
    """Convert old annotation to new type-specific class"""
    
    if old_ann.type == AnnotationType.POINT:
        return PointAnnotation(
            position=old_ann.coordinates,
            color=old_ann.color
        )
    elif old_ann.type == AnnotationType.CIRCLE:
        return CircleAnnotation(
            center=old_ann.coordinates,
            radius=old_ann.radius or 5,
            thickness=old_ann.thickness or 2,
            color=old_ann.color
        )
    elif old_ann.type == AnnotationType.RECTANGLE:
        x, y, w, h = old_ann.coordinates
        return RectangleAnnotation.from_xywh(x, y, w, h, color=old_ann.color)
    # ... etc
```

## Benefits Summary

### 1. **Type Safety**
```python
# Impossible to create invalid annotations
# circle = CircleAnnotation(mask=...)  # ❌ Type error!
circle = CircleAnnotation(center=(100, 100), radius=50)  # ✅ Clear!
```

### 2. **Better IDE Support**
- Autocomplete shows only relevant parameters
- Parameter names are descriptive (`center` not `coordinates`)
- Type hints work properly

### 3. **Self-Documenting Code**
```python
# The class name tells you exactly what it does
ann = BoundingBoxAnnotation(...)  # Obviously a bounding box
ann = OrientationAnnotation(...)   # Obviously for 3D orientation
```

### 4. **No Optional Field Confusion**
```python
# Current: Is radius used for RECTANGLE? Is mask used for CIRCLE?
# New: Each class has exactly the fields it needs
```

### 5. **Cleaner Processing**
```python
# No giant if-elif chain
def render(annotation):
    annotation.accept(processor)  # Polymorphism handles dispatch
```

### 6. **Easier Testing**
```python
# Test each annotation type independently
def test_circle_annotation():
    ann = CircleAnnotation(center=(100, 100), radius=50)
    assert ann.center == (100, 100)
    assert ann.radius == 50
    # No need to test irrelevant fields
```

### 7. **Validation at Construction**
```python
# Validation happens when creating the object
try:
    ann = LabeledPointsAnnotation(
        points=[(1, 1), (2, 2)],
        labels=["A", "B", "C"]  # Too many labels!
    )
except ValueError:  # Caught immediately, not at render time
    pass
```

## Real-World Example

```python
# Clean, type-safe, self-documenting code
def annotate_detection(image, detection_result):
    annotations = []
    
    for det in detection_result.boxes:
        # Type-safe, autocomplete works perfectly
        annotations.append(BoundingBoxAnnotation(
            bbox=(det.x, det.y, det.w, det.h),
            label=det.class_name,
            confidence=det.score,
            color=color_from_class(det.class_id)
        ))
        
        # Add center point
        annotations.append(PointAnnotation(
            position=(det.x + det.w//2, det.y + det.h//2),
            size=3,
            color=(255, 0, 0)
        ))
    
    # Add overall info
    annotations.append(TextAnnotation(
        text=f"Detected: {len(detection_result.boxes)} objects",
        position=(10, 30),
        background=(0, 0, 0),
        color=(255, 255, 255)
    ))
    
    return annotations
```

## Conclusion

Type-specific annotation classes provide:
- **Compile-time safety** instead of runtime errors
- **Clear APIs** with descriptive parameter names
- **Better tooling** with IDE autocomplete and type checking
- **Simpler code** without optional field confusion
- **Easier maintenance** with clear separation of concerns

This refactor transforms annotations from error-prone data bags into well-defined, type-safe objects that are a pleasure to work with!