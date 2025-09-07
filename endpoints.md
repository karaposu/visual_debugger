# Visual Debugger API Reference

Complete API documentation for the Visual Debugger library.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Annotation Classes](#annotation-classes)
3. [Factory Functions](#factory-functions)
4. [Info Panel System](#info-panel-system)
5. [Image Composition](#image-composition)
6. [Annotation Processor](#annotation-processor)

---

## Core Classes

### VisualDebugger

Main orchestrator for visual debugging operations.

```python
class VisualDebugger:
    def __init__(
        self,
        tag: str = "visual_debugger",
        debug_folder_path: str = ".",
        active: bool = True,
        output: str = 'save'  # 'save' | 'return' | 'both'
    )
```

#### Methods

##### visual_debug
```python
def visual_debug(
    self,
    img: np.ndarray,
    annotations: Union[List[BaseAnnotation], BaseAnnotation, InfoPanel, None] = [],
    name: str = "generic",
    stage_name: Optional[str] = None,
    transparent: bool = False,
    mask: bool = False
) -> Optional[np.ndarray]
```
Apply annotations to an image.

**Parameters:**
- `img`: Input image as numpy array or file path
- `annotations`: Single annotation, list of annotations, InfoPanel, or None
  - Can pass InfoPanel directly without wrapping in InfoPanelAnnotation
  - Can mix InfoPanel with other annotations in a list
- `name`: Base name for saved files
- `stage_name`: Optional stage name for file naming
- `transparent`: Whether to convert to RGBA
- `mask`: Whether to save as mask format

**Returns:**
- `None` if output='save'
- `np.ndarray` if output='return' or 'both'

---

## Annotation Classes

### BaseAnnotation

Abstract base class for all annotations.

```python
@dataclass
class BaseAnnotation(ABC):
    @abstractmethod
    def accept(self, processor: 'AnnotationProcessor') -> None:
        """Accept a processor visitor"""
        
    @abstractmethod
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Calculate bounding box as (x_min, y_min, x_max, y_max)"""
```

### Point Annotations

#### PointAnnotation
```python
@dataclass
class PointAnnotation(BaseAnnotation):
    position: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
```

#### LabeledPointAnnotation
```python
@dataclass
class LabeledPointAnnotation(BaseAnnotation):
    position: Tuple[int, int]
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
```

#### PointsAnnotation
```python
@dataclass
class PointsAnnotation(BaseAnnotation):
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
```

#### LabeledPointsAnnotation
```python
@dataclass
class LabeledPointsAnnotation(BaseAnnotation):
    points: List[Tuple[int, int]]
    labels: List[str]
    color: Tuple[int, int, int] = (0, 255, 0)
    size: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
```

### Shape Annotations

#### CircleAnnotation
```python
@dataclass
class CircleAnnotation(BaseAnnotation):
    center: Tuple[int, int]
    radius: int
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
```

#### LabeledCircleAnnotation
```python
@dataclass
class LabeledCircleAnnotation(BaseAnnotation):
    center: Tuple[int, int]
    radius: int
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
    font_scale: float = 0.5
    font_thickness: int = 1
```

#### RectangleAnnotation
```python
@dataclass
class RectangleAnnotation(BaseAnnotation):
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    filled: bool = False
```

#### AlphaRectangleAnnotation
```python
@dataclass
class AlphaRectangleAnnotation(BaseAnnotation):
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.7  # 0.0 (transparent) to 1.0 (opaque)
```

### Line Annotations

#### LineAnnotation
```python
@dataclass
class LineAnnotation(BaseAnnotation):
    start: Tuple[int, int]
    end: Tuple[int, int]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    arrow: bool = False
```

#### LabeledLineAnnotation
```python
@dataclass
class LabeledLineAnnotation(BaseAnnotation):
    start: Tuple[int, int]
    end: Tuple[int, int]
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    arrow: bool = False
    font_scale: float = 0.5
    font_thickness: int = 1
```

#### PolylineAnnotation
```python
@dataclass
class PolylineAnnotation(BaseAnnotation):
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    closed: bool = False
```

### Text Annotations

#### TextAnnotation
```python
@dataclass
class TextAnnotation(BaseAnnotation):
    text: str
    position: Tuple[int, int]
    color: Tuple[int, int, int] = (255, 255, 255)
    font_scale: float = 0.5
    font_thickness: int = 1
    background_color: Optional[Tuple[int, int, int]] = None
    background_padding: int = 5
```

### Complex Annotations

#### BoundingBoxAnnotation
```python
@dataclass
class BoundingBoxAnnotation(BaseAnnotation):
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    label: Optional[str] = None
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
    show_label_background: bool = True
```

#### MaskAnnotation
```python
@dataclass
class MaskAnnotation(BaseAnnotation):
    mask: np.ndarray  # 2D array with class IDs
    alpha: float = 0.5
    colormap: str = 'default'  # 'default' | 'random' | 'jet' | 'hot' | 'cool' | 'hsv'
```

#### OrientationAnnotation
```python
@dataclass
class OrientationAnnotation(BaseAnnotation):
    position: Tuple[int, int]
    pitch: float  # degrees
    yaw: float    # degrees
    roll: float   # degrees
    size: int = 50
    x_color: Tuple[int, int, int] = (255, 0, 0)  # red
    y_color: Tuple[int, int, int] = (0, 255, 0)  # green
    z_color: Tuple[int, int, int] = (0, 0, 255)  # blue
```

#### InfoPanelAnnotation
```python
@dataclass
class InfoPanelAnnotation(BaseAnnotation):
    panel: 'InfoPanel'
    
    def get_component_annotations(
        self, 
        image_shape: Tuple[int, int]
    ) -> List[BaseAnnotation]:
        """Decompose into basic annotations"""
```

---

## Factory Functions

Convenience functions for creating annotations quickly.

```python
def point(x: int, y: int, **kwargs) -> PointAnnotation:
    """Create a point annotation"""

def labeled_point(x: int, y: int, label: str, **kwargs) -> LabeledPointAnnotation:
    """Create a labeled point annotation"""

def circle(center_x: int, center_y: int, radius: int, **kwargs) -> CircleAnnotation:
    """Create a circle annotation"""

def rectangle(x: int, y: int, width: int, height: int, **kwargs) -> RectangleAnnotation:
    """Create a rectangle annotation from xywh"""

def line(start_x: int, start_y: int, end_x: int, end_y: int, **kwargs) -> LineAnnotation:
    """Create a line annotation"""

def text(text: str, x: int, y: int, **kwargs) -> TextAnnotation:
    """Create a text annotation"""

def bbox(x: int, y: int, width: int, height: int, label: Optional[str] = None, **kwargs) -> BoundingBoxAnnotation:
    """Create a bounding box annotation"""
```

---

## Info Panel System

### InfoPanel

Dashboard-style information overlay.

```python
class InfoPanel:
    def __init__(
        self,
        position: Union[PanelPosition, Tuple[int, int]] = PanelPosition.TOP_LEFT,
        title: Optional[str] = None,  # Optional for compact display
        style: Optional[PanelStyle] = None
    )
```

#### Methods

##### add
```python
def add(self, key: str, value: str) -> None:
    """Add a key-value pair"""
```

##### add_separator
```python
def add_separator(self) -> None:
    """Add a horizontal separator line"""
```

##### add_progress
```python
def add_progress(self, label: str, value: float, max_width: int = 100) -> None:
    """Add a progress bar (value: 0.0 to 1.0)"""
```

##### add_table
```python
def add_table(self, headers: List[str], rows: List[List[str]]) -> None:
    """Add a formatted table"""
```

##### clear
```python
def clear(self) -> None:
    """Clear all entries"""
```

##### calculate_dimensions
```python
def calculate_dimensions(self) -> Tuple[int, int]:
    """Calculate panel width and height"""
```

##### calculate_position
```python
def calculate_position(self, image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Calculate top-left position on image"""
```

### PanelPosition

Enum for preset panel positions.

```python
class PanelPosition(Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    CENTER = "center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"
```

### PanelStyle

Styling configuration for panels.

```python
@dataclass
class PanelStyle:
    background_color: Tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.7  # 0.0-1.0 transparency
    text_color: Tuple[int, int, int] = (255, 255, 255)
    title_color: Tuple[int, int, int] = (255, 255, 255)
    separator_color: Tuple[int, int, int] = (128, 128, 128)
    border_color: Optional[Tuple[int, int, int]] = None
    border_thickness: int = 2
    padding: int = 10
    font_scale: float = 0.5
    font_thickness: int = 1
    show_background: bool = True
```

---

## Image Composition

### ImageCompositor

Combine multiple images into layouts.

```python
class ImageCompositor:
    def __init__(self, style: Optional[CompositionStyle] = None)
```

#### Methods

##### concatenate
```python
def concatenate(
    self,
    images: List[np.ndarray],
    direction: LayoutDirection = LayoutDirection.HORIZONTAL,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """Concatenate images horizontally or vertically"""
```

##### create_grid
```python
def create_grid(
    self,
    images: List[np.ndarray],
    cols: int,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """Arrange images in a grid layout"""
```

##### create_comparison
```python
def create_comparison(
    self,
    before: np.ndarray,
    after: np.ndarray,
    before_label: str = "Before",
    after_label: str = "After",
    direction: LayoutDirection = LayoutDirection.HORIZONTAL
) -> np.ndarray:
    """Create before/after comparison view"""
```

### CompositionStyle

Styling for image compositions.

```python
@dataclass
class CompositionStyle:
    border_width: int = 2
    border_color: Tuple[int, int, int] = (255, 255, 255)
    label_height: int = 30
    label_font_scale: float = 0.7
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_background: Tuple[int, int, int] = (0, 0, 0)
    padding: int = 0
```

### LayoutDirection

Enum for layout orientations.

```python
class LayoutDirection(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
```

---

## Annotation Processor

### AnnotationProcessor

Visitor pattern processor for rendering annotations.

```python
class AnnotationProcessor:
    def __init__(self)
```

#### Methods

##### process
```python
def process(
    self,
    image: np.ndarray,
    annotation: BaseAnnotation
) -> np.ndarray:
    """Process an annotation on an image"""
```

##### render
```python
def render(
    self,
    image: np.ndarray,
    annotation: BaseAnnotation
) -> np.ndarray:
    """Alias for process method"""
```

#### Specialized Renderers

Each annotation type has a corresponding render method:

```python
def render_point(self, annotation: PointAnnotation) -> None
def render_labeled_point(self, annotation: LabeledPointAnnotation) -> None
def render_points(self, annotation: PointsAnnotation) -> None
def render_labeled_points(self, annotation: LabeledPointsAnnotation) -> None
def render_circle(self, annotation: CircleAnnotation) -> None
def render_labeled_circle(self, annotation: LabeledCircleAnnotation) -> None
def render_rectangle(self, annotation: RectangleAnnotation) -> None
def render_alpha_rectangle(self, annotation: AlphaRectangleAnnotation) -> None
def render_line(self, annotation: LineAnnotation) -> None
def render_labeled_line(self, annotation: LabeledLineAnnotation) -> None
def render_polyline(self, annotation: PolylineAnnotation) -> None
def render_text(self, annotation: TextAnnotation) -> None
def render_bounding_box(self, annotation: BoundingBoxAnnotation) -> None
def render_mask(self, annotation: MaskAnnotation) -> None
def render_orientation(self, annotation: OrientationAnnotation) -> None
```

---

## Usage Examples

### Basic Annotation
```python
from visual_debugger import VisualDebugger
from visual_debugger.annotations import point, circle, bbox

vd = VisualDebugger(output='return')
img = cv2.imread('image.jpg')

annotations = [
    point(100, 200, color=(255, 0, 0)),
    circle(300, 300, 50),
    bbox(10, 10, 100, 100, label="Object 95%")
]

result = vd.visual_debug(img, annotations)
```

### Info Panel with Custom Style
```python
from visual_debugger.info_panel import InfoPanel, PanelPosition, PanelStyle
from visual_debugger.annotations import InfoPanelAnnotation

style = PanelStyle(
    background_color=(20, 20, 80),
    background_alpha=0.8,
    text_color=(200, 200, 255),
    font_scale=0.6
)

panel = InfoPanel(
    position=PanelPosition.TOP_RIGHT,
    title="Status",  # Optional
    style=style
)

panel.add("FPS", "30.0")
panel.add_separator()
panel.add_progress("Processing", 0.75)

panel_ann = InfoPanelAnnotation(panel=panel)
result = vd.visual_debug(img, [panel_ann])
```

### Image Grid Composition
```python
from visual_debugger.composition import ImageCompositor

compositor = ImageCompositor()
grid = compositor.create_grid(
    images=[img1, img2, img3, img4],
    cols=2,
    labels=["Step 1", "Step 2", "Step 3", "Step 4"]
)
```

### Custom Font Sizes
```python
from visual_debugger.annotations import TextAnnotation, LabeledPointAnnotation

annotations = [
    TextAnnotation(
        text="Large Text",
        position=(100, 100),
        font_scale=1.2,  # Larger font
        font_thickness=2  # Bold
    ),
    LabeledPointAnnotation(
        position=(200, 200),
        label="Tiny Label",
        font_scale=0.3  # Small font
    )
]
```

---

## Type Hints

All methods and classes are fully typed. Common types used:

```python
from typing import List, Optional, Tuple, Union
import numpy as np

# Common type aliases
Color = Tuple[int, int, int]  # RGB color
Point = Tuple[int, int]       # 2D coordinate
BBox = Tuple[int, int, int, int]  # x, y, width, height
```

---

## Error Handling

The library handles common errors gracefully:

- Invalid image shapes
- Out-of-bounds coordinates (clipped to image boundaries)
- Mismatched annotation parameters
- Empty annotation lists

---

## Performance Notes

- Annotations are rendered in order (later annotations appear on top)
- Image modifications are done in-place when possible
- Alpha blending uses optimized OpenCV operations
- Boundary calculations are cached where applicable

---

## Thread Safety

The Visual Debugger is not thread-safe. For multi-threaded applications:
- Create separate VisualDebugger instances per thread
- Or use threading locks around visual_debug calls

---

## Version Compatibility

- Requires Python 3.7+
- OpenCV 4.0+
- NumPy 1.16+
- Optional: Pillow for additional image formats