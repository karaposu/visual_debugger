# Visual Debugger Refactoring Discussion

## Overview
This document outlines key refactoring opportunities for the Visual Debugger codebase, focusing on hiding implementation details and improving maintainability through better abstraction patterns.

## 1. Strategy Pattern for Annotation Rendering

### Current Problem
The `put_annotation_on_image()` method contains a massive if-elif chain that violates the Open-Closed Principle. Adding new annotation types requires modifying this core method.

### Current Code (Problematic)
```python
def put_annotation_on_image(self, image, annotation):
    if annotation.type == AnnotationType.CIRCLE:
        cv2.circle(image, annotation.coordinates, 5, annotation.color, -1)
    elif annotation.type == AnnotationType.RECTANGLE:
        x, y, w, h = annotation.coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), annotation.color, 2)
    elif annotation.type == AnnotationType.MASK:
        self.put_mask_on_image(image, annotation.mask)
    # ... 10+ more elif branches
```

### Proposed Solution
```python
# renderers.py
from abc import ABC, abstractmethod

class AnnotationRenderer(ABC):
    @abstractmethod
    def render(self, image, annotation): pass

class CircleRenderer(AnnotationRenderer):
    def render(self, image, annotation):
        cv2.circle(image, annotation.coordinates, 5, annotation.color, -1)

class RectangleRenderer(AnnotationRenderer):
    def render(self, image, annotation):
        x, y, w, h = annotation.coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), annotation.color, 2)

# image_processor.py
class ImageProcessor:
    def __init__(self):
        self._renderers = {
            AnnotationType.CIRCLE: CircleRenderer(),
            AnnotationType.RECTANGLE: RectangleRenderer(),
            AnnotationType.MASK: MaskRenderer(),
        }
    
    def put_annotation_on_image(self, image, annotation):
        renderer = self._renderers.get(annotation.type)
        if renderer:
            renderer.render(image, annotation)
```

### Benefits
- **Extensibility**: Add new annotation types without modifying existing code
- **Testability**: Each renderer can be tested independently
- **Single Responsibility**: Each renderer handles only one annotation type
- **Encapsulation**: Implementation details are hidden in specific renderer classes

## 2. Drawing Facade to Hide OpenCV

### Current Problem
OpenCV implementation details are scattered throughout the codebase. Switching to another graphics library would require changes everywhere.

### Current Code (Problematic)
```python
# OpenCV details exposed everywhere
cv2.circle(image, point, 5, color, -1)
cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
```

### Proposed Solution
```python
# drawing.py
class DrawingContext:
    """Facade that hides OpenCV implementation"""
    
    def circle(self, image, center, radius=5, color=(0, 255, 0), filled=True):
        thickness = -1 if filled else 2
        cv2.circle(image, center, radius, color, thickness)
        return image
    
    def text(self, image, text, position, size=0.5, color=(0, 255, 0)):
        cv2.putText(image, text, position, 
                   cv2.FONT_HERSHEY_SIMPLEX, size, color, 1)
        return image
    
    def blend(self, image1, image2, alpha=0.5):
        return cv2.addWeighted(image1, alpha, image2, 1-alpha, 0)

# Usage
class CircleRenderer(AnnotationRenderer):
    def __init__(self, drawer: DrawingContext):
        self.drawer = drawer
    
    def render(self, image, annotation):
        self.drawer.circle(image, annotation.coordinates, 
                          color=annotation.color)
```

### Benefits
- **Library Independence**: Can switch from OpenCV without changing business logic
- **Cleaner API**: More intuitive method names and parameters
- **Mockability**: Easy to mock for testing
- **Consistency**: Enforces consistent drawing styles

## 3. Extract Complex Image Composition

### Current Problem
Image concatenation logic is deeply embedded in `ImageProcessor`, mixing low-level pixel manipulation with high-level layout logic.

### Current Code (Problematic)
```python
class ImageProcessor:
    def concat_images(self, images, axis=1, border_thickness=5, ...):
        # 100+ lines of complex logic
        grouped_images, group_names = self.group_images_by_name(images)
        if axis == 1:
            return self.concat_images_horizontally(grouped_images, ...)
        else:
            return self.concat_images_vertically(grouped_images, ...)
    
    def calculate_horizontal_dimensions(self, ...):
        # Complex dimension calculations
    
    def add_labels_and_borders(self, ...):
        # Border and label logic
```

### Proposed Solution
```python
# composition.py
@dataclass
class CompositionStyle:
    border_thickness: int = 5
    border_color: Tuple = (255, 255, 255)
    spacing: int = 20
    label_font_size: float = 0.8

class ImageCompositor:
    def __init__(self, style: CompositionStyle = None):
        self.style = style or CompositionStyle()
    
    def create_grid(self, images: List[np.ndarray], cols: int = None):
        """Create a grid layout of images"""
        if cols is None:
            cols = int(np.ceil(np.sqrt(len(images))))
        
        grid = Grid(cols=cols, style=self.style)
        for img in images:
            grid.add(img)
        return grid.render()
    
    def create_comparison(self, before: np.ndarray, after: np.ndarray):
        """Create a side-by-side comparison"""
        return SideBySide(before, after, self.style).render()

# Usage
compositor = ImageCompositor(style=CompositionStyle(border_thickness=3))
grid_image = compositor.create_grid(debug_images, cols=3)
```

### Benefits
- **Separation of Concerns**: Layout logic separate from annotation logic
- **Reusability**: Composition logic can be used independently
- **Flexibility**: Easy to add new layout types
- **Cleaner Interface**: Intuitive methods instead of parameter-heavy functions

## 4. Configuration Objects Instead of Parameters

### Current Problem
Methods have too many parameters, making them hard to use and maintain.

### Current Code (Problematic)
```python
def visual_debug(self, img, annotations=[], name="generic", 
                stage_name=None, transparent=False, mask=False):
    # Hard to remember parameter order and defaults

def concat_images(self, images, axis=1, border_thickness=5, 
                 border_color=(255, 255, 255), vertical_space=20, 
                 horizontal_space=20):
    # Too many parameters!
```

### Proposed Solution
```python
# config.py
@dataclass
class DebugConfig:
    """Configuration for debug visualization"""
    name: str = "generic"
    stage_name: Optional[str] = None
    transparent: bool = False
    save_mask: bool = False
    output_mode: str = 'save'  # 'save' or 'return'

@dataclass
class LayoutConfig:
    """Configuration for image layout"""
    direction: str = 'horizontal'  # or 'vertical'
    border: BorderStyle = field(default_factory=BorderStyle)
    spacing: SpacingStyle = field(default_factory=SpacingStyle)

# Usage
config = DebugConfig(name="detection", stage_name="preprocessing")
debugger.visual_debug(image, annotations, config)

layout = LayoutConfig(direction='vertical', 
                      border=BorderStyle(thickness=3))
compositor.arrange(images, layout)
```

### Benefits
- **Self-Documenting**: Configuration objects document their purpose
- **Extensible**: Easy to add new options without breaking existing code
- **Reusable**: Configurations can be saved and reused
- **Type Safety**: IDEs can provide better autocomplete and type checking

## 5. Extract 3D Orientation Mathematics

### Current Problem
Complex 3D mathematics is embedded in the image processing class, mixing mathematical operations with rendering logic.

### Current Code (Problematic)
```python
def draw_orientation(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    roll = np.deg2rad(roll)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], ...])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], ...])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], ...])
    R = Rz @ Ry @ Rx
    # More math and drawing mixed together
```

### Proposed Solution
```python
# geometry.py
@dataclass
class Orientation3D:
    pitch: float  # degrees
    yaw: float    # degrees
    roll: float   # degrees
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert orientation to 3x3 rotation matrix"""
        p = np.deg2rad(self.pitch)
        y = np.deg2rad(self.yaw)
        r = np.deg2rad(self.roll)
        
        Rx = self._rotation_x(p)
        Ry = self._rotation_y(y)
        Rz = self._rotation_z(r)
        return Rz @ Ry @ Rx

class OrientationVisualizer:
    def __init__(self, drawer: DrawingContext):
        self.drawer = drawer
    
    def draw_axes(self, image, orientation: Orientation3D, 
                  origin: Tuple[int, int], size: int = 100):
        """Draw 3D axes on 2D image"""
        axes_3d = self._create_axes(size)
        axes_2d = self._project_to_2d(axes_3d, orientation, origin)
        
        self.drawer.arrow(image, origin, axes_2d.x_axis, color=(255, 0, 0))
        self.drawer.arrow(image, origin, axes_2d.y_axis, color=(0, 255, 0))
        self.drawer.arrow(image, origin, axes_2d.z_axis, color=(0, 0, 255))

# Usage
orientation = Orientation3D(pitch=30, yaw=45, roll=15)
visualizer = OrientationVisualizer(drawer)
visualizer.draw_axes(image, orientation, origin=(100, 100))
```

### Benefits
- **Separation of Math and Rendering**: Mathematical operations isolated from drawing
- **Testability**: Can test mathematics without image operations
- **Reusability**: Orientation math can be used elsewhere
- **Clarity**: Clear separation between "what to draw" and "how to draw"

## 6. Builder Pattern for Complex Annotations

### Current Problem
Creating complex annotations with multiple properties requires remembering many optional parameters.

### Current Code (Problematic)
```python
annotation = Annotation(
    type=AnnotationType.CIRCLE_AND_LABEL,
    coordinates=(150, 195),
    radius=20,
    thickness=2,
    labels="Circle 1",
    color=(255, 0, 0)
)
```

### Proposed Solution
```python
# builders.py
class AnnotationBuilder:
    def __init__(self):
        self._annotation = Annotation(type=AnnotationType.POINT)
    
    def circle(self, center, radius=5):
        self._annotation.type = AnnotationType.CIRCLE
        self._annotation.coordinates = center
        self._annotation.radius = radius
        return self
    
    def with_label(self, text):
        if self._annotation.type == AnnotationType.CIRCLE:
            self._annotation.type = AnnotationType.CIRCLE_AND_LABEL
        self._annotation.labels = text
        return self
    
    def color(self, r, g, b):
        self._annotation.color = (r, g, b)
        return self
    
    def build(self):
        return self._annotation

# Usage - Much more readable!
annotation = (AnnotationBuilder()
    .circle(center=(150, 195), radius=20)
    .with_label("Detection")
    .color(255, 0, 0)
    .build())
```

### Benefits
- **Fluent Interface**: Readable, chainable API
- **Validation**: Builder can validate combinations
- **Defaults**: Smart defaults based on annotation type
- **Discoverability**: IDE autocomplete guides users

## Implementation Priority

1. **Strategy Pattern for Renderers** (High Impact, Medium Effort)
   - Eliminates the biggest code smell
   - Makes adding new features trivial
   
2. **Drawing Facade** (High Impact, Low Effort)
   - Hides all OpenCV details
   - Makes testing much easier
   
3. **Configuration Objects** (Medium Impact, Low Effort)
   - Immediate API improvement
   - Better documentation
   
4. **Image Compositor** (Medium Impact, High Effort)
   - Complex but valuable refactoring
   - Could be done incrementally
   
5. **Orientation Extraction** (Low Impact, Low Effort)
   - Nice to have for code organization
   - Good candidate for first refactoring

## Conclusion

These refactorings would transform the codebase from a monolithic, tightly-coupled system into a modular, extensible, and testable architecture. The key benefits are:

- **Maintainability**: Each component has a single, clear responsibility
- **Extensibility**: New features can be added without modifying existing code
- **Testability**: Components can be tested in isolation
- **Flexibility**: Implementation details are hidden behind abstractions
- **Usability**: Cleaner, more intuitive APIs for users

Start with the Strategy Pattern and Drawing Facade for maximum immediate impact with reasonable effort.