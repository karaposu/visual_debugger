# Visual Debugger for Computer Vision

A visual debugging toolkit for computer vision and image processing workflows. Annotate, visualize, and debug your image processing pipelines with ease.

## 🚀 Features

## Features

- **Multiple Annotation Types**: Support for a variety of annotations such as points, labeled points, rectangles, circles, and orientation vectors based on pitch, yaw, and roll.
- **Image Concatenation**: Capability to concatenate multiple debugged images into a single composite image, facilitating easier visualization of sequential image processing steps.
- **Dynamic Image Handling**: Handles a wide range of image inputs including file paths, in-memory image arrays, base64 encoded images, and images from web links, integrating seamlessly with OpenCV.
- **Customizable Debugging**: Debugging can be turned on or off, and the module supports generating merged debug images for a sequence of operations.

### Annotation Types

| Category | Annotation Class | Description | Key Features |
|----------|-----------------|-------------|--------------|
| **📍 Points** | `PointAnnotation` | Single point marker | Customizable size and color |
| | `LabeledPointAnnotation` | Point with text label | Font size/thickness control |
| | `PointsAnnotation` | Multiple points | Uniform styling for all points |
| | `LabeledPointsAnnotation` | Multiple labeled points | Individual labels per point |
| **🔷 Shapes** | `CircleAnnotation` | Circle shape | Outline or filled |
| | `LabeledCircleAnnotation` | Circle with label | Text positioning options |
| | `RectangleAnnotation` | Rectangle shape | Corner or xywh specification |
| | `AlphaRectangleAnnotation` | Semi-transparent rectangle | Alpha blending (0.0-1.0) |
| **📏 Lines** | `LineAnnotation` | Straight line | Optional arrow heads |
| | `LabeledLineAnnotation` | Line with label | Midpoint text placement |
| | `PolylineAnnotation` | Connected segments | Open or closed path |
| **📝 Text** | `TextAnnotation` | Standalone text | Optional background, padding |
| **📦 Complex** | `BoundingBoxAnnotation` | Detection box | Label with background |
| | `MaskAnnotation` | Segmentation mask | Multiple colormaps |
| | `OrientationAnnotation` | 3D axes visualization | Pitch, yaw, roll display |
| | `InfoPanelAnnotation` | Dashboard overlay | Composite of basic annotations |

### 📊 Info Panel System
Flexible dashboard overlays with extensive customization:
- **Positioning**: 9 preset positions or custom coordinates
- **Styling**: Colors, borders, padding, fonts
- **Content**: Key-value pairs, separators, progress bars, tables
- **Optimization**: Optional title for space saving
- **Themes**: Dark, light, minimal, or custom styles

### 🖼️ Image Composition
Combine multiple images with:
- Horizontal/vertical concatenation
- Grid layouts with automatic sizing
- Before/after comparisons
- Customizable borders and labels

## 📦 Installation

```bash
pip install visual_debugger
```

## 🎯 Quick Start

### Basic Usage

```python
from visual_debugger import VisualDebugger
from visual_debugger.annotations import *

# Initialize debugger
vd = VisualDebugger(
    tag="my_project",
    debug_folder_path="./debug_output",
    active=True,
    output='save'  # 'save', 'return', or 'both'
)

# Load your image
import cv2
img = cv2.imread("image.jpg")

# Create annotations
annotations = [
    point(100, 200, color=(255, 0, 0), size=10),
    circle(300, 300, 50, color=(0, 255, 0)),
    bbox(50, 50, 200, 150, label="Person 95%"),
    text("Debug Info", 10, 30, font_scale=1.0)
]

# Apply annotations
result = vd.visual_debug(img, annotations, process_step="detection")


```

### Using Info Panels

```python
from visual_debugger.info_panel import InfoPanel, PanelPosition

# Create info panel (title is optional)
panel = InfoPanel(
    position=PanelPosition.TOP_LEFT,
    title="System Status"  # Can be None or omitted for compact display
)

# Or create a compact panel without title (saves space)
panel_compact = InfoPanel(position=PanelPosition.TOP_RIGHT)

# Custom styling
from visual_debugger.info_panel import PanelStyle

custom_style = PanelStyle(
    background_color=(40, 20, 80),      # Dark blue
    background_alpha=0.7,                # 70% opacity (30% transparent)
    text_color=(200, 220, 255),         # Light blue text
    title_color=(255, 200, 100),        # Orange title
    border_color=(100, 150, 255),       # Blue border
    border_thickness=3,
    padding=20,
    font_scale=0.6,
    show_background=True                # Set False for no background
)

styled_panel = InfoPanel(
    position=PanelPosition.BOTTOM_LEFT,
    title="Custom Theme",
    style=custom_style
)

# Add information
panel.add("FPS", "30.0")
panel.add("Objects", "5")
panel.add_separator()
panel.add_progress("Processing", 0.75)

# Use with VisualDebugger (panels are now composite annotations)
from visual_debugger.annotations import InfoPanelAnnotation
panel_ann = InfoPanelAnnotation(panel=panel)
result = vd.visual_debug(img, [panel_ann])
```

### Image Composition

```python
from visual_debugger.composition import ImageCompositor, LayoutDirection

compositor = ImageCompositor()

# Create image grid
grid = compositor.create_grid(
    images=[img1, img2, img3, img4],
    cols=2,
    labels=["Step 1", "Step 2", "Step 3", "Step 4"]
)

# Create before/after comparison
comparison = compositor.create_comparison(
    before=original_img,
    after=processed_img,
    before_label="Original",
    after_label="Enhanced"
)
```

## 🔧 Advanced Features

### Type-Specific Annotations

Each annotation type is a dedicated class with only relevant parameters:

```python
# No more generic dictionaries or enums!
circle_ann = CircleAnnotation(
    center=(100, 100),
    radius=30,
    color=(255, 0, 0),
    thickness=2,
    filled=False
)

# Bounding boxes with labels
bbox_ann = BoundingBoxAnnotation(
    bbox=(x, y, width, height),
    label="Car 92%",  # Include any info in the label
    color=(0, 255, 0)
)
```

### Visitor Pattern Processing

The system uses a clean visitor pattern for extensibility:

```python
class CustomProcessor(AnnotationProcessor):
    def render_custom(self, annotation):
        # Your custom rendering logic
        pass
```

### Boundary Detection

All annotations can calculate their visual footprint:

```python
ann = CircleAnnotation(center=(100, 100), radius=30)
x_min, y_min, x_max, y_max = ann.get_bounding_box()
# Returns: (69, 69, 131, 131) accounting for thickness
```

### Factory Functions

Convenient factory functions for quick annotation creation:

```python
# Instead of: PointAnnotation(position=(100, 200), color=(255, 0, 0))
# You can use: point(100, 200, color=(255, 0, 0))

annotations = [
    point(100, 200),
    labeled_point(200, 200, "Target"),
    circle(300, 300, 50),
    rectangle(400, 400, 100, 100),
    line(0, 0, 500, 500, arrow=True),
    text("Status: OK", 10, 30),
    bbox(50, 50, 200, 150, label="Detection")
]
```

## 🏗️ Architecture

```
visual_debugger/
├── visual_debugger.py      # Main orchestrator
├── annotations.py           # Type-specific annotation classes
├── annotation_processor.py  # Visitor pattern renderer
├── info_panel.py           # Dashboard overlay system
├── composition.py          # Image layout and grids
├── image_processor.py      # Core image operations
└── utils.py               # Utilities and helpers
```

## 🧪 Testing

Comprehensive smoke tests are included:

```bash
# Run all smoke tests
python -m smoke_tests.test_01_type_specific_annotations
python -m smoke_tests.test_02_annotation_processor
python -m smoke_tests.test_03_image_composition
python -m smoke_tests.test_04_info_panel
python -m smoke_tests.test_05_visual_debugger_integration

# Visual showcase with sample image
python -m smoke_tests.test_08_visual_showcase
```

## 📊 Performance

- Optimized for real-time visualization
- Efficient numpy operations for masks
- Lazy evaluation where possible
- Minimal memory footprint

## 🔄 Migration from Legacy API

If upgrading from the enum-based API:

```python
# Old style (deprecated)
from visual_debugger import Annotation, AnnotationType
ann = Annotation(type=AnnotationType.POINT, coordinates=(100, 100))

# New style (recommended)
from visual_debugger.annotations import PointAnnotation
ann = PointAnnotation(position=(100, 100))

# Or use factory functions
from visual_debugger.annotations import point
ann = point(100, 100)
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- New annotation types
- Performance optimizations
- Additional colormaps for masks
- Export formats (video, GIF)

## 📄 License

MIT License - see LICENSE file for details

## 🌟 Examples Gallery

Check out `smoke_tests/test_08_outputs/` after running the visual showcase for examples of all annotation types in action.

## 💡 Tips & Best Practices

1. **Use factory functions** for cleaner code
2. **Leverage type hints** - all classes are fully typed
3. **Check boundaries** with `get_bounding_box()` before rendering
4. **Compose views** for side-by-side comparisons
5. **Add info panels** for professional debugging output
6. **Use process steps** for organized output naming

## 🔗 Related Projects

- OpenCV: Core image processing
- NumPy: Efficient array operations
- Pillow: Additional image format support

---

Built with ❤️ for the computer vision community