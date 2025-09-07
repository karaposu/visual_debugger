# Visual Debugger Enhancement Proposal

## Overview

This document outlines proposed enhancements to the `visual_debugger` package to better support video analysis debugging, particularly for integration with VideoKurt and similar video processing frameworks.

## Current Capabilities

The visual_debugger currently supports:
- Basic annotations (points, circles, rectangles, lines)
- Labels with annotations
- Mask overlays
- Orientation visualization (pitch/yaw/roll)
- Image concatenation
- Side-by-side comparisons

## Proposed Enhancements

### 1. Time-Series Visualization

**Need:** Display temporal data trends directly on video frames.

**Proposed Annotation Types:**
```python
class AnnotationType(Enum):
    GRAPH = auto()      # Line graph overlay
    BAR_CHART = auto()  # Bar chart for discrete values
    TIMELINE = auto()   # Horizontal timeline bar
    SPARKLINE = auto()  # Minimal inline graph
```

**Implementation Example:**
```python
@dataclass
class GraphData:
    """Data for graph annotations."""
    values: np.ndarray
    x_range: Optional[Tuple[int, int]] = None
    y_range: Optional[Tuple[float, float]] = None
    style: str = 'line'  # 'line', 'bar', 'scatter', 'area'
    fill: bool = False
    grid: bool = False
    labels: Optional[List[str]] = None  # Axis labels

# Usage
annotation = Annotation(
    type=AnnotationType.GRAPH,
    coordinates=(10, 10),  # Top-left position
    size=(200, 100),       # Width, height
    graph_data=GraphData(
        values=frame_diff_values,
        style='line',
        fill=True
    ),
    color=(0, 255, 0)
)
```

### 2. Information Display

**Need:** Display multiple metrics and information in organized formats.

**Proposed Annotation Types:**
```python
class AnnotationType(Enum):
    INFO_BOX = auto()    # Multi-line text box with background
    TABLE = auto()       # Tabular data display
    DASHBOARD = auto()   # Metrics dashboard panel
```

**Implementation Example:**
```python
@dataclass
class InfoBoxData:
    """Data for info box annotations."""
    lines: List[str]
    title: Optional[str] = None
    background_color: Tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.7
    padding: int = 10
    border: bool = True

# Usage
annotation = Annotation(
    type=AnnotationType.INFO_BOX,
    coordinates=(10, 50),
    info_data=InfoBoxData(
        title="Detection Results",
        lines=[
            "Scene: #3",
            "Activity: Active",
            "Motion: 45%",
            "Objects: 5"
        ]
    ),
    color=(255, 255, 255)
)
```

### 3. Progress and Status Indicators

**Need:** Show continuous values and states visually.

**Proposed Annotation Types:**
```python
class AnnotationType(Enum):
    PROGRESS_BAR = auto()   # Horizontal/vertical progress bar
    METER = auto()          # Gauge/meter visualization
    INDICATOR = auto()      # Status indicator (LED-style)
    SLIDER = auto()         # Value slider visualization
```

**Implementation Example:**
```python
@dataclass
class ProgressData:
    """Data for progress/meter annotations."""
    value: float  # 0.0 to 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    show_percentage: bool = True
    orientation: str = 'horizontal'  # 'horizontal' or 'vertical'
    style: str = 'bar'  # 'bar', 'circle', 'gauge'

# Usage
annotation = Annotation(
    type=AnnotationType.PROGRESS_BAR,
    coordinates=(10, 100),
    size=(200, 20),
    progress_data=ProgressData(
        value=0.75,
        show_percentage=True
    ),
    labels="Processing",
    color=(0, 255, 0)
)
```

### 4. Advanced Overlays

**Need:** Better support for heatmaps and transparent overlays.

**Proposed Annotation Types:**
```python
class AnnotationType(Enum):
    HEATMAP = auto()        # Proper heatmap with colormaps
    ALPHA_OVERLAY = auto()  # Transparent overlay
    GRADIENT = auto()       # Gradient overlay
    CONTOUR_MAP = auto()    # Contour visualization
```

**Implementation Example:**
```python
@dataclass
class HeatmapData:
    """Data for heatmap annotations."""
    data: np.ndarray
    colormap: str = 'jet'  # OpenCV colormap name
    alpha: float = 0.5
    normalize: bool = True
    show_colorbar: bool = False

# Usage
annotation = Annotation(
    type=AnnotationType.HEATMAP,
    heatmap_data=HeatmapData(
        data=motion_heatmap,
        colormap='hot',
        alpha=0.4
    )
)
```

### 5. Directional Indicators

**Need:** Show movement, direction, and vectors.

**Proposed Annotation Types:**
```python
class AnnotationType(Enum):
    ARROW = auto()          # Single directional arrow
    VECTOR_FIELD = auto()   # Multiple vectors (optical flow)
    TRAJECTORY = auto()     # Path/trajectory visualization
    FLOW_LINES = auto()     # Flow streamlines
```

**Implementation Example:**
```python
@dataclass
class VectorData:
    """Data for vector/arrow annotations."""
    start_points: np.ndarray
    end_points: np.ndarray
    magnitudes: Optional[np.ndarray] = None
    scale: float = 1.0
    arrow_size: float = 0.3
    show_magnitude: bool = False

# Usage
annotation = Annotation(
    type=AnnotationType.VECTOR_FIELD,
    vector_data=VectorData(
        start_points=flow_points,
        end_points=flow_destinations,
        scale=2.0
    ),
    color=(255, 255, 0)
)
```

### 6. Enhanced Positioning System

**Need:** More flexible positioning options beyond absolute coordinates.

**Proposed Enhancement:**
```python
@dataclass
class Annotation:
    # Existing fields...
    
    # New positioning fields
    position_mode: str = 'absolute'  # 'absolute', 'relative', 'anchor'
    anchor: Optional[str] = None  # 'top-left', 'center', 'bottom-right', etc.
    offset: Tuple[int, int] = (0, 0)  # Offset from anchor
    margin: int = 0  # Margin from edges
    z_order: int = 0  # Drawing order (higher = on top)

# Usage
annotation = Annotation(
    type=AnnotationType.INFO_BOX,
    position_mode='anchor',
    anchor='top-right',
    offset=(-10, 10),  # 10 pixels from right, 10 from top
    info_data=InfoBoxData(lines=["Status: Active"])
)
```

### 7. Animation and Temporal Effects

**Need:** Support for animations and time-based effects.

**Proposed Enhancement:**
```python
@dataclass
class Annotation:
    # Existing fields...
    
    # Animation fields
    fade_in: int = 0  # Frames to fade in
    fade_out: int = 0  # Frames to fade out
    duration: Optional[int] = None  # Total frames to display
    blink_rate: int = 0  # Blinking rate (0 = no blink)
    animation: Optional[str] = None  # 'pulse', 'slide', 'rotate'
    
    # Time-based visibility
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
```

### 8. Annotation Groups and Layers

**Need:** Manage multiple related annotations as groups.

**Proposed Enhancement:**
```python
@dataclass
class AnnotationGroup:
    """Group related annotations together."""
    name: str
    annotations: List[Annotation]
    visible: bool = True
    opacity: float = 1.0
    position: Optional[Tuple[int, int]] = None  # Group position offset
    
    def toggle(self):
        """Toggle visibility of entire group."""
        self.visible = not self.visible
    
    def move(self, dx: int, dy: int):
        """Move all annotations in group."""
        for ann in self.annotations:
            if ann.coordinates:
                ann.coordinates = (
                    ann.coordinates[0] + dx,
                    ann.coordinates[1] + dy
                )

# Usage
motion_group = AnnotationGroup(
    name="motion_indicators",
    annotations=[
        motion_vector_annotation,
        motion_heatmap_annotation,
        motion_stats_annotation
    ]
)
```

## Implementation Details

### Enhanced ImageProcessor Methods

```python
class ImageProcessor:
    def put_graph_on_image(self, image, annotation: Annotation):
        """Draw a graph overlay on the image."""
        data = annotation.graph_data
        x, y = annotation.coordinates
        width, height = annotation.size or (200, 100)
        
        # Create semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Calculate graph points
        values = data.values
        if len(values) > width:
            # Downsample if too many points
            indices = np.linspace(0, len(values) - 1, width).astype(int)
            values = values[indices]
        
        # Normalize values
        min_val = data.y_range[0] if data.y_range else values.min()
        max_val = data.y_range[1] if data.y_range else values.max()
        normalized = (values - min_val) / (max_val - min_val + 1e-8)
        
        # Draw graph
        points = []
        for i, val in enumerate(normalized):
            px = x + int(i * width / len(values))
            py = y + height - int(val * (height - 10)) - 5
            points.append((px, py))
        
        # Draw based on style
        if data.style == 'line':
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i+1], annotation.color, 2)
        elif data.style == 'bar':
            bar_width = max(1, width // len(values))
            for i, (px, py) in enumerate(points):
                cv2.rectangle(image, (px, y + height), 
                            (px + bar_width, py), annotation.color, -1)
        
        # Draw grid if requested
        if data.grid:
            # Horizontal grid lines
            for i in range(0, height, height // 4):
                cv2.line(image, (x, y + i), (x + width, y + i), (50, 50, 50), 1)
            # Vertical grid lines
            for i in range(0, width, width // 4):
                cv2.line(image, (x + i, y), (x + i, y + height), (50, 50, 50), 1)
    
    def put_info_box_on_image(self, image, annotation: Annotation):
        """Draw an info box with multiple lines of text."""
        data = annotation.info_data
        x, y = annotation.coordinates
        
        lines = data.lines
        if data.title:
            lines = [data.title, "─" * len(data.title)] + lines
        
        # Calculate box dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = annotation.font_scale or 0.5
        thickness = 1
        
        max_width = 0
        total_height = data.padding * 2
        line_heights = []
        
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, w)
            line_heights.append(h)
            total_height += h + 5
        
        box_width = max_width + data.padding * 2
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + box_width, y + total_height),
                     data.background_color, -1)
        cv2.addWeighted(overlay, data.background_alpha, image, 
                       1 - data.background_alpha, 0, image)
        
        # Draw border if requested
        if data.border:
            cv2.rectangle(image, (x, y), (x + box_width, y + total_height),
                         annotation.color, 2)
        
        # Draw text
        current_y = y + data.padding + line_heights[0]
        for i, line in enumerate(lines):
            # Special handling for title separator
            if i == 1 and data.title and line.startswith("─"):
                cv2.line(image, (x + data.padding, current_y - 5),
                        (x + box_width - data.padding, current_y - 5),
                        annotation.color, 1)
            else:
                cv2.putText(image, line, (x + data.padding, current_y),
                           font, font_scale, annotation.color, thickness)
            
            if i < len(line_heights) - 1:
                current_y += line_heights[i] + 5
    
    def put_progress_bar_on_image(self, image, annotation: Annotation):
        """Draw a progress bar."""
        data = annotation.progress_data
        x, y = annotation.coordinates
        width, height = annotation.size or (200, 20)
        
        # Calculate fill amount
        value = (data.value - data.min_val) / (data.max_val - data.min_val)
        value = np.clip(value, 0, 1)
        
        if data.orientation == 'horizontal':
            # Draw background
            cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
            
            # Draw filled portion
            fill_width = int(width * value)
            cv2.rectangle(image, (x, y), (x + fill_width, y + height),
                         annotation.color, -1)
            
            # Draw border
            cv2.rectangle(image, (x, y), (x + width, y + height), 
                         (255, 255, 255), 2)
            
            # Draw text
            if data.show_percentage:
                text = f"{int(value * 100)}%"
                if annotation.labels:
                    text = f"{annotation.labels}: {text}"
                cv2.putText(image, text, (x + 5, y + height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Vertical orientation
            # Draw background
            cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
            
            # Draw filled portion (from bottom up)
            fill_height = int(height * value)
            cv2.rectangle(image, (x, y + height - fill_height), 
                         (x + width, y + height), annotation.color, -1)
            
            # Draw border
            cv2.rectangle(image, (x, y), (x + width, y + height),
                         (255, 255, 255), 2)
    
    def put_heatmap_on_image(self, image, annotation: Annotation):
        """Apply a heatmap overlay."""
        data = annotation.heatmap_data
        heatmap = data.data
        
        # Resize heatmap to match image if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize if requested
        if data.normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        colormap = getattr(cv2, f'COLORMAP_{data.colormap.upper()}', cv2.COLORMAP_JET)
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), colormap)
        
        # Blend with original image
        cv2.addWeighted(image, 1 - data.alpha, heatmap_colored, data.alpha, 0, image)
    
    def put_vector_field_on_image(self, image, annotation: Annotation):
        """Draw a vector field (e.g., optical flow)."""
        data = annotation.vector_data
        
        for start, end in zip(data.start_points, data.end_points):
            start = tuple(start.astype(int))
            end = tuple((end * data.scale).astype(int))
            
            # Draw arrow
            cv2.arrowedLine(image, start, end, annotation.color, 2,
                           tipLength=data.arrow_size)
            
            # Draw magnitude if requested
            if data.show_magnitude and data.magnitudes is not None:
                mag_text = f"{data.magnitudes[i]:.1f}"
                cv2.putText(image, mag_text, start, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                           annotation.color, 1)
```

## Usage Examples

### Example 1: Video Analysis Dashboard
```python
# Create dashboard with multiple visualizations
annotations = [
    # Info box in top-left
    Annotation(
        type=AnnotationType.INFO_BOX,
        position_mode='anchor',
        anchor='top-left',
        offset=(10, 10),
        info_data=InfoBoxData(
            title="Analysis Status",
            lines=[
                f"Frame: {frame_idx}/{total_frames}",
                f"Scene: {scene_id}",
                f"Activity: {'Active' if is_active else 'Idle'}",
                f"Motion Level: {motion_level:.1%}"
            ]
        )
    ),
    
    # Graph in top-right
    Annotation(
        type=AnnotationType.GRAPH,
        position_mode='anchor',
        anchor='top-right',
        offset=(-210, 10),
        size=(200, 80),
        graph_data=GraphData(
            values=motion_history,
            style='area',
            fill=True
        )
    ),
    
    # Progress bar at bottom
    Annotation(
        type=AnnotationType.PROGRESS_BAR,
        position_mode='anchor',
        anchor='bottom-center',
        offset=(-100, -30),
        size=(200, 20),
        progress_data=ProgressData(
            value=frame_idx / total_frames,
            show_percentage=True
        ),
        labels="Processing"
    )
]
```

### Example 2: Motion Analysis Overlay
```python
annotations = [
    # Motion heatmap
    Annotation(
        type=AnnotationType.HEATMAP,
        heatmap_data=HeatmapData(
            data=motion_accumulation,
            colormap='hot',
            alpha=0.3
        )
    ),
    
    # Optical flow vectors
    Annotation(
        type=AnnotationType.VECTOR_FIELD,
        vector_data=VectorData(
            start_points=flow_points,
            end_points=flow_vectors,
            scale=2.0,
            arrow_size=0.2
        ),
        color=(0, 255, 255)
    ),
    
    # Motion statistics
    Annotation(
        type=AnnotationType.INFO_BOX,
        position_mode='anchor',
        anchor='bottom-left',
        info_data=InfoBoxData(
            title="Motion Stats",
            lines=[
                f"Avg Flow: {avg_flow:.2f}",
                f"Max Flow: {max_flow:.2f}",
                f"Active Pixels: {active_pixels}"
            ]
        )
    )
]
```

## Benefits of These Enhancements

1. **Comprehensive Visualization**: Support for all common video analysis visualizations
2. **Flexible Positioning**: Anchor-based positioning adapts to different frame sizes
3. **Professional Appearance**: Proper backgrounds, borders, and styling options
4. **Performance**: Efficient OpenCV-based rendering
5. **Extensibility**: Clean architecture for adding new visualization types
6. **Ease of Use**: High-level API with sensible defaults

## Implementation Priority

### Phase 1: Core Enhancements (High Priority)
- Graph and timeline support
- Info boxes with proper backgrounds
- Progress bars and indicators
- Enhanced positioning system

### Phase 2: Advanced Features (Medium Priority)
- Heatmap overlays
- Vector field visualization
- Animation support
- Annotation groups

### Phase 3: Polish (Low Priority)
- Additional graph styles
- Complex dashboards
- Custom colormaps
- Export utilities

## Backward Compatibility

All enhancements should be additive, maintaining full backward compatibility with existing code. New features use optional parameters with sensible defaults.

## Conclusion

These enhancements would transform `visual_debugger` into a comprehensive visualization toolkit for video analysis, making it invaluable for debugging and presenting results from systems like VideoKurt.