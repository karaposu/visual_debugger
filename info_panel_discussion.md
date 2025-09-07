# Info Panel Implementation Discussion

## Overview
The Info Panel is an **additional** unified information display component that consolidates dashboard-style metrics into a single, well-organized panel. It complements the existing annotation system by providing an alternative way to display information that isn't tied to specific image coordinates. Both approaches are valuable and serve different purposes in visual debugging.

## Core Concept

### Traditional Approach (Essential for Many Use Cases)
```python
# Direct text annotations - perfect for labeling specific image regions
annotations = [
    Annotation(type=AnnotationType.POINT_AND_LABEL, 
               coordinates=(150, 200), labels="Object A"),  # Label at object location
    Annotation(type=AnnotationType.POINT_AND_LABEL, 
               coordinates=(300, 150), labels="Anomaly"),    # Mark specific point
    Annotation(type=AnnotationType.POINT_AND_LABEL, 
               coordinates=(400, 350), labels="Region 3"),   # Label image region
]
# These annotations are tied to specific image content - not suitable for panels
```

### Info Panel Approach (Alternative for Dashboard-Style Information)
```python
# Consolidated metrics that aren't tied to specific image coordinates
panel = InfoPanel(position="top-left")
panel.add("FPS", 30)
panel.add("Frame", 145)
panel.add("Total Objects", 5)
panel.add("Avg Motion", "45%")

# Use BOTH approaches together
annotations = [
    # Panel for general metrics
    panel.to_annotation(),
    
    # Traditional annotations for specific image features
    Annotation(type=AnnotationType.RECTANGLE, 
               coordinates=(100, 100, 50, 50)),
    Annotation(type=AnnotationType.POINT_AND_LABEL,
               coordinates=(125, 125), labels="Detection")
]
```

### When to Use Each Approach

**Use Traditional Annotations When:**
- Labeling specific objects or regions in the image
- Marking detection results at their actual locations
- Drawing attention to particular image features
- Annotations need to move with tracked objects
- Visual connection between label and image content is important

**Use Info Panel When:**
- Displaying general metrics (FPS, frame count, statistics)
- Showing system status or debugging information
- Presenting aggregated data (totals, averages, summaries)
- Information isn't tied to specific image coordinates
- Want a clean, dashboard-like appearance

## Implementation Design

### 1. Basic Info Panel Class

```python
# info_panel.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any
from enum import Enum
import numpy as np

class PanelPosition(Enum):
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    TOP_CENTER = "top-center"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_CENTER = "bottom-center"
    CENTER_LEFT = "center-left"
    CENTER_RIGHT = "center-right"
    CENTER = "center"

@dataclass
class PanelStyle:
    """Visual style configuration for the panel"""
    background_color: Tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.7
    text_color: Tuple[int, int, int] = (255, 255, 255)
    title_color: Tuple[int, int, int] = (255, 255, 100)
    border_color: Optional[Tuple[int, int, int]] = (255, 255, 255)
    border_thickness: int = 2
    padding: int = 15
    line_spacing: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
    min_width: int = 200
    show_background: bool = True

class InfoPanel:
    """A clean, organized panel for displaying information on images"""
    
    def __init__(self, 
                 position: Union[str, PanelPosition] = PanelPosition.TOP_LEFT,
                 title: Optional[str] = None,
                 style: Optional[PanelStyle] = None,
                 margin: int = 10):
        self.position = PanelPosition(position) if isinstance(position, str) else position
        self.title = title
        self.style = style or PanelStyle()
        self.margin = margin
        self._entries: List[Tuple[Optional[str], Any]] = []
        
    def add(self, key: Optional[str] = None, value: Any = None):
        """Add a key-value pair or just a value to the panel"""
        if key is None and value is None:
            self._entries.append((None, ""))  # Empty line
        elif key is None:
            self._entries.append((None, str(value)))  # Just value
        else:
            self._entries.append((str(key), str(value)))  # Key-value pair
        return self
    
    def add_line(self, text: str = ""):
        """Add a simple text line"""
        self._entries.append((None, text))
        return self
    
    def add_separator(self, char: str = "─"):
        """Add a separator line"""
        self._entries.append((None, char * 20))
        return self
    
    def clear(self):
        """Clear all entries"""
        self._entries.clear()
        return self
```

### 2. Rendering Implementation

```python
class InfoPanel:
    # ... previous methods ...
    
    def calculate_dimensions(self, font=cv2.FONT_HERSHEY_SIMPLEX):
        """Calculate the required panel dimensions"""
        max_width = self.style.min_width
        total_height = self.style.padding * 2
        
        # Add title dimensions if present
        if self.title:
            (title_w, title_h), _ = cv2.getTextSize(
                self.title, font, self.style.font_scale * 1.2, 
                self.style.font_thickness)
            max_width = max(max_width, title_w + self.style.padding * 2)
            total_height += title_h + self.style.line_spacing * 2
        
        # Calculate dimensions for each entry
        for key, value in self._entries:
            if key:
                text = f"{key}: {value}"
            else:
                text = value
                
            (text_w, text_h), _ = cv2.getTextSize(
                text, font, self.style.font_scale, 
                self.style.font_thickness)
            
            max_width = max(max_width, text_w + self.style.padding * 2)
            total_height += text_h + self.style.line_spacing
        
        return max_width, total_height
    
    def calculate_position(self, image_shape, panel_width, panel_height):
        """Calculate the top-left corner position based on anchor"""
        img_h, img_w = image_shape[:2]
        
        positions = {
            PanelPosition.TOP_LEFT: (self.margin, self.margin),
            PanelPosition.TOP_RIGHT: (img_w - panel_width - self.margin, self.margin),
            PanelPosition.TOP_CENTER: ((img_w - panel_width) // 2, self.margin),
            PanelPosition.BOTTOM_LEFT: (self.margin, img_h - panel_height - self.margin),
            PanelPosition.BOTTOM_RIGHT: (img_w - panel_width - self.margin, 
                                        img_h - panel_height - self.margin),
            PanelPosition.BOTTOM_CENTER: ((img_w - panel_width) // 2, 
                                          img_h - panel_height - self.margin),
            PanelPosition.CENTER_LEFT: (self.margin, (img_h - panel_height) // 2),
            PanelPosition.CENTER_RIGHT: (img_w - panel_width - self.margin, 
                                         (img_h - panel_height) // 2),
            PanelPosition.CENTER: ((img_w - panel_width) // 2, 
                                   (img_h - panel_height) // 2),
        }
        
        return positions.get(self.position, (self.margin, self.margin))
    
    def render(self, image: np.ndarray) -> np.ndarray:
        """Render the info panel onto the image"""
        if not self._entries and not self.title:
            return image
        
        # Calculate dimensions
        panel_w, panel_h = self.calculate_dimensions()
        x, y = self.calculate_position(image.shape, panel_w, panel_h)
        
        # Draw semi-transparent background if enabled
        if self.style.show_background:
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h),
                         self.style.background_color, -1)
            cv2.addWeighted(overlay, self.style.background_alpha, 
                           image, 1 - self.style.background_alpha, 0, image)
        
        # Draw border if specified
        if self.style.border_color:
            cv2.rectangle(image, (x, y), (x + panel_w, y + panel_h),
                         self.style.border_color, self.style.border_thickness)
        
        # Draw content
        font = cv2.FONT_HERSHEY_SIMPLEX
        current_y = y + self.style.padding
        
        # Draw title if present
        if self.title:
            title_scale = self.style.font_scale * 1.2
            (_, title_h), _ = cv2.getTextSize(self.title, font, title_scale, 
                                              self.style.font_thickness)
            current_y += title_h
            cv2.putText(image, self.title, (x + self.style.padding, current_y),
                       font, title_scale, self.style.title_color, 
                       self.style.font_thickness + 1)
            
            # Draw separator after title
            current_y += self.style.line_spacing
            cv2.line(image, (x + self.style.padding, current_y),
                    (x + panel_w - self.style.padding, current_y),
                    self.style.text_color, 1)
            current_y += self.style.line_spacing
        
        # Draw entries
        for key, value in self._entries:
            if key:
                text = f"{key}: {value}"
            else:
                text = value
            
            (_, text_h), _ = cv2.getTextSize(text, font, self.style.font_scale,
                                             self.style.font_thickness)
            current_y += text_h
            
            cv2.putText(image, text, (x + self.style.padding, current_y),
                       font, self.style.font_scale, self.style.text_color,
                       self.style.font_thickness)
            
            current_y += self.style.line_spacing
        
        return image
```

### 3. Integration with Existing Annotation System

```python
class InfoPanel:
    # ... previous methods ...
    
    def to_annotation(self) -> 'Annotation':
        """Convert InfoPanel to a custom Annotation for compatibility"""
        # This would require adding INFO_PANEL to AnnotationType enum
        return Annotation(
            type=AnnotationType.INFO_PANEL,
            info_panel=self
        )

# Enhanced ImageProcessor
class ImageProcessor:
    def put_annotation_on_image(self, image, annotation: Annotation):
        # ... existing code ...
        elif annotation.type == AnnotationType.INFO_PANEL:
            annotation.info_panel.render(image)
```

## Usage Examples

### Example 1: Basic Metrics Panel
```python
# Create a simple metrics panel
panel = InfoPanel(position="top-right", title="Performance Metrics")
panel.add("FPS", 30)
panel.add("Processing Time", "45ms")
panel.add("Memory Usage", "128MB")
panel.add_separator()
panel.add("Total Frames", 1523)

# Apply to image
debugger.visual_debug(image, [panel.to_annotation()])
```

### Example 2: Object Detection Panel
```python
# Detection results panel
detections_panel = InfoPanel(
    position="top-left",
    title="Detection Results",
    style=PanelStyle(
        background_alpha=0.8,
        text_color=(0, 255, 0),
        font_scale=0.6
    )
)

detections_panel.add("Total Objects", len(detections))
detections_panel.add_separator()
for i, det in enumerate(detections[:5]):  # Show top 5
    detections_panel.add(f"Object {i+1}", f"{det.class_name} ({det.confidence:.2f})")

debugger.visual_debug(image, [detections_panel.to_annotation()])
```

### Example 3: Multi-Panel Layout
```python
# Multiple panels for different information categories
stats_panel = InfoPanel("top-left", "Statistics")
stats_panel.add("Mean", f"{np.mean(data):.2f}")
stats_panel.add("Std Dev", f"{np.std(data):.2f}")
stats_panel.add("Min/Max", f"{np.min(data):.1f}/{np.max(data):.1f}")

status_panel = InfoPanel("top-right", "System Status")
status_panel.add("State", "Processing")
status_panel.add("Queue", "15 items")
status_panel.add("Errors", "0")

info_panel = InfoPanel("bottom-left", "Frame Info")
info_panel.add("Frame", frame_number)
info_panel.add("Timestamp", timestamp)
info_panel.add("Scene", scene_id)

# Apply all panels
annotations = [
    stats_panel.to_annotation(),
    status_panel.to_annotation(),
    info_panel.to_annotation(),
    # ... other annotations
]
debugger.visual_debug(image, annotations)
```

### Example 4: Dynamic Panel with Conditional Styling
```python
# Create panel with dynamic styling based on conditions
def create_status_panel(metrics):
    style = PanelStyle()
    
    # Change background color based on status
    if metrics['error_rate'] > 0.1:
        style.background_color = (0, 0, 100)  # Red tint
        style.title_color = (100, 100, 255)   # Red title
    elif metrics['warning_count'] > 0:
        style.background_color = (0, 50, 100)  # Yellow tint
        style.title_color = (100, 255, 255)    # Yellow title
    else:
        style.background_color = (0, 50, 0)    # Green tint
        style.title_color = (100, 255, 100)    # Green title
    
    panel = InfoPanel("top-center", "System Health", style)
    panel.add("Status", "🟢 OK" if metrics['error_rate'] == 0 else "🔴 ERROR")
    panel.add("Uptime", metrics['uptime'])
    panel.add("Error Rate", f"{metrics['error_rate']:.1%}")
    panel.add("Warnings", metrics['warning_count'])
    
    return panel
```

### Example 5: Compact Builder Pattern
```python
# Fluent interface for panel creation
panel = (InfoPanel("top-left", "Debug Info")
    .add("Frame", frame_num)
    .add("Time", f"{elapsed:.2f}s")
    .add_separator()
    .add("Objects", count)
    .add("Motion", f"{motion:.1%}")
)

debugger.visual_debug(image, [panel.to_annotation()])
```

## Advanced Features

### 1. Table Support
```python
class InfoPanel:
    def add_table(self, headers: List[str], rows: List[List[Any]]):
        """Add a formatted table to the panel"""
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add headers
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        self.add_line(header_line)
        self.add_line("─" * len(header_line))
        
        # Add rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) 
                                 for cell, w in zip(row, col_widths))
            self.add_line(row_line)
        
        return self

# Usage
panel.add_table(
    headers=["Class", "Count", "Conf"],
    rows=[
        ["Person", 3, 0.95],
        ["Car", 2, 0.87],
        ["Dog", 1, 0.92]
    ]
)
```

### 2. Progress Bars
```python
class InfoPanel:
    def add_progress(self, label: str, value: float, max_value: float = 1.0):
        """Add a text-based progress bar"""
        percentage = value / max_value
        bar_width = 20
        filled = int(bar_width * percentage)
        bar = "█" * filled + "░" * (bar_width - filled)
        self.add(label, f"{bar} {percentage:.1%}")
        return self

# Usage
panel.add_progress("Processing", 0.75)  # Shows: Processing: ███████████████░░░░░ 75.0%
```

### 3. Auto-Layout for Multiple Values
```python
class InfoPanel:
    def add_metrics(self, metrics: dict, format_spec: str = ".2f"):
        """Add multiple metrics with automatic formatting"""
        for key, value in metrics.items():
            if isinstance(value, float):
                self.add(key, f"{value:{format_spec}}")
            else:
                self.add(key, value)
        return self

# Usage
panel.add_metrics({
    "accuracy": 0.9234,
    "precision": 0.8756,
    "recall": 0.9012,
    "f1_score": 0.8882
})
```

## Implementation Benefits

1. **Clean Code**: One panel replaces dozens of text annotations
2. **Automatic Layout**: No manual coordinate calculations
3. **Consistent Styling**: All information follows the same visual style
4. **Easy Updates**: Add/remove metrics without repositioning
5. **Professional Appearance**: Looks like a proper debugging dashboard
6. **Reusable**: Panel configurations can be saved and reused
7. **Flexible**: Works with existing annotation system
8. **Maintainable**: Changes to panel layout don't affect other code

## Technical Considerations

### Performance
- Cache rendered panels if content doesn't change between frames
- Pre-calculate dimensions for static panels
- Use numpy operations for bulk pixel operations

### Memory
- Panels are lightweight - just store text and style information
- Rendering happens on-demand, no persistent image storage

### Thread Safety
- Panel objects should be immutable after creation for thread safety
- Or use copy-on-write semantics for modifications

## Integration Roadmap

### Phase 1: Basic Implementation
- Core InfoPanel class with basic text support
- Integration with existing Annotation system
- Basic positioning (corners only)

### Phase 2: Enhanced Features
- Table support
- Progress bars
- Custom separators
- All 9 position anchors

### Phase 3: Advanced Features
- Caching for performance
- Animation support (fade in/out)
- Multi-column layout
- Custom fonts and icons

## Conclusion

The Info Panel provides a clean, professional way to display debugging information without cluttering the image with scattered text annotations. It's easy to use, maintains visual consistency, and significantly reduces the complexity of adding informational overlays to images.