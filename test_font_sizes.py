#!/usr/bin/env python3
"""Demonstrate font size options in various annotations"""

import numpy as np
import cv2
from visual_debugger import VisualDebugger
from visual_debugger.annotations import *
from visual_debugger.info_panel import InfoPanel, PanelPosition, PanelStyle

# Create test image
img = np.ones((600, 800, 3), dtype=np.uint8) * 50

vd = VisualDebugger(output='return')

# All annotations with text and their font_scale parameters:
annotations = []

# 1. TextAnnotation - has font_scale
annotations.extend([
    TextAnnotation(text="Small Text", position=(50, 50), 
                   font_scale=0.3, color=(255, 255, 255)),
    TextAnnotation(text="Normal Text", position=(50, 80), 
                   font_scale=0.5, color=(255, 255, 255)),
    TextAnnotation(text="Large Text", position=(50, 120), 
                   font_scale=0.8, color=(255, 255, 255)),
    TextAnnotation(text="Huge Text", position=(50, 170), 
                   font_scale=1.2, font_thickness=2, color=(255, 255, 255)),
])

# 2. LabeledPointAnnotation - has font_scale  
annotations.extend([
    LabeledPointAnnotation(position=(400, 50), label="Tiny", 
                           font_scale=0.3, color=(255, 100, 100)),
    LabeledPointAnnotation(position=(400, 100), label="Normal", 
                           font_scale=0.5, color=(255, 100, 100)),
    LabeledPointAnnotation(position=(400, 150), label="Big", 
                           font_scale=0.8, color=(255, 100, 100)),
])

# 3. LabeledPointsAnnotation - has font_scale
annotations.append(
    LabeledPointsAnnotation(
        points=[(550, 50), (550, 100), (550, 150)],
        labels=["A", "B", "C"],
        font_scale=0.7,  # All labels same size
        color=(100, 255, 100)
    )
)

# 4. LabeledCircleAnnotation - has font_scale
annotations.extend([
    LabeledCircleAnnotation(center=(100, 300), radius=30, label="Small Label",
                            font_scale=0.4, color=(255, 255, 100)),
    LabeledCircleAnnotation(center=(250, 300), radius=30, label="Large Label",
                            font_scale=0.7, color=(255, 255, 100)),
])

# 5. LabeledLineAnnotation - has font_scale
annotations.append(
    LabeledLineAnnotation(start=(50, 400), end=(300, 400), 
                          label="Distance", font_scale=0.6,
                          color=(100, 255, 255))
)

# 6. BoundingBoxAnnotation - has font_scale
annotations.extend([
    BoundingBoxAnnotation(bbox=(50, 450, 100, 60), label="Small Box Label",
                          font_scale=0.4, color=(255, 100, 255)),
    BoundingBoxAnnotation(bbox=(200, 450, 100, 60), label="Large Box Label",
                          font_scale=0.7, font_thickness=2, color=(255, 100, 255)),
])

# 7. InfoPanel - font_scale via PanelStyle
panel_style = PanelStyle(
    font_scale=0.6,          # Larger font for panel
    font_thickness=1,
    background_alpha=0.8
)
panel = InfoPanel(position=PanelPosition.TOP_RIGHT, 
                  title="Panel Font Size", style=panel_style)
panel.add("Font Scale", "0.6")
panel.add("Thickness", "1")

annotations.append(InfoPanelAnnotation(panel=panel))

# Render all
result = vd.visual_debug(img, annotations)

if result is not None:
    cv2.imwrite("font_sizes_demo.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print("✓ Saved font_sizes_demo.jpg")
    print("\n📝 Annotations with font_scale parameter:")
    print("  1. TextAnnotation         - font_scale, font_thickness")
    print("  2. LabeledPointAnnotation - font_scale, font_thickness")  
    print("  3. LabeledPointsAnnotation- font_scale, font_thickness")
    print("  4. LabeledCircleAnnotation- font_scale, font_thickness")
    print("  5. LabeledLineAnnotation  - font_scale, font_thickness")
    print("  6. BoundingBoxAnnotation  - font_scale, font_thickness")
    print("  7. InfoPanel (via PanelStyle) - font_scale, font_thickness")
    print("\n💡 Typical font_scale values:")
    print("  • 0.3-0.4: Small/compact text")
    print("  • 0.5: Default size")
    print("  • 0.6-0.8: Larger, more readable")
    print("  • 1.0+: Large/emphasis text")
    print("\n💪 font_thickness: 1 (normal), 2+ (bold)")
else:
    print("Failed to render")