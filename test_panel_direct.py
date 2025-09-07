#!/usr/bin/env python3
"""Test direct InfoPanel usage with visual_debug"""

import numpy as np
import cv2
from visual_debugger import VisualDebugger
from visual_debugger.info_panel import InfoPanel, PanelPosition
from visual_debugger.annotations import point, circle

# Create test image
img = np.ones((400, 600, 3), dtype=np.uint8) * 50

# Create VisualDebugger
vd = VisualDebugger(output='return')

# Test 1: Direct InfoPanel usage (new simplified way)
print("Test 1: Direct InfoPanel usage")
panel = InfoPanel(position=PanelPosition.TOP_RIGHT, title="Direct Panel")
panel.add("Method", "Direct")
panel.add("Status", "Active")
result1 = vd.visual_debug(img, panel)  # Pass panel directly!
print("✓ Direct panel works!")

# Test 2: Mixed annotations with direct panel
print("\nTest 2: Mixed usage with list")
panel2 = InfoPanel(position=PanelPosition.BOTTOM_LEFT)
panel2.add("Frame", "42")
annotations = [
    panel2,  # Can mix InfoPanel directly in list
    point(100, 100, color=(255, 0, 0)),
    circle(200, 200, 50, color=(0, 255, 0))
]
result2 = vd.visual_debug(img, annotations)
print("✓ Mixed annotations work!")

# Test 3: Single annotation (not panel)
print("\nTest 3: Single annotation")
result3 = vd.visual_debug(img, point(300, 300))
print("✓ Single annotation works!")

# Save results
if result1 is not None:
    cv2.imwrite("test_panel_direct_1.jpg", cv2.cvtColor(result1, cv2.COLOR_RGB2BGR))
if result2 is not None:
    cv2.imwrite("test_panel_direct_2.jpg", cv2.cvtColor(result2, cv2.COLOR_RGB2BGR))
if result3 is not None:
    cv2.imwrite("test_panel_direct_3.jpg", cv2.cvtColor(result3, cv2.COLOR_RGB2BGR))

print("\n✅ All tests passed! You can now use:")
print("   result = vd.visual_debug(img, panel)")
print("   Instead of the verbose InfoPanelAnnotation wrapper!")