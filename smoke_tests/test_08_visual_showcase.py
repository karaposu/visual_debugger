"""
Test 08: Visual Showcase with Sample Image
===========================================
Comprehensive visual test that demonstrates all annotation types, info panel,
and image composition features using a sample image.

Why this is critical:
- Provides visual verification of all features working together
- Creates actual output files for manual inspection
- Tests real-world usage patterns
- Demonstrates the full capabilities of the system
- Serves as a visual reference for all annotation types

Test Cases:
1. test_all_point_annotations - Tests all point-based annotations
2. test_all_shape_annotations - Tests all shape-based annotations  
3. test_all_line_annotations - Tests all line and polyline annotations
4. test_complex_annotations - Tests masks, orientation, bounding boxes
5. test_info_panel_showcase - Tests info panel with various configurations
6. test_composition_showcase - Tests image composition and grid layouts

python -m smoke_tests.test_08_visual_showcase
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path

from visual_debugger.visual_debugger import VisualDebugger
from visual_debugger.annotations import (
    # Point annotations
    PointAnnotation, LabeledPointAnnotation, PointsAnnotation, LabeledPointsAnnotation,
    # Shape annotations
    CircleAnnotation, LabeledCircleAnnotation, RectangleAnnotation,
    # Line annotations
    LineAnnotation, LabeledLineAnnotation, PolylineAnnotation,
    # Text and complex annotations
    TextAnnotation, BoundingBoxAnnotation, MaskAnnotation, OrientationAnnotation,
    # InfoPanelAnnotation not needed with new direct API
    # Factory functions
    point, labeled_point, circle, rectangle, line, text, bbox
)
from visual_debugger.info_panel import InfoPanel, PanelPosition, PanelStyle
from visual_debugger.composition import ImageCompositor, CompositionStyle, ImageEntry, LayoutDirection


def load_sample_image():
    """Load the sample_image.jpg file"""
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_image.jpg")
    if not os.path.exists(sample_path):
        print(f"Warning: sample_image.jpg not found at {sample_path}")
        print("Creating synthetic image instead...")
        # Fallback to synthetic image
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        for y in range(600):
            for x in range(800):
                img[y, x] = [
                    int(255 * (1 - y/600)),
                    int(255 * (x/800)),
                    int(128 + 127 * np.sin(np.pi * x/800) * np.cos(np.pi * y/600))
                ]
        return img
    
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✓ Loaded sample_image.jpg ({img.shape[1]}x{img.shape[0]})")
    return img


def save_or_check_image(img, filename, output_dir):
    """Save image and verify it was created"""
    filepath = os.path.join(output_dir, filename)
    success = cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if success and os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ Saved {filename} ({size:,} bytes)")
        return True
    else:
        print(f"❌ Failed to save {filename}")
        return False


def test_all_point_annotations(output_dir):
    """Test Case 1: All point-based annotations"""
    print("\n" + "="*60)
    print("TEST 1: Point Annotations Showcase")
    print("="*60)
    
    try:
        img = load_sample_image()
        h, w = img.shape[:2]  # Get actual image dimensions (364x386)
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        
        # Single point
        ann1 = PointAnnotation(position=(100, 150), color=(255, 0, 0), size=10)
        img1 = vd.visual_debug(img.copy(), [ann1], name="point_single")
        
        # Labeled point
        ann2 = LabeledPointAnnotation(
            position=(200, 150), 
            label="Target", 
            color=(0, 255, 0), 
            size=8,
            font_scale=0.8
        )
        img2 = vd.visual_debug(img.copy(), [ann2], name="point_labeled")
        
        # Multiple points (ensure all within bounds)
        ann3 = PointsAnnotation(
            points=[(100, 200), (150, 200), (200, 200), (250, 200)],
            color=(0, 0, 255),
            size=6
        )
        img3 = vd.visual_debug(img.copy(), [ann3], name="points_multiple")
        
        # Labeled points (within bounds)
        ann4 = LabeledPointsAnnotation(
            points=[(50, 250), (120, 250), (190, 250), (260, 250)],
            labels=["P1", "P2", "P3", "P4"],
            color=(255, 255, 0),
            size=7,
            font_scale=0.6
        )
        img4 = vd.visual_debug(img.copy(), [ann4], name="points_labeled_multiple")
        
        # Combined showcase (all within image bounds)
        all_points = [
            point(100, 300, color=(255, 0, 0), size=5),
            point(150, 300, color=(0, 255, 0), size=10),
            point(200, 300, color=(0, 0, 255), size=15),
            labeled_point(250, 300, "Small", color=(255, 0, 255), size=3),
            labeled_point(320, 300, "Large", color=(0, 255, 255), size=20),
        ]
        img5 = vd.visual_debug(img.copy(), all_points, name="points_combined")
        
        # Save all results
        results = [
            save_or_check_image(img1, "01_point_single.jpg", output_dir),
            save_or_check_image(img2, "01_point_labeled.jpg", output_dir),
            save_or_check_image(img3, "01_points_multiple.jpg", output_dir),
            save_or_check_image(img4, "01_points_labeled_multiple.jpg", output_dir),
            save_or_check_image(img5, "01_points_combined.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All point annotation tests passed!")
            return True
        else:
            print("\n❌ Some point annotation tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Point annotations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_shape_annotations(output_dir):
    """Test Case 2: All shape-based annotations"""
    print("\n" + "="*60)
    print("TEST 2: Shape Annotations Showcase")
    print("="*60)
    
    try:
        img = load_sample_image()
        h, w = img.shape[:2]  # Get actual image dimensions (364x386)
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        
        # Circles (ensure all within bounds)
        circle_anns = [
            CircleAnnotation(center=(80, 80), radius=25, color=(255, 0, 0), thickness=2),
            CircleAnnotation(center=(190, 80), radius=25, color=(0, 255, 0), filled=True),
            LabeledCircleAnnotation(center=(300, 80), radius=25, label="Zone A", color=(0, 0, 255)),
        ]
        img1 = vd.visual_debug(img.copy(), circle_anns, name="circles")
        
        # Rectangles (ensure all within image bounds)
        rect_anns = [
            RectangleAnnotation(top_left=(30, 200), bottom_right=(120, 290), color=(255, 0, 0), thickness=2),
            RectangleAnnotation.from_xywh(150, 200, 80, 90, color=(0, 255, 0), filled=True),
            rectangle(250, 200, 80, 90, color=(0, 0, 255), thickness=3),
        ]
        img2 = vd.visual_debug(img.copy(), rect_anns, name="rectangles")
        
        # Mixed shapes (centered within image bounds)
        center_x, center_y = w//2, h//2
        mixed_shapes = [
            circle(center_x, center_y, 30, color=(255, 0, 255), thickness=3),
            circle(center_x, center_y, 50, color=(255, 255, 0), thickness=2),
            circle(center_x, center_y, 70, color=(0, 255, 255), thickness=1),
            rectangle(center_x-80, center_y-80, 160, 160, color=(128, 128, 128), thickness=1),
        ]
        img3 = vd.visual_debug(img.copy(), mixed_shapes, name="shapes_concentric")
        
        # Shape patterns (ensure all within bounds)
        pattern_anns = []
        for i in range(4):  # Reduced to 4 to fit within width
            x = 60 + i * 80
            y = h - 80  # Near bottom but within bounds
            pattern_anns.append(circle(x, y, 15 + i*3, color=(255-i*60, i*60, 128), filled=True))
            pattern_anns.append(rectangle(x-25, y-25, 50, 50, color=(i*60, 255-i*60, 128), thickness=2))
        img4 = vd.visual_debug(img.copy(), pattern_anns, name="shapes_pattern")
        
        # Save all results
        results = [
            save_or_check_image(img1, "02_circles.jpg", output_dir),
            save_or_check_image(img2, "02_rectangles.jpg", output_dir),
            save_or_check_image(img3, "02_shapes_concentric.jpg", output_dir),
            save_or_check_image(img4, "02_shapes_pattern.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All shape annotation tests passed!")
            return True
        else:
            print("\n❌ Some shape annotation tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Shape annotations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_line_annotations(output_dir):
    """Test Case 3: All line and polyline annotations"""
    print("\n" + "="*60)
    print("TEST 3: Line Annotations Showcase")
    print("="*60)
    
    try:
        img = load_sample_image()
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        
        # Basic lines
        line_anns = [
            LineAnnotation(start=(50, 50), end=(750, 50), color=(255, 0, 0), thickness=2),
            LineAnnotation(start=(50, 100), end=(750, 100), color=(0, 255, 0), thickness=3, arrow=True),
            LabeledLineAnnotation(start=(50, 150), end=(750, 150), label="Distance: 700px", color=(0, 0, 255)),
        ]
        img1 = vd.visual_debug(img.copy(), line_anns, name="lines_basic")
        
        # Arrow patterns
        arrow_anns = []
        center_x, center_y = 400, 300
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            end_x = int(center_x + 100 * np.cos(rad))
            end_y = int(center_y + 100 * np.sin(rad))
            color = (int(255 * (angle/360)), int(255 * (1-angle/360)), 128)
            arrow_anns.append(LineAnnotation(start=(center_x, center_y), end=(end_x, end_y), 
                                            color=color, thickness=2, arrow=True))
        img2 = vd.visual_debug(img.copy(), arrow_anns, name="lines_arrows")
        
        # Polylines
        poly_anns = [
            # Open polyline (zigzag)
            PolylineAnnotation(
                points=[(100, 400), (150, 350), (200, 400), (250, 350), (300, 400)],
                color=(255, 0, 0),
                thickness=2,
                closed=False
            ),
            # Closed polyline (polygon)
            PolylineAnnotation(
                points=[(500, 350), (600, 350), (650, 400), (600, 450), (500, 450), (450, 400)],
                color=(0, 255, 0),
                thickness=3,
                closed=True
            ),
        ]
        img3 = vd.visual_debug(img.copy(), poly_anns, name="polylines")
        
        # Complex line art
        art_anns = []
        # Create a star pattern
        n_points = 10
        for i in range(n_points):
            angle1 = 2 * np.pi * i / n_points
            angle2 = 2 * np.pi * ((i + 3) % n_points) / n_points
            x1, y1 = 400 + 150*np.cos(angle1), 300 + 150*np.sin(angle1)
            x2, y2 = 400 + 150*np.cos(angle2), 300 + 150*np.sin(angle2)
            art_anns.append(line(int(x1), int(y1), int(x2), int(y2), 
                                color=(128, 128, 255), thickness=1))
        img4 = vd.visual_debug(img.copy(), art_anns, name="lines_art")
        
        # Save all results
        results = [
            save_or_check_image(img1, "03_lines_basic.jpg", output_dir),
            save_or_check_image(img2, "03_lines_arrows.jpg", output_dir),
            save_or_check_image(img3, "03_polylines.jpg", output_dir),
            save_or_check_image(img4, "03_lines_art.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All line annotation tests passed!")
            return True
        else:
            print("\n❌ Some line annotation tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Line annotations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complex_annotations(output_dir):
    """Test Case 4: Complex annotations (text, bbox, mask, orientation)"""
    print("\n" + "="*60)
    print("TEST 4: Complex Annotations Showcase")
    print("="*60)
    
    try:
        img = load_sample_image()
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        
        # Text annotations
        text_anns = [
            TextAnnotation(text="Simple Text", position=(50, 50), font_scale=1.0, color=(255, 255, 255)),
            TextAnnotation(text="With Background", position=(50, 100), font_scale=0.8, 
                          color=(255, 255, 0), background_color=(0, 0, 0), background_padding=10),
            text("Large Text", 50, 150, font_scale=1.5, color=(0, 255, 255)),
        ]
        img1 = vd.visual_debug(img.copy(), text_anns, name="text")
        
        # Bounding boxes
        bbox_anns = [
            BoundingBoxAnnotation(bbox=(100, 200, 150, 100), label="Person 95%", color=(255, 0, 0)),
            BoundingBoxAnnotation(bbox=(300, 200, 150, 100), label="Car 87%", color=(0, 255, 0)),
            bbox(500, 200, 150, 100, label="Dog 92%", color=(0, 0, 255)),
        ]
        img2 = vd.visual_debug(img.copy(), bbox_anns, name="bounding_boxes")
        
        # Segmentation mask (match actual image dimensions)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        # Create regions proportional to image size
        mask[h//6:h//3, w//8:w//3] = 1  # Region 1
        mask[h//3:h//2, w//3:w//2] = 2  # Region 2
        mask[h//2:2*h//3, w//2:3*w//4] = 3  # Region 3
        # Create circular region
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w//2, 2*h//3
        radius = min(h, w) // 8
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 4
        
        mask_ann = MaskAnnotation(mask=mask, alpha=0.5, colormap='random')
        img3 = vd.visual_debug(img.copy(), [mask_ann], name="mask")
        
        # 3D Orientation
        orientation_anns = [
            OrientationAnnotation(position=(200, 400), pitch=30, yaw=45, roll=15, size=100,
                                 x_color=(255, 0, 0), y_color=(0, 255, 0), z_color=(0, 0, 255)),
            OrientationAnnotation(position=(600, 400), pitch=-30, yaw=90, roll=45, size=80),
        ]
        img4 = vd.visual_debug(img.copy(), orientation_anns, name="orientation")
        
        # Combined complex
        complex_anns = [
            bbox(100, 100, 200, 150, label="Detection 98%", color=(255, 255, 0)),
            text("Annotated Region", 110, 90, font_scale=0.7, color=(255, 255, 255)),
            OrientationAnnotation(position=(200, 175), pitch=0, yaw=45, roll=0, size=50),
            circle(500, 300, 100, color=(0, 255, 255), thickness=3),
            text("Multiple\nLayers", 450, 280, font_scale=0.8, color=(255, 255, 255)),
        ]
        img5 = vd.visual_debug(img.copy(), complex_anns, name="complex_combined")
        
        # Save all results
        results = [
            save_or_check_image(img1, "04_text.jpg", output_dir),
            save_or_check_image(img2, "04_bounding_boxes.jpg", output_dir),
            save_or_check_image(img3, "04_mask.jpg", output_dir),
            save_or_check_image(img4, "04_orientation.jpg", output_dir),
            save_or_check_image(img5, "04_complex_combined.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All complex annotation tests passed!")
            return True
        else:
            print("\n❌ Some complex annotation tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Complex annotations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_info_panel_showcase(output_dir):
    """Test Case 5: Info panel with various configurations"""
    print("\n" + "="*60)
    print("TEST 5: Info Panel Showcase")
    print("="*60)
    
    try:
        img = load_sample_image()
        
        # Panel 1: Top-left with metrics
        panel1 = InfoPanel(position=PanelPosition.TOP_LEFT, title="System Metrics")
        panel1.add("FPS", "30.0")
        panel1.add("CPU", "45%")
        panel1.add("Memory", "2.3 GB")
        panel1.add("GPU", "87%")
        panel1.add_separator()
        panel1.add("Status", "Running")
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        # Direct panel passing (new simplified API)
        img1 = vd.visual_debug(img.copy(), panel1, name="panel_metrics")
        
        # Panel 2: Top-right with custom style
        style2 = PanelStyle(
            background_color=(50, 50, 150),
            background_alpha=0.8,
            text_color=(255, 255, 255),
            title_color=(255, 255, 0),
            border_color=(255, 255, 255),
            border_thickness=3
        )
        panel2 = InfoPanel(position=PanelPosition.TOP_RIGHT, title="Detection Results", style=style2)
        panel2.add("Objects", "15")
        panel2.add("Confidence", "0.94")
        panel2.add("Time", "23.5ms")
        # Direct panel passing
        img2 = vd.visual_debug(img.copy(), panel2, name="panel_styled")
        
        # Panel 3: Bottom with progress bars
        panel3 = InfoPanel(position=PanelPosition.BOTTOM_CENTER, title="Progress")
        panel3.add_progress("Training", 0.75, width=20)
        panel3.add_progress("Validation", 0.45, width=20)
        panel3.add_progress("Testing", 0.90, width=20)
        # Direct panel passing
        img3 = vd.visual_debug(img.copy(), panel3, name="panel_progress")
        
        # Panel 4: Multiple panels on same image
        panel4a = InfoPanel(position=PanelPosition.TOP_LEFT, title="Camera")
        panel4a.add("Resolution", "1920x1080")
        panel4a.add("Format", "RGB")
        
        panel4b = InfoPanel(position=PanelPosition.TOP_RIGHT, title="Processing")
        panel4b.add("Filter", "Gaussian")
        panel4b.add("Kernel", "5x5")
        
        panel4c = InfoPanel(position=PanelPosition.BOTTOM_LEFT, title="Output")
        panel4c.add("Format", "JPEG")
        panel4c.add("Quality", "95")
        
        panel4d = InfoPanel(position=PanelPosition.BOTTOM_RIGHT, title="Performance")
        panel4d.add("Latency", "12ms")
        panel4d.add("Throughput", "83 fps")
        
        # Direct panels in list (new simplified API)
        panel4_list = [
            panel4a,  # Direct panels
            panel4b,
            panel4c,
            panel4d
        ]
        img4 = vd.visual_debug(img.copy(), panel4_list, name="panel_multiple")
        
        # Panel 5: Center panel with table
        panel5 = InfoPanel(position=PanelPosition.CENTER, title="Model Comparison")
        panel5.add_table(
            headers=["Model", "Acc", "Speed"],
            rows=[
                ["ResNet", "94.2", "23ms"],
                ["VGG", "92.1", "45ms"],
                ["YOLO", "89.5", "12ms"],
            ]
        )
        # Direct panel passing
        img5 = vd.visual_debug(img.copy(), panel5, name="panel_table")
        
        # Panel 6: Compact panel without title (saves space)
        panel6 = InfoPanel(position=PanelPosition.TOP_LEFT)  # No title for compact display
        panel6.add("Frame", "42")
        panel6.add("Timestamp", "00:01:23")
        # Direct panel passing
        img6 = vd.visual_debug(img.copy(), panel6, name="panel_annotation")
        
        # Save all results
        results = [
            save_or_check_image(img1, "05_panel_metrics.jpg", output_dir),
            save_or_check_image(img2, "05_panel_styled.jpg", output_dir),
            save_or_check_image(img3, "05_panel_progress.jpg", output_dir),
            save_or_check_image(img4, "05_panel_multiple.jpg", output_dir),
            save_or_check_image(img5, "05_panel_table.jpg", output_dir),
            save_or_check_image(img6, "05_panel_annotation.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All info panel tests passed!")
            return True
        else:
            print("\n❌ Some info panel tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Info panel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_showcase(output_dir):
    """Test Case 6: Image composition and grid layouts"""
    print("\n" + "="*60)
    print("TEST 6: Composition Showcase")
    print("="*60)
    
    try:
        # Create different annotated versions
        img_base = load_sample_image()
        # Resize for composition demo if needed
        if img_base.shape[0] > 600 or img_base.shape[1] > 800:
            img_base = cv2.resize(img_base, (400, 300))
        vd = VisualDebugger(debug_folder_path=output_dir, output='return')
        
        # Create various annotated versions
        img1 = vd.visual_debug(img_base.copy(), [
            circle(200, 150, 50, color=(255, 0, 0), thickness=3),
            text("Original", 150, 50, font_scale=1.0, color=(255, 255, 255))
        ], name="comp1")
        
        img2 = vd.visual_debug(img_base.copy(), [
            rectangle(100, 75, 200, 150, color=(0, 255, 0), thickness=3),
            text("Detected", 150, 50, font_scale=1.0, color=(255, 255, 255))
        ], name="comp2")
        
        img3 = vd.visual_debug(img_base.copy(), [
            line(50, 150, 350, 150, color=(0, 0, 255), thickness=2, arrow=True),
            text("Processed", 150, 50, font_scale=1.0, color=(255, 255, 255))
        ], name="comp3")
        
        img4 = vd.visual_debug(img_base.copy(), [
            bbox(75, 75, 250, 150, label="Object 95%", color=(255, 255, 0)),
            text("Final", 150, 50, font_scale=1.0, color=(255, 255, 255))
        ], name="comp4")
        
        # Test 1: Horizontal concatenation
        compositor = ImageCompositor()
        entries_h = [
            ImageEntry(img1, "Step 1", "Original"),
            ImageEntry(img2, "Step 2", "Detection"),
            ImageEntry(img3, "Step 3", "Processing"),
            ImageEntry(img4, "Step 4", "Final"),
        ]
        result_h = compositor.concatenate(entries_h, direction=LayoutDirection.HORIZONTAL)
        
        # Test 2: Vertical concatenation
        entries_v = [
            ImageEntry(img1, "Original"),
            ImageEntry(img2, "After Detection"),
            ImageEntry(img3, "After Processing"),
        ]
        result_v = compositor.concatenate(entries_v, direction=LayoutDirection.VERTICAL)
        
        # Test 3: Grid layout
        images_grid = [img1, img2, img3, img4, img1, img2]
        labels_grid = ["Image 1", "Image 2", "Image 3", "Image 4", "Image 5", "Image 6"]
        result_grid = compositor.create_grid(images_grid, cols=3, labels=labels_grid)
        
        # Test 4: Styled composition
        style = CompositionStyle(
            border_thickness=5,
            border_color=(255, 255, 255),
            vertical_spacing=30,
            horizontal_spacing=30,
            show_labels=True,
            show_borders=True,
            show_separators=True,
            separator_color=(128, 128, 128),
            background_color=(50, 50, 50)
        )
        compositor_styled = ImageCompositor(style)
        result_styled = compositor_styled.concatenate(entries_h, direction=LayoutDirection.HORIZONTAL)
        
        # Test 5: Comparison view
        result_compare = compositor.create_comparison(
            before=img1,
            after=img4,
            before_label="Before Processing",
            after_label="After Processing"
        )
        
        # Save all results
        results = [
            save_or_check_image(result_h, "06_composition_horizontal.jpg", output_dir),
            save_or_check_image(result_v, "06_composition_vertical.jpg", output_dir),
            save_or_check_image(result_grid, "06_composition_grid.jpg", output_dir),
            save_or_check_image(result_styled, "06_composition_styled.jpg", output_dir),
            save_or_check_image(result_compare, "06_composition_compare.jpg", output_dir),
        ]
        
        if all(results):
            print("\n✅ All composition tests passed!")
            return True
        else:
            print("\n❌ Some composition tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Composition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all visual showcase tests"""
    print("\n" + "="*80)
    print("SMOKE TEST 08: VISUAL SHOWCASE")
    print("="*80)
    print("\nThis test creates visual outputs demonstrating all features")
    print("of the Visual Debugger system using sample images.")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "test_08_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print("(Files will be preserved for inspection)")
    
    results = []
    
    try:
        # Run all tests
        results.append(("Point Annotations", test_all_point_annotations(output_dir)))
        results.append(("Shape Annotations", test_all_shape_annotations(output_dir)))
        results.append(("Line Annotations", test_all_line_annotations(output_dir)))
        results.append(("Complex Annotations", test_complex_annotations(output_dir)))
        results.append(("Info Panel", test_info_panel_showcase(output_dir)))
        results.append(("Composition", test_composition_showcase(output_dir)))
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{name:.<50} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        # List output files
        print("\n" + "="*80)
        print("OUTPUT FILES")
        print("="*80)
        print(f"Location: {output_dir}")
        
        files = sorted(Path(output_dir).glob("*.jpg"))
        if files:
            print(f"\nGenerated {len(files)} image files:")
            for f in files:
                size = f.stat().st_size
                print(f"  • {f.name:<40} {size:>10,} bytes")
        else:
            print("\n⚠️  No output files were generated")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Visual showcase completed successfully.")
            print(f"\n📁 View the results in: {output_dir}")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
        
        return passed == total
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)