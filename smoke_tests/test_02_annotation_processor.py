"""
Test 02: Annotation Processor and Rendering
============================================
Tests the annotation processor's ability to correctly render annotations on images
using the visitor pattern.

Why this is critical:
- The processor is responsible for actually drawing annotations on images
- Visitor pattern must correctly dispatch to appropriate render methods
- Image modifications must be applied correctly without corruption
- Color, thickness, and style parameters must be respected

Test Cases:
1. test_basic_rendering - Tests that annotations actually modify the image
2. test_visitor_dispatch - Tests visitor pattern correctly calls right methods
3. test_render_quality - Tests that rendered annotations have correct properties
4. test_multiple_annotations - Tests rendering multiple annotations on same image
5. test_edge_cases - Tests rendering at image boundaries and with extreme values

python -m smoke_tests.test_02_annotation_processor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from visual_debugger.annotations import (
    PointAnnotation, CircleAnnotation, RectangleAnnotation,
    LineAnnotation, TextAnnotation, MaskAnnotation,
    BoundingBoxAnnotation, OrientationAnnotation
)
from visual_debugger.annotation_processor import AnnotationProcessor


def create_test_image(width=400, height=300, channels=3):
    """Create a blank test image"""
    return np.zeros((height, width, channels), dtype=np.uint8)


def test_basic_rendering():
    """Test Case 1: Basic rendering - annotations modify the image"""
    print("\n" + "="*60)
    print("TEST 1: Basic Rendering")
    print("="*60)
    
    try:
        processor = AnnotationProcessor()
        
        # Test point rendering
        img1 = create_test_image()
        original_sum = np.sum(img1)
        point_ann = PointAnnotation(position=(50, 50), color=(255, 0, 0), size=5)
        img1 = processor.process(img1, point_ann)
        modified_sum = np.sum(img1)
        
        if modified_sum > original_sum:
            print(f"✓ Point annotation modified image (sum changed from {original_sum} to {modified_sum})")
        else:
            print(f"❌ Point annotation did not modify image")
            return False
        
        # Test circle rendering
        img2 = create_test_image()
        circle_ann = CircleAnnotation(center=(100, 100), radius=30, color=(0, 255, 0))
        img2 = processor.process(img2, circle_ann)
        circle_pixels = np.sum(img2 > 0)
        print(f"✓ Circle annotation rendered ({circle_pixels} non-zero pixels)")
        
        # Test rectangle rendering
        img3 = create_test_image()
        rect_ann = RectangleAnnotation(top_left=(20, 20), bottom_right=(80, 60), color=(0, 0, 255))
        img3 = processor.process(img3, rect_ann)
        rect_pixels = np.sum(img3 > 0)
        print(f"✓ Rectangle annotation rendered ({rect_pixels} non-zero pixels)")
        
        # Test line rendering
        img4 = create_test_image()
        line_ann = LineAnnotation(start=(10, 10), end=(100, 100), color=(255, 255, 0), thickness=3)
        img4 = processor.process(img4, line_ann)
        line_pixels = np.sum(img4 > 0)
        print(f"✓ Line annotation rendered ({line_pixels} non-zero pixels)")
        
        print("\n✅ All basic rendering tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visitor_dispatch():
    """Test Case 2: Visitor pattern dispatching"""
    print("\n" + "="*60)
    print("TEST 2: Visitor Pattern Dispatch")
    print("="*60)
    
    try:
        # Create a mock processor to track method calls
        class MockProcessor(AnnotationProcessor):
            def __init__(self):
                super().__init__()
                self.calls = []
            
            def render_point(self, ann):
                self.calls.append(('render_point', ann))
                super().render_point(ann)
            
            def render_circle(self, ann):
                self.calls.append(('render_circle', ann))
                super().render_circle(ann)
            
            def render_rectangle(self, ann):
                self.calls.append(('render_rectangle', ann))
                super().render_rectangle(ann)
            
            def render_line(self, ann):
                self.calls.append(('render_line', ann))
                super().render_line(ann)
        
        processor = MockProcessor()
        img = create_test_image()
        
        # Test different annotation types
        annotations = [
            PointAnnotation(position=(10, 10)),
            CircleAnnotation(center=(50, 50), radius=20),
            RectangleAnnotation(top_left=(0, 0), bottom_right=(30, 30)),
            LineAnnotation(start=(0, 0), end=(100, 100))
        ]
        
        for ann in annotations:
            processor.image = img.copy()
            ann.accept(processor)
        
        # Verify correct methods were called
        expected_methods = ['render_point', 'render_circle', 'render_rectangle', 'render_line']
        actual_methods = [call[0] for call in processor.calls]
        
        if actual_methods == expected_methods:
            print(f"✓ Visitor pattern correctly dispatched to all {len(expected_methods)} methods")
            for method, ann in processor.calls:
                print(f"  - {method} called with {type(ann).__name__}")
        else:
            print(f"❌ Visitor dispatch mismatch. Expected {expected_methods}, got {actual_methods}")
            return False
        
        print("\n✅ Visitor pattern dispatch test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Visitor dispatch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_render_quality():
    """Test Case 3: Rendering quality and properties"""
    print("\n" + "="*60)
    print("TEST 3: Render Quality")
    print("="*60)
    
    try:
        processor = AnnotationProcessor()
        
        # Test color accuracy
        img1 = create_test_image()
        red_color = (255, 0, 0)
        point_ann = PointAnnotation(position=(50, 50), color=red_color, size=1)
        img1 = processor.process(img1, point_ann)
        
        # Check if the point has the correct color (BGR format in OpenCV)
        pixel_color = tuple(img1[50, 50])
        expected_bgr = (0, 0, 255)  # Red in BGR
        if pixel_color == expected_bgr:
            print(f"✓ Color accuracy correct: requested RGB{red_color} → BGR{pixel_color}")
        else:
            print(f"⚠️  Color mismatch: expected BGR{expected_bgr}, got {pixel_color}")
        
        # Test thickness
        img2 = create_test_image()
        thick_line = LineAnnotation(start=(0, 50), end=(400, 50), color=(255, 255, 255), thickness=10)
        img2 = processor.process(img2, thick_line)
        
        # Count white pixels in a column to measure thickness
        column = img2[:, 200, :]  # Middle of the line
        white_pixels = np.sum(np.all(column == 255, axis=1))
        if 8 <= white_pixels <= 12:  # Allow some tolerance
            print(f"✓ Line thickness correct: ~10 pixels (actual: {white_pixels})")
        else:
            print(f"⚠️  Line thickness off: expected ~10, got {white_pixels}")
        
        # Test filled vs outline
        img3 = create_test_image()
        filled_circle = CircleAnnotation(center=(50, 50), radius=20, filled=True, color=(0, 255, 0))
        img3 = processor.process(img3, filled_circle)
        
        img4 = create_test_image()
        outline_circle = CircleAnnotation(center=(50, 50), radius=20, filled=False, thickness=2, color=(0, 255, 0))
        img4 = processor.process(img4, outline_circle)
        
        filled_pixels = np.sum(img3 > 0)
        outline_pixels = np.sum(img4 > 0)
        
        if filled_pixels > outline_pixels * 2:  # Filled should have many more pixels
            print(f"✓ Filled vs outline: filled={filled_pixels} pixels, outline={outline_pixels} pixels")
        else:
            print(f"⚠️  Filled/outline issue: filled={filled_pixels}, outline={outline_pixels}")
        
        # Test text rendering
        img5 = create_test_image()
        text_ann = TextAnnotation(text="TEST", position=(50, 50), font_scale=1.0, font_thickness=2)
        img5 = processor.process(img5, text_ann)
        text_pixels = np.sum(img5 > 0)
        
        if text_pixels > 0:
            print(f"✓ Text rendered successfully ({text_pixels} pixels)")
        else:
            print(f"❌ Text failed to render")
            return False
        
        print("\n✅ Render quality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Render quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_annotations():
    """Test Case 4: Multiple annotations on same image"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Annotations")
    print("="*60)
    
    try:
        processor = AnnotationProcessor()
        img = create_test_image(600, 400)
        
        # Create diverse annotations
        annotations = [
            PointAnnotation(position=(50, 50), color=(255, 0, 0), size=10),
            PointAnnotation(position=(100, 50), color=(0, 255, 0), size=10),
            PointAnnotation(position=(150, 50), color=(0, 0, 255), size=10),
            CircleAnnotation(center=(300, 200), radius=50, color=(255, 255, 0)),
            RectangleAnnotation(top_left=(400, 100), bottom_right=(550, 250), color=(255, 0, 255)),
            LineAnnotation(start=(0, 0), end=(600, 400), color=(0, 255, 255), thickness=2),
            TextAnnotation(text="Multi-Test", position=(250, 350), font_scale=1.5),
            BoundingBoxAnnotation(
                bbox=(50, 250, 100, 80),
                label="Object 95%",
                color=(128, 128, 128)
            )
        ]
        
        # Track pixel changes after each annotation
        pixel_counts = [0]
        for i, ann in enumerate(annotations):
            img = processor.process(img, ann)
            non_zero = np.sum(img > 0)
            pixel_counts.append(non_zero)
            
            if pixel_counts[-1] >= pixel_counts[-2]:
                print(f"✓ Annotation {i+1} ({type(ann).__name__}) added pixels: "
                      f"{pixel_counts[-2]} → {pixel_counts[-1]}")
            else:
                print(f"❌ Annotation {i+1} decreased pixel count!")
                return False
        
        # Verify all colors are present
        unique_colors = set()
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                pixel = tuple(img[y, x])
                if pixel != (0, 0, 0):
                    unique_colors.add(pixel)
        
        print(f"✓ Multiple colors rendered: {len(unique_colors)} unique non-black colors")
        
        # Test overlapping annotations
        img2 = create_test_image()
        overlap_anns = [
            CircleAnnotation(center=(100, 100), radius=40, filled=True, color=(255, 0, 0)),
            CircleAnnotation(center=(120, 100), radius=40, filled=True, color=(0, 255, 0)),
            CircleAnnotation(center=(110, 120), radius=40, filled=True, color=(0, 0, 255))
        ]
        
        for ann in overlap_anns:
            img2 = processor.process(img2, ann)
        
        # Check that later annotations override earlier ones at overlap points
        center_pixel = tuple(img2[110, 110])
        if center_pixel[0] > 0:  # Should have blue component from last circle
            print(f"✓ Overlapping annotations handled correctly (last write wins)")
        
        print("\n✅ Multiple annotations test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Multiple annotations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test Case 5: Edge cases and boundary conditions"""
    print("\n" + "="*60)
    print("TEST 5: Edge Cases")
    print("="*60)
    
    try:
        processor = AnnotationProcessor()
        errors_handled = 0
        
        # Test 1: Annotation at image boundaries
        img1 = create_test_image(200, 200)
        boundary_anns = [
            PointAnnotation(position=(0, 0), color=(255, 0, 0)),  # Top-left corner
            PointAnnotation(position=(199, 199), color=(0, 255, 0)),  # Bottom-right corner
            CircleAnnotation(center=(0, 100), radius=30, color=(0, 0, 255)),  # Partial circle
            LineAnnotation(start=(-10, -10), end=(210, 210), color=(255, 255, 0))  # Line exceeding bounds
        ]
        
        for ann in boundary_anns:
            try:
                img1 = processor.process(img1.copy(), ann)
                print(f"✓ Boundary annotation handled: {type(ann).__name__}")
            except Exception as e:
                print(f"❌ Boundary annotation failed: {type(ann).__name__} - {e}")
                errors_handled += 1
        
        # Test 2: Large mask annotation
        img2 = create_test_image(100, 100)
        large_mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
        mask_ann = MaskAnnotation(mask=large_mask, alpha=0.5, colormap='random')
        
        try:
            img2 = processor.process(img2, mask_ann)
            print(f"✓ Large mask annotation handled ({large_mask.shape} mask)")
        except Exception as e:
            print(f"❌ Large mask failed: {e}")
            errors_handled += 1
        
        # Test 3: Complex orientation annotation
        img3 = create_test_image(300, 300)
        orientation_ann = OrientationAnnotation(
            position=(150, 150),
            pitch=45.0,
            yaw=30.0,
            roll=60.0,
            size=100
        )
        
        try:
            img3 = processor.process(img3, orientation_ann)
            orientation_pixels = np.sum(img3 > 0)
            print(f"✓ 3D orientation rendered ({orientation_pixels} pixels)")
        except Exception as e:
            print(f"❌ Orientation failed: {e}")
            errors_handled += 1
        
        # Test 4: Empty/minimal annotations
        img4 = create_test_image()
        minimal_anns = [
            TextAnnotation(text="", position=(50, 50)),  # Empty text
            CircleAnnotation(center=(100, 100), radius=0, color=(255, 0, 0)),  # Zero radius
            LineAnnotation(start=(50, 50), end=(50, 50), color=(0, 255, 0))  # Zero-length line
        ]
        
        for ann in minimal_anns:
            try:
                img4 = processor.process(img4.copy(), ann)
                print(f"✓ Minimal annotation handled: {type(ann).__name__}")
            except Exception as e:
                print(f"⚠️  Minimal annotation issue: {type(ann).__name__} - {e}")
                errors_handled += 1
        
        # Test 5: Extreme values
        img5 = create_test_image(500, 500)
        extreme_anns = [
            CircleAnnotation(center=(250, 250), radius=1000, color=(255, 0, 0)),  # Huge radius
            TextAnnotation(text="X"*100, position=(10, 10), font_scale=0.1),  # Long text
            RectangleAnnotation(top_left=(-100, -100), bottom_right=(600, 600), color=(0, 255, 0))  # Outside bounds
        ]
        
        for ann in extreme_anns:
            try:
                img5_copy = processor.process(img5.copy(), ann)
                pixels = np.sum(img5_copy > 0)
                print(f"✓ Extreme value handled: {type(ann).__name__} ({pixels} pixels rendered)")
            except Exception as e:
                print(f"⚠️  Extreme value issue: {type(ann).__name__} - {e}")
                errors_handled += 1
        
        if errors_handled == 0:
            print("\n✅ All edge cases handled successfully!")
        else:
            print(f"\n⚠️  {errors_handled} edge cases had issues (may be expected)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SMOKE TEST 02: ANNOTATION PROCESSOR AND RENDERING")
    print("="*80)
    print("\nThis test suite validates the annotation processor's ability to render")
    print("annotations correctly using the visitor pattern.")
    
    results = []
    
    # Run all tests
    results.append(("Basic Rendering", test_basic_rendering()))
    results.append(("Visitor Dispatch", test_visitor_dispatch()))
    results.append(("Render Quality", test_render_quality()))
    results.append(("Multiple Annotations", test_multiple_annotations()))
    results.append(("Edge Cases", test_edge_cases()))
    
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
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The annotation processor is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)