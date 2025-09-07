"""
Test 01: Type-Specific Annotations
===================================
Tests the new type-specific annotation classes to ensure they create correctly,
validate their parameters, and maintain type safety.

Why this is critical:
- These are the fundamental building blocks of the new annotation system
- Type safety prevents runtime errors and improves developer experience
- Proper validation ensures annotations are created with valid parameters

Test Cases:
1. test_point_annotations - Tests PointAnnotation and LabeledPointAnnotation creation
2. test_shape_annotations - Tests CircleAnnotation, RectangleAnnotation creation
3. test_line_annotations - Tests LineAnnotation, PolylineAnnotation creation
4. test_complex_annotations - Tests BoundingBoxAnnotation, MaskAnnotation
5. test_validation_errors - Tests that invalid parameters raise appropriate errors


python -m smoke_tests.test_01_type_specific_annotations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from visual_debugger.annotations import (
    PointAnnotation, LabeledPointAnnotation, PointsAnnotation, LabeledPointsAnnotation,
    CircleAnnotation, LabeledCircleAnnotation, RectangleAnnotation,
    LineAnnotation, LabeledLineAnnotation, PolylineAnnotation,
    TextAnnotation, BoundingBoxAnnotation, MaskAnnotation, OrientationAnnotation,
    # Factory functions
    point, labeled_point, circle, rectangle, line, text, bbox
)


def test_point_annotations():
    """Test Case 1: Point-based annotations"""
    print("\n" + "="*60)
    print("TEST 1: Point Annotations")
    print("="*60)
    
    try:
        # Test PointAnnotation
        p1 = PointAnnotation(position=(100, 200), color=(255, 0, 0), size=10)
        print(f"✓ PointAnnotation created: pos={p1.position}, color={p1.color}, size={p1.size}")
        
        # Test factory function
        p2 = point(150, 250, color=(0, 255, 0))
        print(f"✓ point() factory created: pos={p2.position}, color={p2.color}")
        
        # Test LabeledPointAnnotation
        lp1 = LabeledPointAnnotation(position=(50, 50), label="Target", color=(0, 0, 255))
        print(f"✓ LabeledPointAnnotation created: pos={lp1.position}, label='{lp1.label}'")
        
        # Test PointsAnnotation (multiple points)
        pts = PointsAnnotation(points=[(10, 10), (20, 20), (30, 30)], color=(255, 255, 0))
        print(f"✓ PointsAnnotation created with {len(pts.points)} points")
        
        # Test LabeledPointsAnnotation
        lpts = LabeledPointsAnnotation(
            points=[(100, 100), (200, 200)],
            labels=["P1", "P2"],
            color=(128, 128, 128)
        )
        print(f"✓ LabeledPointsAnnotation created with {len(lpts.points)} labeled points")
        
        print("\n✅ All point annotation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Point annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_shape_annotations():
    """Test Case 2: Shape-based annotations"""
    print("\n" + "="*60)
    print("TEST 2: Shape Annotations")
    print("="*60)
    
    try:
        # Test CircleAnnotation
        c1 = CircleAnnotation(center=(200, 200), radius=50, color=(255, 0, 0), thickness=3)
        print(f"✓ CircleAnnotation created: center={c1.center}, radius={c1.radius}, thickness={c1.thickness}")
        
        # Test filled circle
        c2 = CircleAnnotation(center=(300, 300), radius=30, filled=True)
        print(f"✓ Filled CircleAnnotation created: filled={c2.filled}")
        
        # Test LabeledCircleAnnotation
        lc = LabeledCircleAnnotation(center=(150, 150), radius=40, label="Zone A")
        print(f"✓ LabeledCircleAnnotation created: label='{lc.label}'")
        
        # Test RectangleAnnotation with corners
        r1 = RectangleAnnotation(top_left=(10, 10), bottom_right=(100, 50), color=(0, 255, 0))
        print(f"✓ RectangleAnnotation created: top_left={r1.top_left}, bottom_right={r1.bottom_right}")
        
        # Test RectangleAnnotation from xywh
        r2 = RectangleAnnotation.from_xywh(50, 50, 200, 100, thickness=2)
        print(f"✓ RectangleAnnotation from_xywh: width={r2.width}, height={r2.height}, center={r2.center}")
        
        # Test rectangle factory
        r3 = rectangle(20, 20, 80, 60, filled=True)
        print(f"✓ rectangle() factory created: filled={r3.filled}")
        
        print("\n✅ All shape annotation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Shape annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_line_annotations():
    """Test Case 3: Line-based annotations"""
    print("\n" + "="*60)
    print("TEST 3: Line Annotations")
    print("="*60)
    
    try:
        # Test LineAnnotation
        l1 = LineAnnotation(start=(0, 0), end=(100, 100), color=(255, 0, 0), thickness=2)
        print(f"✓ LineAnnotation created: start={l1.start}, end={l1.end}")
        
        # Test arrow line
        l2 = LineAnnotation(start=(50, 50), end=(150, 150), arrow=True)
        print(f"✓ Arrow LineAnnotation created: arrow={l2.arrow}")
        
        # Test LabeledLineAnnotation
        ll = LabeledLineAnnotation(
            start=(10, 10), end=(90, 90), 
            label="Distance", 
            color=(0, 255, 0)
        )
        print(f"✓ LabeledLineAnnotation created: label='{ll.label}', midpoint={ll.midpoint}")
        
        # Test PolylineAnnotation
        poly1 = PolylineAnnotation(
            points=[(10, 10), (50, 30), (80, 20), (100, 60)],
            thickness=3,
            closed=False
        )
        print(f"✓ PolylineAnnotation created with {len(poly1.points)} points, closed={poly1.closed}")
        
        # Test closed polyline (polygon)
        poly2 = PolylineAnnotation(
            points=[(100, 100), (200, 100), (200, 200), (100, 200)],
            closed=True,
            color=(0, 0, 255)
        )
        print(f"✓ Closed PolylineAnnotation (polygon) created: {len(poly2.points)} vertices")
        
        # Test line factory
        l3 = line(25, 25, 75, 75, thickness=4)
        print(f"✓ line() factory created: thickness={l3.thickness}")
        
        print("\n✅ All line annotation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Line annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complex_annotations():
    """Test Case 4: Complex annotations (BoundingBox, Mask, Orientation, Text)"""
    print("\n" + "="*60)
    print("TEST 4: Complex Annotations")
    print("="*60)
    
    try:
        # Test TextAnnotation
        t1 = TextAnnotation(text="Hello World", position=(50, 50), font_scale=1.0)
        print(f"✓ TextAnnotation created: text='{t1.text}', position={t1.position}")
        
        # Test text with background
        t2 = TextAnnotation(
            text="Important", 
            position=(100, 100),
            background_color=(0, 0, 0),
            background_padding=10
        )
        print(f"✓ TextAnnotation with background created: bg_color={t2.background_color}")
        
        # Test BoundingBoxAnnotation
        bb1 = BoundingBoxAnnotation(
            bbox=(50, 50, 100, 80),
            label="Person 95%",
            color=(255, 0, 0)
        )
        print(f"✓ BoundingBoxAnnotation created: label='{bb1.label}'")
        print(f"  Properties: top_left={bb1.top_left}, bottom_right={bb1.bottom_right}, center={bb1.center}")
        print(f"  Label text: '{bb1.get_label_text()}'")
        
        # Test bbox factory
        bb2 = bbox(10, 10, 50, 50, label="Car 87%")
        print(f"✓ bbox() factory created: label='{bb2.label}'")
        
        # Test MaskAnnotation
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[25:75, 25:75] = 1
        test_mask[40:60, 40:60] = 2
        
        m1 = MaskAnnotation(mask=test_mask, alpha=0.7, colormap='random')
        print(f"✓ MaskAnnotation created: shape={m1.mask.shape}, alpha={m1.alpha}, colormap='{m1.colormap}'")
        print(f"  Unique values in mask: {np.unique(m1.mask).tolist()}")
        
        # Test OrientationAnnotation
        o1 = OrientationAnnotation(
            position=(200, 200),
            pitch=30.0,
            yaw=45.0,
            roll=15.0,
            size=100
        )
        print(f"✓ OrientationAnnotation created: pitch={o1.pitch}°, yaw={o1.yaw}°, roll={o1.roll}°")
        
        print("\n✅ All complex annotation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Complex annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_errors():
    """Test Case 5: Validation and error handling"""
    print("\n" + "="*60)
    print("TEST 5: Validation and Error Handling")
    print("="*60)
    
    errors_caught = 0
    expected_errors = 4
    
    # Test 1: LabeledPointsAnnotation with mismatched points and labels
    try:
        bad_lpts = LabeledPointsAnnotation(
            points=[(10, 10), (20, 20)],
            labels=["A", "B", "C"]  # Too many labels!
        )
        print("❌ Should have raised ValueError for mismatched points/labels count")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for mismatched points/labels: {e}")
        errors_caught += 1
    
    # Test 2: PolylineAnnotation with too few points
    try:
        bad_poly = PolylineAnnotation(points=[(10, 10)])  # Only 1 point!
        print("❌ Should have raised ValueError for too few points")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for too few points: {e}")
        errors_caught += 1
    
    # Test 3: MaskAnnotation with wrong dimensions
    try:
        bad_mask = np.zeros((100, 100, 3))  # 3D array instead of 2D!
        m = MaskAnnotation(mask=bad_mask)
        print("❌ Should have raised ValueError for wrong mask dimensions")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for wrong mask dimensions: {e}")
        errors_caught += 1
    
    # Test 4: MaskAnnotation with invalid alpha
    try:
        good_mask = np.zeros((100, 100))
        m = MaskAnnotation(mask=good_mask, alpha=1.5)  # Alpha > 1!
        print("❌ Should have raised ValueError for invalid alpha")
    except ValueError as e:
        print(f"✓ Correctly caught ValueError for invalid alpha: {e}")
        errors_caught += 1
    
    if errors_caught == expected_errors:
        print(f"\n✅ All {expected_errors} validation errors correctly caught!")
        return True
    else:
        print(f"\n⚠️  Only {errors_caught}/{expected_errors} validation errors caught")
        return False


def main():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SMOKE TEST 01: TYPE-SPECIFIC ANNOTATIONS")
    print("="*80)
    print("\nThis test suite validates the new type-specific annotation classes,")
    print("ensuring they create correctly, maintain type safety, and validate parameters.")
    
    results = []
    
    # Run all tests
    results.append(("Point Annotations", test_point_annotations()))
    results.append(("Shape Annotations", test_shape_annotations()))
    results.append(("Line Annotations", test_line_annotations()))
    results.append(("Complex Annotations", test_complex_annotations()))
    results.append(("Validation Errors", test_validation_errors()))
    
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
        print("\n🎉 ALL TESTS PASSED! The type-specific annotation system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)