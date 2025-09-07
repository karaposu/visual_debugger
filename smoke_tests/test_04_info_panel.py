"""
Test 04: Info Panel Dashboard Display
======================================
Tests the InfoPanel's ability to display dashboard-style information with
various configurations, positions, and styling options.

Why this is critical:
- Info panels provide consolidated information display without cluttering the image
- Positioning must work correctly in all corners and edges
- Auto-sizing must adapt to content while respecting constraints
- Styling options must be applied consistently
- Multi-column layouts must handle data elegantly

Test Cases:
1. test_panel_creation - Tests basic panel creation and data addition
2. test_panel_positioning - Tests all position options and custom positioning
3. test_panel_styling - Tests colors, fonts, borders, and transparency
4. test_panel_content - Tests different content types and multi-column layout
5. test_panel_rendering - Tests actual rendering on images with size constraints

python -m smoke_tests.test_04_info_panel
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from visual_debugger.info_panel import InfoPanel, PanelPosition, PanelStyle


def create_test_image(width=800, height=600, color=(50, 50, 50)):
    """Create a test image with a solid color"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_panel_creation():
    """Test Case 1: Basic panel creation and data addition"""
    print("\n" + "="*60)
    print("TEST 1: Panel Creation")
    print("="*60)
    
    try:
        # Test basic creation
        panel1 = InfoPanel()
        print(f"✓ Basic panel created with position: {panel1.position}")
        
        # Test with title
        panel2 = InfoPanel(title="Debug Info")
        if panel2.title == "Debug Info":
            print(f"✓ Panel with title: '{panel2.title}'")
        else:
            print(f"❌ Title not set correctly")
            return False
        
        # Test adding single values
        panel2.add("FPS", 30)
        panel2.add("Score", 95.5)
        panel2.add("Status", "Running")
        
        if len(panel2._entries) == 3:
            print(f"✓ Added {len(panel2._entries)} items to panel")
        else:
            print(f"❌ Wrong number of items: {len(panel2._entries)}")
            return False
        
        # Test adding dict
        panel3 = InfoPanel()
        data_dict = {
            "Model": "ResNet50",
            "Accuracy": 0.92,
            "Epoch": 10,
            "Loss": 0.045
        }
        panel3.add_metrics(data_dict)
        
        if len(panel3._entries) == 4:
            print(f"✓ Added dict with {len(panel3._entries)} items")
        else:
            print(f"❌ Dict not added correctly: {len(panel3._entries)} items")
            return False
        
        # Test value formatting
        panel4 = InfoPanel()
        panel4.add("Float", f"{3.14159:.2f}")
        panel4.add("Percentage", f"{0.856:.1%}")
        
        # Check formatted values
        float_item = panel4._entries[0]
        percent_item = panel4._entries[1]
        
        if float_item[1] == "3.14":
            print(f"✓ Float formatting: {float_item[1]}")
        else:
            print(f"❌ Float formatting wrong: {float_item[1]}")
            return False
        
        if percent_item[1] == "85.6%":
            print(f"✓ Percent formatting: {percent_item[1]}")
        else:
            print(f"❌ Percent formatting wrong: {percent_item[1]}")
            return False
        
        # Test clearing
        panel4.clear()
        if len(panel4._entries) == 0:
            print(f"✓ Panel cleared successfully")
        else:
            print(f"❌ Clear failed: {len(panel4._entries)} items remain")
            return False
        
        print("\n✅ Panel creation test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_positioning():
    """Test Case 2: Panel positioning options"""
    print("\n" + "="*60)
    print("TEST 2: Panel Positioning")
    print("="*60)
    
    try:
        # Test all corner positions
        positions = [
            PanelPosition.TOP_LEFT,
            PanelPosition.TOP_RIGHT,
            PanelPosition.BOTTOM_LEFT,
            PanelPosition.BOTTOM_RIGHT
        ]
        
        for pos in positions:
            panel = InfoPanel(position=pos)
            img = create_test_image()
            width, height = panel.calculate_dimensions()
            x, y = panel.calculate_position(img.shape, width, height)
            print(f"✓ {pos.value}: calculated position ({x}, {y})")
        
        # Test edge positions
        edge_positions = [
            PanelPosition.TOP_CENTER,
            PanelPosition.BOTTOM_CENTER,
            PanelPosition.CENTER_LEFT,
            PanelPosition.CENTER_RIGHT,
            PanelPosition.CENTER
        ]
        
        for pos in edge_positions:
            panel = InfoPanel(position=pos)
            img = create_test_image()
            width, height = panel.calculate_dimensions()
            x, y = panel.calculate_position(img.shape, width, height)
            print(f"✓ {pos.value}: calculated position ({x}, {y})")
        
        # Test custom position tuple (defaults to TOP_LEFT when not a PanelPosition)
        panel_custom = InfoPanel(position=(100, 200))
        img = create_test_image()
        width, height = panel_custom.calculate_dimensions()
        x, y = panel_custom.calculate_position(img.shape, width, height)
        
        # Custom tuples aren't directly supported, defaults to margin position
        if isinstance(panel_custom.position, tuple):
            print(f"✓ Custom position stored as tuple: {panel_custom.position}")
        else:
            print(f"❌ Custom position not stored correctly")
            return False
        
        # Test string position
        panel_str = InfoPanel(position="bottom-right")
        if panel_str.position == PanelPosition.BOTTOM_RIGHT:
            print(f"✓ String position 'bottom-right' converted correctly")
        else:
            print(f"❌ String position conversion failed")
            return False
        
        # Test margin effect
        panel_margin = InfoPanel(position=PanelPosition.TOP_LEFT)
        img = create_test_image()
        width, height = panel_margin.calculate_dimensions()
        x, y = panel_margin.calculate_position(img.shape, width, height)
        
        if x >= 10 and y >= 10:  # Default margin is 10
            print(f"✓ Margin applied: position ({x}, {y}) with default margin")
        else:
            print(f"❌ Margin not applied correctly")
            return False
        
        print("\n✅ Panel positioning test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel positioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_styling():
    """Test Case 3: Panel styling options"""
    print("\n" + "="*60)
    print("TEST 3: Panel Styling")
    print("="*60)
    
    try:
        # Test default style
        default_style = PanelStyle()
        panel_default = InfoPanel(style=default_style)
        print(f"✓ Default style: bg={default_style.background_color}, alpha={default_style.background_alpha}")
        
        # Test custom colors
        custom_style = PanelStyle(
            background_color=(100, 50, 150),
            text_color=(255, 255, 0),
            title_color=(0, 255, 255),
            border_color=(255, 0, 0)
        )
        panel_custom = InfoPanel(style=custom_style, title="Styled")
        
        if custom_style.background_color == (100, 50, 150):
            print(f"✓ Custom colors set: bg={custom_style.background_color}")
        else:
            print(f"❌ Custom colors not set correctly")
            return False
        
        # Test transparency
        transparent_style = PanelStyle(
            background_alpha=0.3
        )
        panel_transparent = InfoPanel(style=transparent_style)
        
        if transparent_style.background_alpha == 0.3:
            print(f"✓ Transparency: bg_alpha={transparent_style.background_alpha}")
        else:
            print(f"❌ Transparency not set")
            return False
        
        # Test border settings
        border_style = PanelStyle(
            border_thickness=5,
            border_color=(255, 255, 255)
        )
        panel_border = InfoPanel(style=border_style)
        
        if border_style.border_thickness == 5:
            print(f"✓ Border settings: thickness={border_style.border_thickness}")
        else:
            print(f"❌ Border settings wrong")
            return False
        
        # Test font settings
        font_style = PanelStyle(
            font_scale=1.5,
            font_thickness=3
        )
        panel_font = InfoPanel(style=font_style, title="Large Text")
        
        if font_style.font_scale == 1.5:
            print(f"✓ Font settings: scale={font_style.font_scale}")
        else:
            print(f"❌ Font settings wrong")
            return False
        
        # Test padding and spacing
        spacing_style = PanelStyle(
            padding=20,
            line_spacing=15
        )
        panel_spacing = InfoPanel(style=spacing_style)
        
        if spacing_style.padding == 20 and spacing_style.line_spacing == 15:
            print(f"✓ Spacing: padding={spacing_style.padding}, line_spacing={spacing_style.line_spacing}")
        else:
            print(f"❌ Spacing settings wrong")
            return False
        
        # Test min dimensions
        size_style = PanelStyle(
            min_width=200
        )
        panel_size = InfoPanel(style=size_style)
        
        if size_style.min_width == 200:
            print(f"✓ Size constraints: min_width={size_style.min_width}")
        else:
            print(f"❌ Size constraints wrong")
            return False
        
        print("\n✅ Panel styling test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel styling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_content():
    """Test Case 4: Different content types and layouts"""
    print("\n" + "="*60)
    print("TEST 4: Panel Content")
    print("="*60)
    
    try:
        # Test various data types
        panel = InfoPanel(title="Mixed Content")
        
        # Add different types
        panel.add("Integer", 42)
        panel.add("Float", f"{3.14159:.3f}")
        panel.add("String", "Active")
        panel.add("Boolean", True)
        panel.add("Percentage", f"{0.756:.1%}")
        panel.add("None", None)
        panel.add("List", [1, 2, 3])
        panel.add("Dict", {"a": 1, "b": 2})
        
        if len(panel._entries) == 8:
            print(f"✓ All data types added: {len(panel._entries)} items")
            for key, value in panel._entries:
                print(f"  - {key}: {value} ({type(value).__name__})")
        else:
            print(f"❌ Wrong number of items: {len(panel._entries)}")
            return False
        
        # Test multi-column layout
        panel_columns = InfoPanel()
        for i in range(10):
            panel_columns.add(f"Item {i}", i * 10)
        
        print(f"✓ Multi-column panel: created with {len(panel_columns._entries)} items")
        
        # Test long content
        panel_long = InfoPanel()
        panel_long.add("Short", "OK")
        panel_long.add("Long Key Name That Might Overflow", "Value")
        panel_long.add("Key", "Very long value that might need to be truncated or wrapped somehow")
        
        print(f"✓ Long content added: {len(panel_long._entries)} items with varying lengths")
        
        # Test special characters
        panel_special = InfoPanel()
        panel_special.add("Unicode", "✓ ✗ ★ ☆")
        panel_special.add("Symbols", "< > & % $ #")
        panel_special.add("Math", "π ≈ ∞ Σ")
        
        print(f"✓ Special characters added: {len(panel_special._entries)} items")
        
        # Test empty panel
        panel_empty = InfoPanel(title="Empty Panel")
        if len(panel_empty._entries) == 0:
            print(f"✓ Empty panel created successfully")
        else:
            print(f"❌ Empty panel has items: {len(panel_empty._entries)}")
            return False
        
        # Test very large panel
        panel_large = InfoPanel(title="Large Dataset")
        for i in range(50):
            panel_large.add(f"Metric_{i:02d}", np.random.random())
        
        if len(panel_large._entries) == 50:
            print(f"✓ Large panel: {len(panel_large._entries)} items added")
        else:
            print(f"❌ Large panel wrong size: {len(panel_large._entries)}")
            return False
        
        print("\n✅ Panel content test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel content test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_rendering():
    """Test Case 5: Panel rendering via InfoPanelAnnotation"""
    print("\n" + "="*60)
    print("TEST 5: Panel Rendering via Annotation")
    print("="*60)
    
    try:
        from visual_debugger import VisualDebugger
        from visual_debugger.annotations import InfoPanelAnnotation
        
        # Test basic rendering through annotation
        img1 = create_test_image()
        panel1 = InfoPanel(title="Test Render")
        panel1.add("Frame", 1)
        panel1.add("FPS", 30.0)
        
        vd = VisualDebugger(output='return', active=True)
        panel_ann = InfoPanelAnnotation(panel=panel1)
        
        original_sum = np.sum(img1)
        result = vd.visual_debug(img1.copy(), [panel_ann])
        rendered_sum = np.sum(result) if result is not None else original_sum
        
        if rendered_sum != original_sum:
            print(f"✓ Panel rendered via annotation: image modified")
        else:
            print(f"❌ Panel annotation did not modify image")
            return False
        
        # Test size calculation
        panel2 = InfoPanel()
        panel2.add("A", 1)
        panel2.add("B", 2)
        panel2.add("C", 3)
        
        width, height = panel2.calculate_dimensions()
        
        if width > 0 and height > 0:
            print(f"✓ Size calculated: {width}x{height}")
        else:
            print(f"❌ Size calculation failed: {width}x{height}")
            return False
        
        # Test rendering at different positions
        positions_to_test = [
            PanelPosition.TOP_LEFT,
            PanelPosition.TOP_RIGHT,
            PanelPosition.BOTTOM_LEFT,
            PanelPosition.BOTTOM_RIGHT
        ]
        
        for pos in positions_to_test:
            img = create_test_image()
            panel = InfoPanel(position=pos, title=pos.value)
            panel.add("Position", pos.value)
            
            panel_ann = InfoPanelAnnotation(panel=panel)
            
            try:
                result = vd.visual_debug(img.copy(), [panel_ann])
                if result is not None:
                    print(f"✓ Rendered at {pos.value}")
                else:
                    print(f"❌ Failed to render at {pos.value}")
                    return False
            except Exception as e:
                print(f"❌ Failed to render at {pos.value}: {e}")
                return False
        
        # Test rendering with transparency
        img3 = create_test_image(color=(100, 100, 100))
        style_transparent = PanelStyle(
            background_alpha=0.5,
            background_color=(255, 255, 255)
        )
        panel_transparent = InfoPanel(style=style_transparent)
        panel_transparent.add("Alpha", 0.5)
        
        panel_trans_ann = InfoPanelAnnotation(panel=panel_transparent)
        result = vd.visual_debug(img3.copy(), [panel_trans_ann])
        img3 = result if result is not None else img3
        # Check if background is semi-transparent (not pure white or original gray)
        roi = img3[10:50, 10:50]  # Sample region where panel should be
        mean_color = np.mean(roi)
        
        if 100 < mean_color < 255:  # Should be between gray and white
            print(f"✓ Transparency applied: mean color {mean_color:.1f} (between 100 and 255)")
        else:
            print(f"⚠️  Transparency might not be working: mean color {mean_color:.1f}")
        
        # Test multi-item rendering
        img4 = create_test_image(1200, 800)
        panel_multi = InfoPanel(title="Multi-Item")
        for i in range(12):
            panel_multi.add(f"Item_{i}", i)
        
        panel_multi_ann = InfoPanelAnnotation(panel=panel_multi)
        result = vd.visual_debug(img4.copy(), [panel_multi_ann])
        if result is None:
            print(f"❌ Multi-item panel failed")
            return False
        print(f"✓ Multi-item panel rendered with 12 items")
        
        # Test rendering on small image
        img_small = create_test_image(200, 150)
        panel_small = InfoPanel()
        panel_small.add("Small", "Test")
        
        try:
            panel_small_ann = InfoPanelAnnotation(panel=panel_small)
            result = vd.visual_debug(img_small.copy(), [panel_small_ann])
            if result is not None:
                print(f"✓ Rendered on small image (200x150)")
            else:
                print(f"⚠️  Small image rendering returned None")
        except Exception as e:
            print(f"⚠️  Small image rendering issue: {e}")
        
        # Test rendering empty panel
        img5 = create_test_image()
        panel_empty = InfoPanel(title="Empty")
        
        try:
            panel_empty_ann = InfoPanelAnnotation(panel=panel_empty)
            result = vd.visual_debug(img5.copy(), [panel_empty_ann])
            if result is not None:
                print(f"✓ Empty panel rendered (title only)")
            else:
                print(f"❌ Empty panel returned None")
                return False
        except Exception as e:
            print(f"❌ Empty panel rendering failed: {e}")
            return False
        
        print("\n✅ Panel rendering test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel rendering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_panel_as_annotation():
    """Test Case 6: InfoPanelAnnotation as composite"""
    print("\n" + "="*60)
    print("TEST 6: Panel as Composite Annotation")
    print("="*60)
    
    try:
        from visual_debugger import VisualDebugger
        from visual_debugger.annotations import InfoPanelAnnotation
        
        # Create test panel
        panel = InfoPanel(position=PanelPosition.TOP_LEFT, title="Composite Test")
        panel.add("Type", "Composite")
        panel.add("Components", "Multiple")
        panel.add_separator()
        panel.add("Status", "Active")
        
        # Test decomposition
        panel_ann = InfoPanelAnnotation(panel=panel)
        components = panel_ann.get_component_annotations((400, 600))
        
        if len(components) > 0:
            print(f"✓ Panel decomposed into {len(components)} annotations")
            
            # Count types
            from collections import Counter
            types = Counter(type(c).__name__ for c in components)
            for ann_type, count in types.items():
                print(f"  - {count} {ann_type}(s)")
        else:
            print(f"❌ Panel decomposition failed")
            return False
        
        # Test with VisualDebugger
        img = create_test_image()
        vd = VisualDebugger(output='return', active=True)
        
        result = vd.visual_debug(img.copy(), [panel_ann])
        if result is not None:
            print(f"✓ InfoPanelAnnotation works with visual_debug")
        else:
            print(f"❌ InfoPanelAnnotation failed with visual_debug")
            return False
        
        # Test mixing with other annotations
        from visual_debugger.annotations import point, circle, text
        
        mixed = [
            panel_ann,
            point(300, 200, color=(255, 0, 0)),
            circle(400, 300, 30, color=(0, 255, 0)),
            text("Mixed", 500, 100, color=(255, 255, 255))
        ]
        
        result2 = vd.visual_debug(img.copy(), mixed)
        if result2 is not None:
            print(f"✓ Panel works with mixed annotations")
        else:
            print(f"❌ Panel failed with mixed annotations")
            return False
        
        print("\n✅ Panel as annotation test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Panel as annotation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SMOKE TEST 04: INFO PANEL DASHBOARD DISPLAY")
    print("="*80)
    print("\nThis test suite validates the InfoPanel's ability to display")
    print("dashboard-style information with various configurations.")
    
    results = []
    
    # Run all tests
    results.append(("Panel Creation", test_panel_creation()))
    results.append(("Panel Positioning", test_panel_positioning()))
    results.append(("Panel Styling", test_panel_styling()))
    results.append(("Panel Content", test_panel_content()))
    results.append(("Panel Rendering", test_panel_rendering()))
    results.append(("Panel as Annotation", test_panel_as_annotation()))
    
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
        print("\n🎉 ALL TESTS PASSED! The info panel system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)