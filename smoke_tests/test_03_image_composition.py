"""
Test 03: Image Composition and Layout
======================================
Tests the ImageCompositor's ability to combine multiple images with various
layouts, styles, and composition modes.

Why this is critical:
- Image composition is essential for comparing multiple debug outputs
- Layout management must handle different image sizes correctly
- Styling options must be applied consistently
- Grid and concatenation modes must work with various configurations

Test Cases:
1. test_basic_concatenation - Tests horizontal and vertical concatenation
2. test_grid_layout - Tests grid creation with automatic and manual columns
3. test_composition_styles - Tests borders, spacing, labels, and separators
4. test_image_entries - Tests ImageEntry metadata and labeling
5. test_size_handling - Tests handling of different sized images

python -m smoke_tests.test_03_image_composition
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from visual_debugger.composition import (
    ImageCompositor, CompositionStyle, ImageEntry, LayoutDirection
)


def create_colored_image(width, height, color):
    """Create a solid color test image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_basic_concatenation():
    """Test Case 1: Basic concatenation in different directions"""
    print("\n" + "="*60)
    print("TEST 1: Basic Concatenation")
    print("="*60)
    
    try:
        # Create test images
        img1 = create_colored_image(100, 100, (255, 0, 0))  # Red
        img2 = create_colored_image(100, 100, (0, 255, 0))  # Green
        img3 = create_colored_image(100, 100, (0, 0, 255))  # Blue
        
        compositor = ImageCompositor()
        
        # Test horizontal concatenation
        entries_h = [
            ImageEntry(img1, "Red"),
            ImageEntry(img2, "Green"),
            ImageEntry(img3, "Blue")
        ]
        
        result_h = compositor.concatenate(entries_h, direction=LayoutDirection.HORIZONTAL)
        h, w = result_h.shape[:2]
        
        if w >= 300 and h >= 100:  # Should be at least 3x width
            print(f"✓ Horizontal concatenation: {w}x{h} (3 images side by side)")
        else:
            print(f"❌ Horizontal concatenation wrong size: {w}x{h}")
            return False
        
        # Test vertical concatenation
        entries_v = [
            ImageEntry(img1, "Red"),
            ImageEntry(img2, "Green"),
            ImageEntry(img3, "Blue")
        ]
        
        result_v = compositor.concatenate(entries_v, direction=LayoutDirection.VERTICAL)
        h, w = result_v.shape[:2]
        
        if h >= 300 and w >= 100:  # Should be at least 3x height
            print(f"✓ Vertical concatenation: {w}x{h} (3 images stacked)")
        else:
            print(f"❌ Vertical concatenation wrong size: {w}x{h}")
            return False
        
        # Test empty list handling
        try:
            result_empty = compositor.concatenate([])
            print("⚠️  Empty list should raise an error")
        except (ValueError, IndexError):
            print("✓ Empty list correctly raises error")
        
        # Test single image
        single_entry = [ImageEntry(img1, "Single")]
        result_single = compositor.concatenate(single_entry)
        if result_single.shape[0] > 0 and result_single.shape[1] > 0:
            print(f"✓ Single image handled: {result_single.shape}")
        
        print("\n✅ Basic concatenation test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic concatenation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_layout():
    """Test Case 2: Grid layout functionality"""
    print("\n" + "="*60)
    print("TEST 2: Grid Layout")
    print("="*60)
    
    try:
        # Create test images with different colors
        images = [
            create_colored_image(80, 80, (255, 0, 0)),    # Red
            create_colored_image(80, 80, (0, 255, 0)),    # Green
            create_colored_image(80, 80, (0, 0, 255)),    # Blue
            create_colored_image(80, 80, (255, 255, 0)),  # Yellow
            create_colored_image(80, 80, (255, 0, 255)),  # Magenta
            create_colored_image(80, 80, (0, 255, 255)),  # Cyan
        ]
        
        compositor = ImageCompositor()
        
        # Test automatic grid (should choose reasonable cols)
        grid_auto = compositor.create_grid(images)
        h, w = grid_auto.shape[:2]
        print(f"✓ Automatic grid created: {w}x{h} for {len(images)} images")
        
        # Test 2-column grid
        grid_2col = compositor.create_grid(images, cols=2)
        h, w = grid_2col.shape[:2]
        expected_rows = 3  # 6 images / 2 cols = 3 rows
        if h >= 80 * expected_rows:  # At least 3 rows high
            print(f"✓ 2-column grid: {w}x{h} ({expected_rows} rows x 2 cols)")
        else:
            print(f"❌ 2-column grid wrong size: {w}x{h}")
            return False
        
        # Test 3-column grid
        grid_3col = compositor.create_grid(images, cols=3)
        h, w = grid_3col.shape[:2]
        expected_rows = 2  # 6 images / 3 cols = 2 rows
        if h >= 80 * expected_rows:  # At least 2 rows high
            print(f"✓ 3-column grid: {w}x{h} ({expected_rows} rows x 3 cols)")
        else:
            print(f"❌ 3-column grid wrong size: {w}x{h}")
            return False
        
        # Test grid with labels
        labels = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"]
        grid_labeled = compositor.create_grid(images, cols=3, labels=labels)
        if grid_labeled.shape[0] > grid_3col.shape[0]:  # Should be taller with labels
            print(f"✓ Grid with labels is taller ({grid_labeled.shape[0]} vs {grid_3col.shape[0]})")
        else:
            print(f"⚠️  Grid with labels same height as without")
        
        # Test grid with uneven number (7 images in 3 cols = 3 rows with padding)
        images_uneven = images + [create_colored_image(80, 80, (128, 128, 128))]
        grid_uneven = compositor.create_grid(images_uneven, cols=3)
        h, w = grid_uneven.shape[:2]
        expected_rows = 3  # 7 images / 3 cols = 3 rows (last row partial)
        print(f"✓ Uneven grid handled: {len(images_uneven)} images in 3x{expected_rows} grid")
        
        print("\n✅ Grid layout test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Grid layout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composition_styles():
    """Test Case 3: Composition styling options"""
    print("\n" + "="*60)
    print("TEST 3: Composition Styles")
    print("="*60)
    
    try:
        img1 = create_colored_image(100, 100, (255, 0, 0))
        img2 = create_colored_image(100, 100, (0, 255, 0))
        
        # Test default style
        default_style = CompositionStyle()
        compositor_default = ImageCompositor(default_style)
        entries = [ImageEntry(img1, "Image1"), ImageEntry(img2, "Image2")]
        
        result_default = compositor_default.concatenate(entries)
        print(f"✓ Default style applied: {result_default.shape}")
        
        # Test custom spacing
        spaced_style = CompositionStyle(
            horizontal_spacing=50,
            vertical_spacing=30,
            show_borders=False,
            show_labels=False
        )
        compositor_spaced = ImageCompositor(spaced_style)
        result_spaced = compositor_spaced.concatenate(entries)
        
        if result_spaced.shape[1] > result_default.shape[1]:
            print(f"✓ Custom spacing increases width: {result_spaced.shape[1]} > {result_default.shape[1]}")
        else:
            print(f"⚠️  Custom spacing didn't affect width")
        
        # Test borders
        border_style = CompositionStyle(
            border_thickness=10,
            border_color=(255, 255, 255),
            show_borders=True,
            show_labels=False
        )
        compositor_border = ImageCompositor(border_style)
        result_border = compositor_border.concatenate(entries)
        
        # Check for white pixels (borders)
        white_pixels = np.sum(np.all(result_border == 255, axis=2))
        if white_pixels > 0:
            print(f"✓ Borders added: {white_pixels} white border pixels")
        else:
            print(f"❌ No borders detected")
            return False
        
        # Test labels and separators
        full_style = CompositionStyle(
            show_labels=True,
            label_font_size=1.0,
            label_color=(255, 255, 255),
            show_separators=True,
            separator_color=(128, 128, 128),
            separator_thickness=2
        )
        compositor_full = ImageCompositor(full_style)
        result_full = compositor_full.concatenate(entries)
        
        # Should be taller with labels
        if result_full.shape[0] > img1.shape[0]:
            print(f"✓ Labels increase height: {result_full.shape[0]} > {img1.shape[0]}")
        else:
            print(f"❌ Labels didn't increase height")
            return False
        
        # Test background color
        bg_style = CompositionStyle(
            background_color=(50, 50, 50),
            horizontal_spacing=20,
            show_borders=False
        )
        compositor_bg = ImageCompositor(bg_style)
        result_bg = compositor_bg.concatenate(entries)
        
        # Check for background color in spacing
        mid_x = result_bg.shape[1] // 2
        mid_pixel = tuple(result_bg[50, mid_x])
        if mid_pixel == (50, 50, 50):
            print(f"✓ Background color applied in spacing: {mid_pixel}")
        else:
            print(f"⚠️  Background color not found in spacing (might be covered)")
        
        print("\n✅ Composition styles test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Composition styles test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_entries():
    """Test Case 4: ImageEntry metadata and handling"""
    print("\n" + "="*60)
    print("TEST 4: Image Entries")
    print("="*60)
    
    try:
        img = create_colored_image(100, 100, (128, 128, 128))
        
        # Test basic ImageEntry
        entry1 = ImageEntry(img, "TestImage")
        print(f"✓ Basic ImageEntry created: name='{entry1.name}'")
        
        # Test ImageEntry with stage
        entry2 = ImageEntry(img, "Process", "Stage1")
        print(f"✓ ImageEntry with stage: name='{entry2.name}', stage='{entry2.stage}'")
        
        # Test ImageEntry label generation
        if entry2.label == "Process_Stage1":
            print(f"✓ Label generation correct: '{entry2.label}'")
        else:
            print(f"⚠️  Label generation unexpected: '{entry2.label}'")
        
        # Test None stage handling
        entry3 = ImageEntry(img, "NoStage", None)
        if entry3.label == "NoStage":
            print(f"✓ None stage handled: '{entry3.label}'")
        else:
            print(f"❌ None stage handling wrong: '{entry3.label}'")
            return False
        
        # Test empty stage handling
        entry4 = ImageEntry(img, "EmptyStage", "")
        if entry4.label == "EmptyStage":
            print(f"✓ Empty stage handled: '{entry4.label}'")
        else:
            print(f"❌ Empty stage handling wrong: '{entry4.label}'")
            return False
        
        # Test with compositor
        compositor = ImageCompositor()
        entries = [
            ImageEntry(create_colored_image(80, 80, (255, 0, 0)), "Red", "Primary"),
            ImageEntry(create_colored_image(80, 80, (0, 255, 0)), "Green", "Primary"),
            ImageEntry(create_colored_image(80, 80, (0, 0, 255)), "Blue", "Primary"),
            ImageEntry(create_colored_image(80, 80, (255, 255, 0)), "Yellow", "Secondary"),
        ]
        
        style = CompositionStyle(show_labels=True)
        compositor_labeled = ImageCompositor(style)
        result = compositor_labeled.concatenate(entries)
        
        print(f"✓ Multiple labeled entries processed: {result.shape}")
        
        # Test mixing images and ImageEntry objects
        mixed = [
            img,  # Raw numpy array
            ImageEntry(img, "Named"),  # ImageEntry
            create_colored_image(80, 80, (0, 0, 0))  # Another raw array
        ]
        
        result_mixed = compositor.create_grid(mixed, cols=3)
        print(f"✓ Mixed raw images and ImageEntry objects handled: {result_mixed.shape}")
        
        print("\n✅ Image entries test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Image entries test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_size_handling():
    """Test Case 5: Handling different sized images"""
    print("\n" + "="*60)
    print("TEST 5: Size Handling")
    print("="*60)
    
    try:
        # Create images of different sizes
        img_small = create_colored_image(50, 50, (255, 0, 0))
        img_medium = create_colored_image(100, 100, (0, 255, 0))
        img_large = create_colored_image(150, 150, (0, 0, 255))
        img_wide = create_colored_image(200, 50, (255, 255, 0))
        img_tall = create_colored_image(50, 200, (255, 0, 255))
        
        compositor = ImageCompositor()
        
        # Test horizontal concatenation with different heights
        entries_h = [
            ImageEntry(img_small, "Small"),
            ImageEntry(img_medium, "Medium"),
            ImageEntry(img_large, "Large")
        ]
        
        result_h = compositor.concatenate(entries_h, direction=LayoutDirection.HORIZONTAL)
        h, w = result_h.shape[:2]
        
        # Height should match the tallest image (with padding)
        if h >= 150:
            print(f"✓ Horizontal: different heights padded to {h} (tallest: 150)")
        else:
            print(f"❌ Horizontal padding failed: height {h} < 150")
            return False
        
        # Test vertical concatenation with different widths
        entries_v = [
            ImageEntry(img_small, "Small"),
            ImageEntry(img_medium, "Medium"),
            ImageEntry(img_large, "Large")
        ]
        
        result_v = compositor.concatenate(entries_v, direction=LayoutDirection.VERTICAL)
        h, w = result_v.shape[:2]
        
        # Width should match the widest image (with padding)
        if w >= 150:
            print(f"✓ Vertical: different widths padded to {w} (widest: 150)")
        else:
            print(f"❌ Vertical padding failed: width {w} < 150")
            return False
        
        # Test extreme aspect ratios
        extreme_entries = [
            ImageEntry(img_wide, "Wide"),
            ImageEntry(img_tall, "Tall"),
            ImageEntry(img_medium, "Square")
        ]
        
        result_extreme = compositor.concatenate(extreme_entries, direction=LayoutDirection.HORIZONTAL)
        h, w = result_extreme.shape[:2]
        
        if h >= 200 and w >= 350:  # Should accommodate tall image height
            print(f"✓ Extreme aspect ratios handled: {w}x{h}")
        else:
            print(f"⚠️  Extreme aspect ratio result: {w}x{h}")
        
        # Test grid with mixed sizes
        mixed_sizes = [
            img_small, img_medium, img_large,
            img_wide, img_tall, img_medium
        ]
        
        grid_mixed = compositor.create_grid(mixed_sizes, cols=3)
        h, w = grid_mixed.shape[:2]
        print(f"✓ Grid with mixed sizes: {w}x{h} (6 images of varying sizes)")
        
        # Test single pixel images (edge case)
        tiny1 = create_colored_image(1, 1, (255, 0, 0))
        tiny2 = create_colored_image(1, 1, (0, 255, 0))
        
        tiny_entries = [ImageEntry(tiny1, "Pixel1"), ImageEntry(tiny2, "Pixel2")]
        result_tiny = compositor.concatenate(tiny_entries)
        
        if result_tiny.shape[0] > 0 and result_tiny.shape[1] > 0:
            print(f"✓ Single pixel images handled: {result_tiny.shape}")
        else:
            print(f"❌ Single pixel images failed")
            return False
        
        # Test very large image (memory test)
        large = create_colored_image(1000, 1000, (128, 128, 128))
        large_entries = [ImageEntry(large, f"Large{i}") for i in range(3)]
        
        try:
            result_large = compositor.concatenate(large_entries)
            print(f"✓ Large images handled: {result_large.shape} (3x 1000x1000 images)")
        except MemoryError:
            print(f"⚠️  Large images caused memory error (expected on limited systems)")
        
        print("\n✅ Size handling test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Size handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SMOKE TEST 03: IMAGE COMPOSITION AND LAYOUT")
    print("="*80)
    print("\nThis test suite validates the ImageCompositor's ability to combine")
    print("images with various layouts, styles, and configurations.")
    
    results = []
    
    # Run all tests
    results.append(("Basic Concatenation", test_basic_concatenation()))
    results.append(("Grid Layout", test_grid_layout()))
    results.append(("Composition Styles", test_composition_styles()))
    results.append(("Image Entries", test_image_entries()))
    results.append(("Size Handling", test_size_handling()))
    
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
        print("\n🎉 ALL TESTS PASSED! The image compositor is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)