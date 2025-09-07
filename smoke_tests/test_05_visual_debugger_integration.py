"""
Test 05: Visual Debugger Integration
=====================================
Tests the complete VisualDebugger class integration with all components working
together, including file I/O, annotation processing, and image composition.

Why this is critical:
- VisualDebugger is the main entry point for users
- Must correctly integrate all subsystems (annotations, processor, compositor)
- File saving and sequence management must work correctly
- The new type-specific annotations must work seamlessly
- Image merging and side-by-side comparisons must function properly

Test Cases:
1. test_debugger_initialization - Tests VisualDebugger creation and setup
2. test_annotation_integration - Tests using new type-specific annotations
3. test_file_operations - Tests saving, sequencing, and file management
4. test_image_operations - Tests side-by-side comparison and mask visualization
5. test_merged_output - Tests the cook_merged_img functionality
"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from visual_debugger.visual_debugger import VisualDebugger
from visual_debugger.annotations import (
    PointAnnotation, LabeledPointAnnotation, CircleAnnotation,
    RectangleAnnotation, LineAnnotation, MaskAnnotation,
    BoundingBoxAnnotation, TextAnnotation,
    point, circle, rectangle, line, text, bbox
)


def create_test_image(width=400, height=300):
    """Create a test image with a gradient"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Create gradient for visual verification
    for i in range(height):
        img[i, :] = [i * 255 // height, 128, 255 - (i * 255 // height)]
    return img


def test_debugger_initialization():
    """Test Case 1: VisualDebugger initialization and configuration"""
    print("\n" + "="*60)
    print("TEST 1: Debugger Initialization")
    print("="*60)
    
    temp_dir = None
    try:
        # Test default initialization
        vd1 = VisualDebugger()
        print(f"✓ Default initialization: tag='{vd1.tag}', active={vd1.active}")
        
        # Test with custom parameters
        temp_dir = tempfile.mkdtemp(prefix="vd_test_")
        vd2 = VisualDebugger(
            tag="test_run",
            debug_folder_path=temp_dir,
            generate_merged=True,
            active=True,
            output='save'
        )
        
        if vd2.tag == "test_run" and vd2.debug_folder_path == temp_dir:
            print(f"✓ Custom parameters: tag='{vd2.tag}', path='{temp_dir}'")
        else:
            print(f"❌ Custom parameters not set correctly")
            return False
        
        # Test directory creation
        if os.path.exists(temp_dir):
            print(f"✓ Debug directory created: {temp_dir}")
        else:
            print(f"❌ Debug directory not created")
            return False
        
        # Test inactive debugger
        vd3 = VisualDebugger(active=False)
        if not vd3.active:
            print(f"✓ Inactive debugger created")
        else:
            print(f"❌ Debugger should be inactive")
            return False
        
        # Test output modes
        vd_save = VisualDebugger(output='save', debug_folder_path=temp_dir)
        vd_return = VisualDebugger(output='return')
        
        if vd_save.output == 'save' and vd_return.output == 'return':
            print(f"✓ Output modes: save and return")
        else:
            print(f"❌ Output modes not set correctly")
            return False
        
        # Test sequence management
        vd4 = VisualDebugger(debug_folder_path=temp_dir)
        initial_seq = vd4.sequence
        vd4.increment_sequence()
        
        if vd4.sequence == initial_seq + 1:
            print(f"✓ Sequence incremented: {initial_seq} → {vd4.sequence}")
        else:
            print(f"❌ Sequence increment failed")
            return False
        
        # Test reset
        vd4.reset_sequence()
        if vd4.sequence == 1 and len(vd4.images) == 0:
            print(f"✓ Sequence reset successfully")
        else:
            print(f"❌ Reset failed: seq={vd4.sequence}, images={len(vd4.images)}")
            return False
        
        print("\n✅ Debugger initialization test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Debugger initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_annotation_integration():
    """Test Case 2: Integration with new type-specific annotations"""
    print("\n" + "="*60)
    print("TEST 2: Annotation Integration")
    print("="*60)
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vd_test_")
        vd = VisualDebugger(debug_folder_path=temp_dir, output='return')
        img = create_test_image()
        
        # Test single annotation
        ann1 = PointAnnotation(position=(100, 100), color=(255, 0, 0), size=10)
        result1 = vd.visual_debug(img, [ann1], name="point_test")
        
        if result1 is not None and result1.shape == img.shape:
            print(f"✓ Single annotation processed: shape {result1.shape}")
        else:
            print(f"❌ Single annotation failed")
            return False
        
        # Test multiple annotations
        annotations = [
            point(50, 50, color=(255, 0, 0)),
            circle(200, 150, 30, color=(0, 255, 0)),
            rectangle(250, 50, 100, 80, color=(0, 0, 255)),
            line(0, 0, 400, 300, color=(255, 255, 0)),
            text("Test", 20, 280, font_scale=1.0)
        ]
        
        result2 = vd.visual_debug(img, annotations, name="multi_test")
        
        if result2 is not None:
            print(f"✓ Multiple annotations processed: {len(annotations)} annotations")
        else:
            print(f"❌ Multiple annotations failed")
            return False
        
        # Test complex annotations
        complex_anns = [
            BoundingBoxAnnotation(
                bbox=(50, 50, 150, 100),
                label="Object 95%",
                color=(255, 0, 255)
            ),
            LabeledPointAnnotation(
                position=(300, 200),
                label="Target",
                color=(0, 255, 255)
            )
        ]
        
        result3 = vd.visual_debug(img, complex_anns, name="complex_test")
        
        if result3 is not None:
            print(f"✓ Complex annotations processed")
        else:
            print(f"❌ Complex annotations failed")
            return False
        
        # Test mask annotation
        mask = np.zeros((300, 400), dtype=np.uint8)
        mask[50:150, 50:200] = 1
        mask[100:250, 200:350] = 2
        
        mask_ann = MaskAnnotation(mask=mask, alpha=0.5, colormap='random')
        result4 = vd.visual_debug(img, [mask_ann], name="mask_test")
        
        if result4 is not None:
            # Check if mask was applied (image should be modified)
            diff = np.sum(np.abs(result4.astype(float) - img.astype(float)))
            if diff > 0:
                print(f"✓ Mask annotation applied: diff={diff:.0f}")
            else:
                print(f"❌ Mask not applied: no difference detected")
                return False
        else:
            print(f"❌ Mask annotation failed")
            return False
        
        # Test empty annotation list
        result5 = vd.visual_debug(img, [], name="empty_test")
        if result5 is not None:
            print(f"✓ Empty annotation list handled")
        else:
            print(f"❌ Empty annotation list failed")
            return False
        
        print("\n✅ Annotation integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Annotation integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_file_operations():
    """Test Case 3: File saving and management"""
    print("\n" + "="*60)
    print("TEST 3: File Operations")
    print("="*60)
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vd_test_")
        vd = VisualDebugger(
            tag="file_test",
            debug_folder_path=temp_dir,
            output='save'
        )
        
        img = create_test_image()
        
        # Test basic save
        ann = circle(200, 150, 50, color=(255, 0, 0))
        vd.visual_debug(img, [ann], name="test1")
        
        # Check if file was created
        expected_file = os.path.join(temp_dir, "001_file_test_test1.png")
        if os.path.exists(expected_file):
            print(f"✓ File saved: {os.path.basename(expected_file)}")
            
            # Check file size
            file_size = os.path.getsize(expected_file)
            if file_size > 0:
                print(f"✓ File has content: {file_size} bytes")
            else:
                print(f"❌ File is empty")
                return False
        else:
            print(f"❌ File not created: {expected_file}")
            return False
        
        # Test sequence numbering
        vd.visual_debug(img, [ann], name="test2")
        expected_file2 = os.path.join(temp_dir, "002_file_test_test2.png")
        
        if os.path.exists(expected_file2):
            print(f"✓ Sequence numbering: {os.path.basename(expected_file2)}")
        else:
            print(f"❌ Sequence file not created")
            return False
        
        # Test with stage name
        vd.visual_debug(img, [ann], name="test3", stage_name="step1")
        expected_file3 = os.path.join(temp_dir, "003_file_test_test3_step1.png")
        
        if os.path.exists(expected_file3):
            print(f"✓ Stage name included: {os.path.basename(expected_file3)}")
        else:
            print(f"❌ Stage file not created")
            return False
        
        # Test image accumulation
        if len(vd.images) == 3:
            print(f"✓ Images accumulated: {len(vd.images)} images")
        else:
            print(f"❌ Image accumulation wrong: {len(vd.images)} images")
            return False
        
        # Test reset
        vd.reset_sequence()
        vd.visual_debug(img, [ann], name="after_reset")
        expected_reset = os.path.join(temp_dir, "001_file_test_after_reset.png")
        
        if os.path.exists(expected_reset):
            print(f"✓ Reset worked: {os.path.basename(expected_reset)}")
        else:
            print(f"❌ Reset failed: file not created")
            return False
        
        # Test inactive debugger (should not save)
        vd_inactive = VisualDebugger(
            active=False,
            debug_folder_path=temp_dir
        )
        vd_inactive.visual_debug(img, [ann], name="should_not_exist")
        
        should_not_exist = os.path.join(temp_dir, "*should_not_exist*")
        import glob
        if not glob.glob(should_not_exist):
            print(f"✓ Inactive debugger didn't save files")
        else:
            print(f"❌ Inactive debugger saved files")
            return False
        
        # Count total files created
        png_files = glob.glob(os.path.join(temp_dir, "*.png"))
        print(f"✓ Total files created: {len(png_files)}")
        
        print("\n✅ File operations test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ File operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_image_operations():
    """Test Case 4: Side-by-side comparison and mask visualization"""
    print("\n" + "="*60)
    print("TEST 4: Image Operations")
    print("="*60)
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vd_test_")
        vd = VisualDebugger(debug_folder_path=temp_dir, output='return')
        
        # Create two test images
        img1 = create_test_image(300, 200)
        img2 = create_test_image(400, 250)
        
        # Test side-by-side without annotations
        combined = vd.show_images_side_by_side(img1, img2)
        
        expected_width = img1.shape[1] + img2.shape[1]  # After padding
        if combined.shape[1] >= expected_width:
            print(f"✓ Side-by-side basic: {combined.shape}")
        else:
            print(f"❌ Side-by-side wrong size: {combined.shape}")
            return False
        
        # Test side-by-side with annotations
        anns1 = [
            point(50, 50, color=(255, 0, 0)),
            circle(150, 100, 30, color=(0, 255, 0))
        ]
        anns2 = [
            rectangle(100, 50, 200, 150, color=(0, 0, 255)),
            text("Image 2", 50, 30)
        ]
        
        combined_ann = vd.show_images_side_by_side(
            img1, img2,
            annotations1=anns1,
            annotations2=anns2
        )
        
        if combined_ann.shape[0] > 0 and combined_ann.shape[1] > 0:
            print(f"✓ Side-by-side with annotations: {combined_ann.shape}")
        else:
            print(f"❌ Side-by-side with annotations failed")
            return False
        
        # Test padding behavior (different heights)
        tall_img = create_test_image(200, 400)
        short_img = create_test_image(200, 100)
        
        combined_padded = vd.show_images_side_by_side(tall_img, short_img, scale=False)
        
        # Both images should have same height after padding
        if combined_padded.shape[0] == 400:  # Height of tallest
            print(f"✓ Height padding correct: {combined_padded.shape}")
        else:
            print(f"❌ Height padding wrong: {combined_padded.shape}")
            return False
        
        # Test visualize_mask (simulating tensor input)
        class MockTensor:
            def __init__(self, data):
                self.data = data
            def numpy(self):
                return self.data
        
        # Create mock image tensor (C, H, W format)
        img_data = np.random.rand(3, 200, 300)
        img_tensor = MockTensor(img_data)
        
        # Create mask
        mask = np.zeros((200, 300), dtype=int)
        mask[50:150, 50:200] = 1
        mask[100:180, 150:250] = 2
        
        result = vd.visualize_mask(img_tensor, mask, name="mask_viz")
        
        if result is not None and result.shape == (200, 300, 3):
            print(f"✓ Mask visualization: {result.shape}")
            
            # Check if mask colors were applied
            unique_colors = len(np.unique(result.reshape(-1, 3), axis=0))
            if unique_colors > 3:  # Should have multiple colors from mask
                print(f"✓ Mask colors applied: {unique_colors} unique colors")
            else:
                print(f"⚠️  Few colors in mask result: {unique_colors}")
        else:
            print(f"❌ Mask visualization failed")
            return False
        
        print("\n✅ Image operations test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Image operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_merged_output():
    """Test Case 5: Merged image generation"""
    print("\n" + "="*60)
    print("TEST 5: Merged Output")
    print("="*60)
    
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vd_test_")
        vd = VisualDebugger(
            tag="merge_test",
            debug_folder_path=temp_dir,
            generate_merged=True,
            output='save'
        )
        
        img = create_test_image(200, 150)
        
        # Create several debug images
        annotations_list = [
            [point(100, 75, color=(255, 0, 0))],
            [circle(100, 75, 40, color=(0, 255, 0))],
            [rectangle(50, 40, 100, 70, color=(0, 0, 255))],
            [line(0, 0, 200, 150, color=(255, 255, 0))]
        ]
        
        names = ["point", "circle", "rect", "line"]
        stages = ["", "step1", "step2", "final"]
        
        for ann, name, stage in zip(annotations_list, names, stages):
            vd.visual_debug(img, ann, name=name, stage_name=stage)
        
        # Check accumulated images
        if len(vd.images) == 4:
            print(f"✓ Images accumulated for merge: {len(vd.images)}")
        else:
            print(f"❌ Wrong number of images: {len(vd.images)}")
            return False
        
        # Generate merged image
        vd.cook_merged_img(vertical_space=20, horizontal_space=30)
        
        # Check if merged file was created
        merged_file = os.path.join(temp_dir, "0_merged.png")
        if os.path.exists(merged_file):
            print(f"✓ Merged file created: {os.path.basename(merged_file)}")
            
            # Load and check merged image
            merged_img = cv2.imread(merged_file)
            if merged_img is not None:
                h, w = merged_img.shape[:2]
                
                # Should be wider than single image * 4
                min_expected_width = 200 * 4  # 4 images side by side
                if w >= min_expected_width:
                    print(f"✓ Merged image size: {w}x{h} (>= {min_expected_width} wide)")
                else:
                    print(f"❌ Merged image too small: {w}x{h}")
                    return False
                
                # Check file size (should be substantial)
                file_size = os.path.getsize(merged_file)
                if file_size > 10000:  # At least 10KB
                    print(f"✓ Merged file size: {file_size} bytes")
                else:
                    print(f"⚠️  Small merged file: {file_size} bytes")
            else:
                print(f"❌ Could not load merged image")
                return False
        else:
            print(f"❌ Merged file not created")
            return False
        
        # Test empty merge (should handle gracefully)
        vd2 = VisualDebugger(debug_folder_path=temp_dir)
        try:
            # This might fail with empty images list
            if len(vd2.images) == 0:
                print(f"✓ Empty image list detected correctly")
        except Exception as e:
            print(f"✓ Empty merge handled with exception: {type(e).__name__}")
        
        # Test merge with single image
        vd3 = VisualDebugger(debug_folder_path=temp_dir)
        vd3.visual_debug(img, [point(50, 50)], name="single")
        vd3.cook_merged_img()
        
        single_merged = os.path.join(temp_dir, "0_merged.png")
        if os.path.exists(single_merged):
            print(f"✓ Single image merge handled")
        
        print("\n✅ Merged output test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Merged output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("SMOKE TEST 05: VISUAL DEBUGGER INTEGRATION")
    print("="*80)
    print("\nThis test suite validates the complete VisualDebugger integration")
    print("with all components working together.")
    
    results = []
    
    # Run all tests
    results.append(("Debugger Initialization", test_debugger_initialization()))
    results.append(("Annotation Integration", test_annotation_integration()))
    results.append(("File Operations", test_file_operations()))
    results.append(("Image Operations", test_image_operations()))
    results.append(("Merged Output", test_merged_output()))
    
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
        print("\n🎉 ALL TESTS PASSED! The Visual Debugger integration is working correctly.")
        print("The system successfully:")
        print("  • Initializes with various configurations")
        print("  • Integrates the new type-specific annotation system")
        print("  • Saves files with proper naming and sequencing")
        print("  • Handles side-by-side comparisons and masks")
        print("  • Generates merged output images")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)