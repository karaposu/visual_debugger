import numpy as np
import cv2
import os
from image_input_handler import UniversalImageInputHandler
from .annotations import (
    # Import the new type-specific annotation classes
    PointAnnotation, LabeledPointAnnotation, PointsAnnotation, LabeledPointsAnnotation,
    CircleAnnotation, LabeledCircleAnnotation, RectangleAnnotation,
    LineAnnotation, LabeledLineAnnotation, MaskAnnotation,
    # Factory functions for convenience
    point, labeled_point, circle, rectangle, line, bbox
)
from .image_processor import ImageProcessor
from .composition import ImageCompositor, CompositionStyle, ImageEntry, LayoutDirection


class VisualDebugger:
    def __init__(self, tag="visuals", debug_folder_path=None, generate_merged=False, active=True, output='save'):
        self.sequence = 1
        self.tag = tag
        self.active = active
        self.debug_folder_path = debug_folder_path if debug_folder_path else os.getcwd()
        self.generate_merged = generate_merged
        self.images = []
        self.output = output
        self.processor = ImageProcessor()

        if self.active:
            self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Checks if the directory exists and creates it if it does not."""
        if not os.path.exists(self.debug_folder_path):
            os.makedirs(self.debug_folder_path)
            print(f"Created directory: {self.debug_folder_path}")

    def increment_sequence(self):
        """Increments the sequence number."""
        if self.active:
            self.sequence += 1

    def reset_sequence(self):
        """Resets the sequence number and clears the image list."""
        if self.active:
            self.sequence = 1
            self.images = []

    def visual_debug(self, img, annotations=[], name="generic", stage_name=None, transparent=False, mask=False):
        """Handles visual debugging by annotating and saving or returning images."""

        #todo add assertion img must be uint8

        if not self.active:
            return

        img = cv2.imread(img) if isinstance(img, str) else img.copy()

        # Import here to avoid circular dependency
        from .info_panel import InfoPanel
        from .annotations import InfoPanelAnnotation
        
        # Handle InfoPanel directly - convert to InfoPanelAnnotation
        if isinstance(annotations, InfoPanel):
            annotations = [InfoPanelAnnotation(panel=annotations)]
        elif not isinstance(annotations, list):
            # Handle single annotation
            annotations = [annotations] if annotations else []
        else:
            # Process list - convert any InfoPanels to InfoPanelAnnotations
            processed_annotations = []
            for ann in annotations:
                if isinstance(ann, InfoPanel):
                    processed_annotations.append(InfoPanelAnnotation(panel=ann))
                else:
                    processed_annotations.append(ann)
            annotations = processed_annotations

        for annotation in annotations:
            self.processor.put_annotation_on_image(img, annotation)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if transparent else img

        if self.output == 'save':
            # Use an empty string as the default stage_name if not provided
            stage_name = stage_name if stage_name is not None else ""
            if stage_name != "":
                filename = f"{str(self.sequence).zfill(3)}_{self.tag}_{name}_{stage_name}.png"
            else:
                filename = f"{str(self.sequence).zfill(3)}_{self.tag}_{name}.png"
            
            full_path = os.path.join(self.debug_folder_path, filename)
            if mask:
                cv2.imwrite(full_path, img)
            else:
                cv2.imwrite(full_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.increment_sequence()
            self.images.append((img, name, stage_name))
        elif self.output == 'return':
            return img
        else:
            raise ValueError("Invalid output option. Use 'save' or 'return'.")

    def show_images_side_by_side(self, img1, img2, annotations1=[], annotations2=[], window_name="Comparison", scale=False):
        """
        Displays two images side by side with separate annotations for comparison.
        :param img1: First image or path to the first image.
        :param img2: Second image or path to the second image.
        :param annotations1: List of Annotation objects for the first image.
        :param annotations2: List of Annotation objects for the second image.
        :param window_name: The name of the window in which images will be shown.
        :param scale: If False, pads the images to match heights without scaling.
        """
        img1 = img1.copy()
        img2 = img2.copy()

        # Apply annotations
        for annotation in annotations1:
            self.processor.put_annotation_on_image(img1, annotation)
        for annotation in annotations2:
            self.processor.put_annotation_on_image(img2, annotation)

        if not scale:
            img1, img2 = self.processor.pad_images_to_match_height(img1, img2)

        combined_image = np.hstack((img1, img2))
        return combined_image

    def cook_merged_img(self, vertical_space=40, horizontal_space=50):
        """Creates a merged image from all debug images and saves it."""
        # Create compositor with style
        style = CompositionStyle(
            border_thickness=5,
            border_color=(255, 255, 255),
            vertical_spacing=vertical_space,
            horizontal_spacing=horizontal_space,
            show_labels=True,
            show_borders=True,
            show_separators=True
        )
        compositor = ImageCompositor(style)
        
        # Convert stored images to ImageEntry objects
        entries = [ImageEntry(img, name, stage) for img, name, stage in self.images]
        
        # Create concatenated image
        final_img = compositor.concatenate(entries, direction=LayoutDirection.HORIZONTAL)
        
        # Save the result
        filename = "0_merged.png"
        full_path = os.path.join(self.debug_folder_path, filename)
        print(f"Saving final image to {full_path}")
        cv2.imwrite(full_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    def visualize_mask(self, image_tensor, mask, name="mask", stage_name=None):
        """Visualizes a mask on the image and saves or returns the result."""
        image = image_tensor.numpy().transpose(1, 2, 0) * 255
        image = image.astype(np.uint8)

        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color_mapping = np.array([
            [0, 0, 0],
            [0, 153, 255],
            [102, 255, 153],
            [0, 204, 153],
            [255, 255, 102],
            [255, 255, 204],
            [255, 153, 0],
            [255, 102, 255],
            [102, 0, 51],
            [255, 204, 255],
            [255, 0, 102]
        ])

        for index, color in enumerate(color_mapping):
            color_mask[mask == index] = color

        overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

        if self.output == 'save':
            stage_name = stage_name if stage_name is not None else ""
            filename = f"{str(self.sequence).zfill(3)}_{self.tag}_{name}_{stage_name}.png"
            full_path = os.path.join(self.debug_folder_path, filename)
            cv2.imwrite(full_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
            self.increment_sequence()
            self.images.append((overlayed_image, name, stage_name))
        elif self.output == 'return':
            return overlayed_image
        else:
            raise ValueError("Invalid output option. Use 'save' or 'return'.")

def main():
    """Example usage with the new annotation system"""
    uiih = UniversalImageInputHandler("/mnt/data/Screenshot 2024-07-09 at 14.21.19.png")
    img = uiih.img

    # Using the new type-specific annotations
    annotations1 = [
        labeled_point(100, 100, "Point 1", color=(255, 0, 0))
    ]
    
    annotations2 = [
        LabeledPointsAnnotation(
            points=[(70, 80), (90, 110)], 
            labels=["P2", "P3"],
            color=(0, 255, 0)
        )
    ]
    
    annotations3 = [
        LabeledCircleAnnotation(
            center=(150, 195), 
            radius=20, 
            thickness=2, 
            label="Circle 1",
            color=(0, 0, 255)
        )
    ]
    
    annotations4 = [
        rectangle(50, 50, 100, 100, color=(255, 255, 0)),
        line(10, 10, 190, 190, color=(255, 0, 0)),
        LabeledLineAnnotation(
            start=(20, 20), 
            end=(180, 180), 
            label="Line 1",
            color=(0, 255, 255)
        )
    ]

    # Create mask for testing
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=int)
    section_width = width // 3
    mask[:, :section_width] = 1  # First section
    mask[:, section_width:2 * section_width] = 2  # Second section
    mask[:, 2 * section_width:] = 3  # Third section
    if width % 3 != 0:
        remaining_start = 2 * section_width
        mask[:, remaining_start:] = 3

    annotations5 = [MaskAnnotation(mask=mask, alpha=0.5, colormap='random')]

    print("shape mask:", mask.shape, mask.dtype)
    print("shape img :", img.shape, img.dtype)

    vd = VisualDebugger(tag="visuals", debug_folder_path="./", active=True)

    vd.visual_debug(img, annotations1, name="image1", stage_name="")
    vd.visual_debug(img, annotations2, name="image2", stage_name="")
    vd.visual_debug(img, annotations3, name="image3", stage_name="step1")
    vd.visual_debug(img, annotations4, name="image3", stage_name="step2")
    vd.visual_debug(img, annotations5, name="parsing_mask", stage_name="")

    vd.cook_merged_img(vertical_space=30, horizontal_space=50)

if __name__ == '__main__':
    main()