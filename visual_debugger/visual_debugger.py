import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import os
from enum import Enum, auto
from image_input_handler import UniversalImageInputHandler

class AnnotationType(Enum):
    POINT = auto()
    POINT_AND_LABEL = auto()
    POINTS = auto()
    POINTS_AND_LABELS = auto()
    CIRCLE = auto()
    CIRCLE_AND_LABEL = auto()
    RECTANGLE = auto()
    PITCH_YAW_ROLL = auto()
    LINE = auto()  # New line annotation type
    LINE_AND_LABEL = auto()  # New line with label annotation type

@dataclass
class Annotation:
    type: AnnotationType
    coordinates: Optional[Union[Tuple[int, int], List[Tuple[int, int]]]] = None
    color: Tuple[int, int, int] = (255, 0, 0)
    labels: Optional[Union[str, List[str]]] = None
    radius : Optional[float] = None
    thickness: Optional[float] = None
    orientation: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        if self.type in {AnnotationType.POINTS_AND_LABELS, AnnotationType.POINT_AND_LABEL} and isinstance(self.labels, list):
            if len(self.coordinates) != len(self.labels):
                raise ValueError("The number of labels must match the number of coordinates.")
        if self.type == AnnotationType.PITCH_YAW_ROLL and self.orientation is None:
            raise ValueError("Orientation (pitch, yaw, roll) must be provided for 'pitch_yaw_roll' type.")

class VisualDebugger:
    def __init__(self, tag="visuals", debug_folder_path=None, generate_merged=False, active=True, output='save'):
        self.sequence = 1
        self.tag = tag
        self.active = active
        self.debug_folder_path = debug_folder_path if debug_folder_path is not None else os.getcwd()
        self.generate_merged = generate_merged
        self.images = []
        self.output = output

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

    def concat_images(self, axis=1, border_thickness=5, border_color=(255, 255, 255), vertical_space=20, horizontal_space=20):
        """Concatenates all stored images into one large image with labels and optional borders."""
        grouped_images = self.group_images_by_name()

        if axis == 1:  # Horizontal concatenation
            return self.concat_images_horizontally(grouped_images, border_thickness, border_color, vertical_space, horizontal_space)
        else:  # Vertical concatenation
            return self.concat_images_vertically(grouped_images, border_thickness, border_color, vertical_space, horizontal_space)

    def group_images_by_name(self):
        """Groups images by their name."""
        grouped_images = {}
        for img, name, stage_name in self.images:
            if name not in grouped_images:
                grouped_images[name] = []
            grouped_images[name].append((img, stage_name))
        return grouped_images

    def calculate_horizontal_dimensions(self, grouped_images, border_thickness, vertical_space, horizontal_space):
        """Calculates dimensions for horizontally concatenated images."""
        max_height = 0
        total_width = 0
        for name, imgs in grouped_images.items():
            max_height = max(max_height, sum(img.shape[0] + 2 * border_thickness + vertical_space for img, _ in imgs))
            total_width += max(img.shape[1] + 2 * border_thickness for img, _ in imgs) + horizontal_space
        return max_height, total_width

    def add_labels_and_borders(self, img, name, stage_name, border_thickness, border_color):
        """Adds labels and borders to an individual image."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        padded_img = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness,
                                        cv2.BORDER_CONSTANT, value=border_color)
        label_text = f"{name}_{stage_name}"
        return padded_img, label_text

    def place_image_on_final(self, final_img, padded_img, label_text, current_x, current_y, border_thickness, border_color):
        """Places the image with labels and borders on the final image."""
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_position = (current_x + 5, current_y - 10)
        cv2.putText(final_img, label_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, border_color, 1)
        final_img[current_y:current_y + padded_img.shape[0], current_x:current_x + padded_img.shape[1]] = padded_img
        return final_img

    def create_horizontal_image(self, grouped_images, max_height, total_width, border_thickness, border_color, vertical_space, horizontal_space):
        """Creates the final horizontally concatenated image."""
        final_img = np.zeros((max_height + 100, total_width, 3), dtype=np.uint8)
        current_x = 0
        for name, imgs in grouped_images.items():
            current_y = 70
            for img, stage_name in imgs:
                padded_img, label_text = self.add_labels_and_borders(img, name, stage_name, border_thickness, border_color)
                if current_y + padded_img.shape[0] > final_img.shape[0]:
                    raise ValueError("Current_y exceeds the final image height.")
                final_img = self.place_image_on_final(final_img, padded_img, label_text, current_x, current_y, border_thickness, border_color)
                current_y += padded_img.shape[0] + vertical_space
            current_x += max(img.shape[1] + 2 * border_thickness for img, _ in imgs) + horizontal_space
        return final_img

    def concat_images_horizontally(self, grouped_images, border_thickness, border_color, vertical_space, horizontal_space):
        """Concatenates images horizontally with labels and borders."""
        max_height, total_width = self.calculate_horizontal_dimensions(grouped_images, border_thickness, vertical_space, horizontal_space)
        return self.create_horizontal_image(grouped_images, max_height, total_width, border_thickness, border_color, vertical_space, horizontal_space)

    def concat_images_vertically(self, grouped_images, border_thickness, border_color, vertical_space, horizontal_space):
        """Concatenates images vertically with labels and borders."""
        max_width = 0
        total_height = 0
        for name, imgs in grouped_images.items():
            max_width = max(max_width, sum(img.shape[1] + 2 * border_thickness + horizontal_space for img, _ in imgs))
            total_height += max(img.shape[0] + 2 * border_thickness for img, _ in imgs) + vertical_space

        final_img = np.zeros((total_height + 100, max_width, 3), dtype=np.uint8)
        current_y = 70
        for name, imgs in grouped_images.items():
            current_x = 0
            for img, stage_name in imgs:
                padded_img, label_text = self.add_labels_and_borders(img, name, stage_name, border_thickness, border_color)
                if current_x + padded_img.shape[1] > final_img.shape[1]:
                    raise ValueError("Current_x exceeds the final image width.")
                final_img = self.place_image_on_final(final_img, padded_img, label_text, current_x, current_y, border_thickness, border_color)
                current_x += padded_img.shape[1] + horizontal_space
            current_y += max(img.shape[0] + 2 * border_thickness for img, _ in imgs) + vertical_space
        return final_img

    def visual_debug(self, img, annotations=[], name="generic", stage_name=None, transparent=False, mask=False):
        """Handles visual debugging by annotating and saving or returning images."""
        if not self.active:
            return

        img = cv2.imread(img) if isinstance(img, str) else img.copy()

        for annotation in annotations:
            self.put_annotation_on_image(img, annotation)

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

    def cook_merged_img(self, vertical_space=20, horizontal_space=20):
        """Creates a merged image from all debug images and saves it."""
        final_img = self.concat_images(vertical_space=vertical_space, horizontal_space=horizontal_space)
        filename = "0_merged.png"
        full_path = os.path.join(self.debug_folder_path, filename)
        cv2.imwrite(full_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    def draw_orientation(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        """Draws the orientation axes on the image."""
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
        roll = np.deg2rad(roll)

        if tdx is None or tdy is None:
            height, width = img.shape[:2]
            tdx = width // 2
            tdy = height // 2

        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        axis_points = np.array([[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
        transformed_points = axis_points @ R.T
        transformed_points = transformed_points + np.array([tdx, tdy, 0])

        img = cv2.line(img, (tdx, tdy), (int(transformed_points[1][0]), int(transformed_points[1][1])), (255, 0, 0), 3)
        img = cv2.line(img, (tdx, tdy), (int(transformed_points[2][0]), int(transformed_points[2][1])), (0, 255, 0), 3)
        img = cv2.line(img, (tdx, tdy), (int(transformed_points[3][0]), int(transformed_points[3][1])), (0, 0, 255), 3)
        return img

    def put_circle_and_text_on_image(self, img, text, coordinates, radius, thickness, color):
        """Puts a circle and text on the image at the specified coordinates."""

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (coordinates[0] + 10, coordinates[1])
        fontScale = 0.5
        img = cv2.putText(img, text, org, font, fontScale, color, 1, cv2.LINE_AA)
        img = cv2.circle(img, coordinates, radius, color, thickness)
        return img

    def put_circle_on_image(self, img, coordinates, color):
        """Puts a circle on the image at the specified coordinates."""
        radius = 3
        thickness = -1
        img = cv2.circle(img, coordinates, radius, color, thickness)
        return img

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
            self.put_annotation_on_image(img1, annotation)
        for annotation in annotations2:
            self.put_annotation_on_image(img2, annotation)

        if not scale:
            img1, img2 = self.pad_images_to_match_height(img1, img2)

        combined_image = np.hstack((img1, img2))
        return combined_image

    def put_annotation_on_image(self, image, annotation: Annotation):
        """Puts the specified annotation on the image."""
        if annotation.type == AnnotationType.CIRCLE:
            cv2.circle(image, annotation.coordinates, 5, annotation.color, -1)
        elif annotation.type == AnnotationType.CIRCLE_AND_LABEL:
            self.put_circle_and_text_on_image(image, annotation.labels, annotation.coordinates, annotation.radius, annotation.thickness, annotation.color)
        elif annotation.type == AnnotationType.RECTANGLE:
            x, y, w, h = annotation.coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), annotation.color, 2)
        elif annotation.type in {AnnotationType.POINTS, AnnotationType.POINT}:
            points = annotation.coordinates if isinstance(annotation.coordinates, list) else [annotation.coordinates]
            for point in points:
                cv2.circle(image, point, 5, annotation.color, -1)

        elif annotation.type == AnnotationType.POINT_AND_LABEL:
            point = annotation.coordinates
            label = annotation.labels
            cv2.circle(image, point, 5, annotation.color, -1)
            cv2.putText(image, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation.color, 1)

        elif annotation.type == AnnotationType.POINTS_AND_LABELS:
            for point, label in zip(annotation.coordinates, annotation.labels):
                cv2.circle(image, point, 5, annotation.color, -1)
                cv2.putText(image, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation.color, 1)

        elif annotation.type == AnnotationType.PITCH_YAW_ROLL:
            p, y, r = annotation.orientation
            tdx, tdy = annotation.coordinates if annotation.coordinates else (None, None)
            image = self.draw_orientation(image, y, p, r, tdx, tdy)

        elif annotation.type == AnnotationType.LINE:
            # Draw a line between two points
            cv2.line(image, annotation.coordinates[0], annotation.coordinates[1], annotation.color, 2)
        elif annotation.type == AnnotationType.LINE_AND_LABEL:
            # Draw a line and then put a label
            cv2.line(image, annotation.coordinates[0], annotation.coordinates[1], annotation.color, 2)
            midpoint = ((annotation.coordinates[0][0] + annotation.coordinates[1][0]) // 2,
                        (annotation.coordinates[0][1] + annotation.coordinates[1][1]) // 2)
            cv2.putText(image, annotation.labels, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation.color, 1)

    def pad_images_to_match_height(self, img1, img2):
        """Pads images to match their height."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 > h2:
            pad_amount = h1 - h2
            pad_top = pad_amount // 2
            pad_bottom = pad_amount - pad_top
            img2 = cv2.copyMakeBorder(img2, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif h2 > h1:
            pad_amount = h2 - h1
            pad_top = pad_amount // 2
            pad_bottom = pad_amount - pad_top
            img1 = cv2.copyMakeBorder(img1, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return img1, img2

def main():
    img1 = np.ones((200, 300, 3), dtype=np.uint8) * 255
    img2 = np.ones((150, 350, 3), dtype=np.uint8) * 200
    img3 = np.ones((250, 250, 3), dtype=np.uint8) * 150
    img4 = np.ones((300, 200, 3), dtype=np.uint8) * 100

    images = [img1, img2, img3, img4]

    # Create UniversalImageInputHandler instances for each image
    handlers = [UniversalImageInputHandler(img) for img in images]

    # Create annotations for each image
    annotations1 = [
        Annotation(type=AnnotationType.POINT_AND_LABEL, coordinates=(100, 100), labels="Point 1")
    ]
    annotations2 = [
        Annotation(type=AnnotationType.POINTS_AND_LABELS, coordinates=[(70, 80), (90, 110)], labels=["P2", "P3"])
    ]
    annotations3 = [
        Annotation(type=AnnotationType.CIRCLE_AND_LABEL, coordinates=(150, 195), radius= 20, thickness= 2,  labels="Circle 1")
    ]
    annotations4 = [
        Annotation(type=AnnotationType.RECTANGLE, coordinates=(50, 50, 100, 100)),
        Annotation(type=AnnotationType.LINE, coordinates=((10, 10), (190, 190))),
        Annotation(type=AnnotationType.LINE_AND_LABEL, coordinates=((20, 20), (180, 180)), labels="Line 1")
    ]

    # Create VisualDebugger instance
    vd = VisualDebugger(tag="visuals", debug_folder_path="./", active=True)

    # Apply visual_debug for each image with annotations
    vd.visual_debug(handlers[0].img, annotations1, name="image1", stage_name="")
    vd.visual_debug(handlers[1].img, annotations2, name="image2", stage_name="")
    vd.visual_debug(handlers[2].img, annotations3, name="image3", stage_name="step1")
    vd.visual_debug(handlers[3].img, annotations4, name="image3", stage_name="step2")

    # Merge and save all debug images
    vd.cook_merged_img(vertical_space=30, horizontal_space=50)

if __name__ == '__main__':
    main()
