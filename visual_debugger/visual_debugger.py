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


@dataclass
class Annotation:
    type: AnnotationType
    coordinates: Union[Tuple[int, int], List[Tuple[int, int]]]
    color: Tuple[int, int, int] = (255, 0, 0)
    labels: Optional[Union[str, List[str]]] = None
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

    def visual_debug(self, img, annotations=[], process_step="generic", condition="", transparent=False, mask=False):
        """Handles visual debugging by annotating and saving or returning images."""
        if not self.active:
            return

        img = cv2.imread(img) if isinstance(img, str) else img.copy()

        for annotation in annotations:
            self.put_annotation_on_image(img, annotation)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if transparent else img

        if self.output == 'save':
            filename = f"{str(self.sequence).zfill(3)}_{self.tag}_{process_step}_{condition}.png"
            full_path = os.path.join(self.debug_folder_path, filename)
            if mask:
                cv2.imwrite(full_path, img)
            else:
                cv2.imwrite(full_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            self.increment_sequence()
            self.images.append((img, f"{process_step}_{condition}"))
        elif self.output == 'return':
            return img
        else:
            raise ValueError("Invalid output option. Use 'save' or 'return'.")

    def cook_merged_img(self):
        """Creates a merged image from all debug images and saves it."""
        final_img = self.concat_images()
        filename = "0_merged.png"
        full_path = os.path.join(self.debug_folder_path, filename)
        cv2.imwrite(full_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    def concat_images(self, axis=1):
        """Concatenates all stored images into one large image with labels."""
        max_height = max(img.shape[0] for img, _ in self.images)
        total_width = sum(img.shape[1] for img, _ in self.images)
        final_img = np.zeros((max_height + 50, total_width, 3), dtype=np.uint8)

        current_x = 0
        for img, label in self.images:
            if img.shape[0] != max_height:
                img = cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
            cv2.putText(img, label, (5, max_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            final_img[50:max_height + 50, current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1]

        return final_img

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

    def put_circle_and_text_on_image(self, img, text, coordinates, color):
        """Puts a circle and text on the image at the specified coordinates."""
        radius = 3
        thickness = -1
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

    def put_annotation_on_image(self, image, annotation: Annotation):
        """Puts the specified annotation on the image."""
        if annotation.type == AnnotationType.CIRCLE:
            cv2.circle(image, annotation.coordinates, 5, annotation.color, -1)
        elif annotation.type == AnnotationType.CIRCLE_AND_LABEL:
            self.put_circle_and_text_on_image(image, annotation.labels, annotation.coordinates, annotation.color)
        elif annotation.type == AnnotationType.RECTANGLE:
            x, y, w, h = annotation.coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), annotation.color, 2)
        elif annotation.type in {AnnotationType.POINTS, AnnotationType.POINT}:
            points = annotation.coordinates if isinstance(annotation.coordinates, list) else [annotation.coordinates]
            for point in points:
                cv2.circle(image, point, 5, annotation.color, -1)
        elif annotation.type == AnnotationType.POINTS_AND_LABELS:
            for point, label in zip(annotation.coordinates, annotation.labels):
                cv2.circle(image, point, 5, annotation.color, -1)
                cv2.putText(image, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, annotation.color, 1)
        elif annotation.type == AnnotationType.PITCH_YAW_ROLL:
            p, y, r = annotation.orientation
            image = self.draw_orientation(image, y, p, r)


def main():
    img_input = "sample_image.jpg"
    uiih = UniversalImageInputHandler(img_input, debug=False)

    debug_folder_path = "./"
    visdebugger = VisualDebugger(tag="HS", debug_folder_path=debug_folder_path, active=True, output='return')
    annotations = [Annotation(type=AnnotationType.PITCH_YAW_ROLL, coordinates=(320, 240), orientation=(0.5, 0.5, 0.5))]

    img = visdebugger.visual_debug(uiih.img, annotations, process_step="cropped_headselection_mask", condition="", mask=False)

    if img is not None:
        cv2.imshow("Debug Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
