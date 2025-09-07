"""
Annotation processor using visitor pattern to render type-specific annotations.
"""

import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .annotations import (
        BaseAnnotation, PointAnnotation, LabeledPointAnnotation,
        PointsAnnotation, LabeledPointsAnnotation, CircleAnnotation,
        LabeledCircleAnnotation, RectangleAnnotation, LineAnnotation,
        LabeledLineAnnotation, PolylineAnnotation, TextAnnotation,
        BoundingBoxAnnotation, MaskAnnotation, OrientationAnnotation,
        InfoPanelAnnotation
    )


class AnnotationProcessor:
    """Processes type-specific annotations using visitor pattern"""
    
    def __init__(self):
        """Initialize the annotation processor"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def render(self, image: np.ndarray, annotation: 'BaseAnnotation') -> np.ndarray:
        """
        Main entry point - uses visitor pattern to dispatch to correct method.
        
        Args:
            image: The image to annotate
            annotation: The annotation to render
            
        Returns:
            The annotated image
        """
        return annotation.accept(self)
    
    # =============================================================================
    # Point Renderers
    # =============================================================================
    
    def render_point(self, annotation: 'PointAnnotation') -> None:
        """Render a single point"""
        cv2.circle(self.image, annotation.position, annotation.size, 
                  annotation.color, -1)
    
    def render_labeled_point(self, annotation: 'LabeledPointAnnotation') -> None:
        """Render a point with a label"""
        # Draw point
        cv2.circle(self.image, annotation.position, annotation.size, 
                  annotation.color, -1)
        
        # Draw label
        label_pos = (annotation.position[0] + annotation.size + 2, 
                    annotation.position[1] - annotation.size)
        cv2.putText(self.image, annotation.label, label_pos, self.font,
                   annotation.font_scale, annotation.color, 
                   annotation.font_thickness, cv2.LINE_AA)
    
    def render_points(self, annotation: 'PointsAnnotation') -> None:
        """Render multiple points"""
        for point in annotation.points:
            cv2.circle(self.image, point, annotation.size, 
                      annotation.color, -1)
    
    def render_labeled_points(self, annotation: 'LabeledPointsAnnotation') -> None:
        """Render multiple points with labels"""
        for point, label in zip(annotation.points, annotation.labels):
            # Draw point
            cv2.circle(self.image, point, annotation.size, 
                      annotation.color, -1)
            
            # Draw label
            label_pos = (point[0] + annotation.size + 2, 
                        point[1] - annotation.size)
            cv2.putText(self.image, label, label_pos, self.font,
                       annotation.font_scale, annotation.color,
                       annotation.font_thickness, cv2.LINE_AA)
    
    # =============================================================================
    # Shape Renderers
    # =============================================================================
    
    def render_circle(self, annotation: 'CircleAnnotation') -> None:
        """Render a circle"""
        thickness = -1 if annotation.filled else annotation.thickness
        cv2.circle(self.image, annotation.center, annotation.radius,
                  annotation.color, thickness)
    
    def render_labeled_circle(self, annotation: 'LabeledCircleAnnotation') -> None:
        """Render a circle with a label"""
        # Draw circle
        thickness = -1 if annotation.filled else annotation.thickness
        cv2.circle(self.image, annotation.center, annotation.radius,
                  annotation.color, thickness)
        
        # Draw label
        label_pos = (annotation.center[0] + annotation.radius + 5,
                    annotation.center[1])
        cv2.putText(self.image, annotation.label, label_pos, self.font,
                   annotation.font_scale, annotation.color,
                   annotation.font_thickness, cv2.LINE_AA)
    
    def render_rectangle(self, annotation: 'RectangleAnnotation') -> None:
        """Render a rectangle"""
        thickness = -1 if annotation.filled else annotation.thickness
        cv2.rectangle(self.image, annotation.top_left, annotation.bottom_right,
                     annotation.color, thickness)
    
    def render_alpha_rectangle(self, annotation: 'AlphaRectangleAnnotation') -> None:
        """Render a rectangle with alpha transparency"""
        # Create an overlay
        overlay = self.image.copy()
        
        # Draw the filled rectangle on the overlay
        cv2.rectangle(overlay, annotation.top_left, annotation.bottom_right,
                     annotation.color, -1)
        
        # Blend the overlay with the original image using alpha
        cv2.addWeighted(overlay, annotation.alpha, self.image, 1 - annotation.alpha, 0, self.image)
    
    # =============================================================================
    # Line Renderers
    # =============================================================================
    
    def render_line(self, annotation: 'LineAnnotation') -> None:
        """Render a line"""
        if annotation.arrow:
            cv2.arrowedLine(self.image, annotation.start, annotation.end,
                           annotation.color, annotation.thickness)
        else:
            cv2.line(self.image, annotation.start, annotation.end,
                    annotation.color, annotation.thickness)
    
    def render_labeled_line(self, annotation: 'LabeledLineAnnotation') -> None:
        """Render a line with a label"""
        # Draw line
        if annotation.arrow:
            cv2.arrowedLine(self.image, annotation.start, annotation.end,
                           annotation.color, annotation.thickness)
        else:
            cv2.line(self.image, annotation.start, annotation.end,
                    annotation.color, annotation.thickness)
        
        # Draw label at midpoint
        cv2.putText(self.image, annotation.label, annotation.midpoint,
                   self.font, annotation.font_scale, annotation.color,
                   annotation.font_thickness, cv2.LINE_AA)
    
    def render_polyline(self, annotation: 'PolylineAnnotation') -> None:
        """Render a polyline"""
        points = np.array(annotation.points, dtype=np.int32)
        
        if annotation.closed:
            cv2.polylines(self.image, [points], True, annotation.color,
                         annotation.thickness)
        else:
            for i in range(len(points) - 1):
                cv2.line(self.image, tuple(points[i]), tuple(points[i + 1]),
                        annotation.color, annotation.thickness)
    
    # =============================================================================
    # Text Renderer
    # =============================================================================
    
    def render_text(self, annotation: 'TextAnnotation') -> None:
        """Render text with optional background"""
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            annotation.text, self.font, annotation.font_scale,
            annotation.font_thickness
        )
        
        # Draw background if specified
        if annotation.background_color is not None:
            padding = annotation.background_padding
            top_left = (annotation.position[0] - padding,
                       annotation.position[1] - text_height - padding)
            bottom_right = (annotation.position[0] + text_width + padding,
                           annotation.position[1] + baseline + padding)
            cv2.rectangle(self.image, top_left, bottom_right,
                         annotation.background_color, -1)
        
        # Draw text
        cv2.putText(self.image, annotation.text, annotation.position,
                   self.font, annotation.font_scale, annotation.color,
                   annotation.font_thickness, cv2.LINE_AA)
    
    # =============================================================================
    # Complex Renderers
    # =============================================================================
    
    def render_bounding_box(self, annotation: 'BoundingBoxAnnotation') -> None:
        """Render a bounding box with optional label"""
        # Draw rectangle
        cv2.rectangle(self.image, annotation.top_left, annotation.bottom_right,
                     annotation.color, annotation.thickness)
        
        # Draw label if present
        label_text = annotation.get_label_text()
        if label_text:
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, self.font, annotation.font_scale,
                annotation.font_thickness
            )
            
            # Draw background for label if requested
            if annotation.show_label_background:
                label_bg_top = (annotation.top_left[0],
                               annotation.top_left[1] - text_height - 4)
                label_bg_bottom = (annotation.top_left[0] + text_width + 4,
                                  annotation.top_left[1])
                cv2.rectangle(self.image, label_bg_top, label_bg_bottom,
                             annotation.color, -1)
                
                # White text on colored background
                text_color = (255, 255, 255)
            else:
                text_color = annotation.color
            
            # Draw text
            text_pos = (annotation.top_left[0] + 2,
                       annotation.top_left[1] - 4)
            cv2.putText(self.image, label_text, text_pos, self.font,
                       annotation.font_scale, text_color,
                       annotation.font_thickness, cv2.LINE_AA)
    
    def render_mask(self, annotation: 'MaskAnnotation') -> None:
        """Render a segmentation mask"""
        mask = annotation.mask
        
        # Create color mask
        if annotation.colormap == 'random':
            # Random colors for each class
            num_classes = np.max(mask) + 1
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            color_mapping = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
            
            for class_id in range(num_classes):
                color_mask[mask == class_id] = color_mapping[class_id]
        
        elif annotation.colormap in ['jet', 'hot', 'cool', 'hsv']:
            # Use OpenCV colormap
            normalized_mask = (mask / np.max(mask) * 255).astype(np.uint8)
            colormap = getattr(cv2, f'COLORMAP_{annotation.colormap.upper()}', cv2.COLORMAP_JET)
            color_mask = cv2.applyColorMap(normalized_mask, colormap)
        
        else:
            # Default: different color for each class
            num_classes = np.max(mask) + 1
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            
            # Predefined colors
            default_colors = [
                [0, 0, 0],        # Background - black
                [0, 153, 255],    # Class 1 - orange
                [102, 255, 153],  # Class 2 - light green
                [0, 204, 153],    # Class 3 - teal
                [255, 255, 102],  # Class 4 - light yellow
                [255, 255, 204],  # Class 5 - pale yellow
                [255, 153, 0],    # Class 6 - orange
                [255, 102, 255],  # Class 7 - pink
                [102, 0, 51],     # Class 8 - dark purple
                [255, 204, 255],  # Class 9 - light pink
                [255, 0, 102]     # Class 10 - red-pink
            ]
            
            for class_id in range(min(num_classes, len(default_colors))):
                color_mask[mask == class_id] = default_colors[class_id]
            
            # Random colors for classes beyond predefined
            if num_classes > len(default_colors):
                for class_id in range(len(default_colors), num_classes):
                    color = np.random.randint(0, 255, 3)
                    color_mask[mask == class_id] = color
        
        # Blend with original image
        cv2.addWeighted(self.image, 1 - annotation.alpha, color_mask, 
                       annotation.alpha, 0, self.image)
    
    def render_orientation(self, annotation: 'OrientationAnnotation') -> None:
        """Render 3D orientation axes"""
        # Convert angles to radians
        pitch = np.deg2rad(annotation.pitch)
        yaw = np.deg2rad(annotation.yaw)
        roll = np.deg2rad(annotation.roll)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(pitch), -np.sin(pitch)],
                      [0, np.sin(pitch), np.cos(pitch)]])
        
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                      [np.sin(roll), np.cos(roll), 0],
                      [0, 0, 1]])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        
        # Define axis points
        size = annotation.size
        axis_points = np.array([[0, 0, 0],
                               [size, 0, 0],    # X axis
                               [0, size, 0],    # Y axis
                               [0, 0, size]])   # Z axis
        
        # Transform points
        transformed = axis_points @ R.T
        
        # Project to 2D and add position offset
        origin = annotation.position
        points_2d = transformed[:, :2] + np.array(origin)
        points_2d = points_2d.astype(int)
        
        # Draw axes
        cv2.line(self.image, tuple(points_2d[0]), tuple(points_2d[1]),
                annotation.x_color, 3)  # X axis - red
        cv2.line(self.image, tuple(points_2d[0]), tuple(points_2d[2]),
                annotation.y_color, 3)  # Y axis - green
        cv2.line(self.image, tuple(points_2d[0]), tuple(points_2d[3]),
                annotation.z_color, 3)  # Z axis - blue
        
        # Add axis labels
        cv2.putText(self.image, "X", tuple(points_2d[1]), self.font,
                   0.5, annotation.x_color, 2, cv2.LINE_AA)
        cv2.putText(self.image, "Y", tuple(points_2d[2]), self.font,
                   0.5, annotation.y_color, 2, cv2.LINE_AA)
        cv2.putText(self.image, "Z", tuple(points_2d[3]), self.font,
                   0.5, annotation.z_color, 2, cv2.LINE_AA)
    # =============================================================================
    # Process method with image handling
    # =============================================================================
    
    def process(self, image: np.ndarray, annotation: 'BaseAnnotation') -> np.ndarray:
        """
        Process an annotation on an image.
        
        Args:
            image: The image to annotate
            annotation: The annotation to render
            
        Returns:
            The image (modified in place)
        """
        self.image = image
        annotation.accept(self)
        return self.image