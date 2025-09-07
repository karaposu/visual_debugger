"""
Image processor that uses the new type-specific annotation system.
"""

import numpy as np
import cv2
from typing import List, Union
from .annotations import BaseAnnotation
from .annotation_processor import AnnotationProcessor


class ImageProcessor:
    """
    Image processor that handles type-specific annotations.
    Uses the visitor pattern through AnnotationProcessor for clean separation.
    """
    
    def __init__(self):
        """Initialize the image processor"""
        self.processor = AnnotationProcessor()
    
    def put_annotation_on_image(self, image: np.ndarray, 
                                annotation: BaseAnnotation) -> np.ndarray:
        """
        Process an annotation on an image.
        
        Args:
            image: The image to annotate
            annotation: A type-specific annotation (subclass of BaseAnnotation)
            
        Returns:
            The annotated image (modified in place)
        """
        return self.processor.process(image, annotation)
    
    def put_annotations_on_image(self, image: np.ndarray, 
                                 annotations: List[BaseAnnotation]) -> np.ndarray:
        """
        Process multiple annotations on an image.
        
        Args:
            image: The image to annotate
            annotations: List of type-specific annotations
            
        Returns:
            The annotated image
        """
        for annotation in annotations:
            self.put_annotation_on_image(image, annotation)
        return image
    
    def pad_images_to_match_height(self, img1: np.ndarray, img2: np.ndarray):
        """
        Pads images to match their height.
        Utility method kept for compatibility with existing code.
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        if h1 > h2:
            pad_amount = h1 - h2
            pad_top = pad_amount // 2
            pad_bottom = pad_amount - pad_top
            img2 = cv2.copyMakeBorder(img2, pad_top, pad_bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif h2 > h1:
            pad_amount = h2 - h1
            pad_top = pad_amount // 2
            pad_bottom = pad_amount - pad_top
            img1 = cv2.copyMakeBorder(img1, pad_top, pad_bottom, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return img1, img2