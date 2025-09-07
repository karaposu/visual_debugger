"""
Image composition and layout management for Visual Debugger.
Handles concatenation, grid layouts, and image arrangement.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum


class LayoutDirection(Enum):
    """Direction for image layout"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"


@dataclass
class CompositionStyle:
    """Visual style configuration for image composition"""
    border_thickness: int = 5
    border_color: Tuple[int, int, int] = (255, 255, 255)
    vertical_spacing: int = 20
    horizontal_spacing: int = 20
    label_font_size: float = 0.8
    label_font_thickness: int = 2
    label_color: Optional[Tuple[int, int, int]] = None  # None means use border_color
    background_color: Tuple[int, int, int] = (0, 0, 0)
    padding: int = 10
    show_labels: bool = True
    show_borders: bool = True
    show_separators: bool = True
    separator_color: Tuple[int, int, int] = (128, 128, 128)
    separator_thickness: int = 2


@dataclass
class ImageEntry:
    """Container for an image with metadata"""
    image: np.ndarray
    name: str = "image"
    stage: Optional[str] = None
    
    @property
    def label(self) -> str:
        """Get the label for this image"""
        if self.stage:
            return f"{self.name}_{self.stage}"
        return self.name


class ImageCompositor:
    """Handles image composition and layout for debugging visualizations"""
    
    def __init__(self, style: Optional[CompositionStyle] = None):
        """
        Initialize the ImageCompositor.
        
        Args:
            style: Style configuration for composition
        """
        self.style = style or CompositionStyle()
        
    def create_grid(self, images: List[Union[np.ndarray, ImageEntry]], 
                   cols: Optional[int] = None,
                   labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Create a grid layout of images.
        
        Args:
            images: List of images or ImageEntry objects
            cols: Number of columns (auto-calculated if None)
            labels: Optional labels for images (if images are np.ndarray)
            
        Returns:
            Composed grid image
        """
        # Convert to ImageEntry objects
        entries = self._prepare_entries(images, labels)
        
        if not entries:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Calculate grid dimensions
        num_images = len(entries)
        if cols is None:
            cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Process images (add borders/labels)
        processed = [self._process_image(entry) for entry in entries]
        
        # Calculate cell dimensions
        max_height = max(img.shape[0] for img in processed)
        max_width = max(img.shape[1] for img in processed)
        
        # Calculate total dimensions
        total_width = cols * max_width + (cols - 1) * self.style.horizontal_spacing + 2 * self.style.padding
        total_height = rows * max_height + (rows - 1) * self.style.vertical_spacing + 2 * self.style.padding
        
        # Create output image
        output = np.full((total_height, total_width, 3), 
                        self.style.background_color, dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(processed):
            row = idx // cols
            col = idx % cols
            
            x = self.style.padding + col * (max_width + self.style.horizontal_spacing)
            y = self.style.padding + row * (max_height + self.style.vertical_spacing)
            
            # Center image in cell
            x_offset = (max_width - img.shape[1]) // 2
            y_offset = (max_height - img.shape[0]) // 2
            
            output[y + y_offset:y + y_offset + img.shape[0],
                  x + x_offset:x + x_offset + img.shape[1]] = img
        
        return output
    
    def create_comparison(self, before: np.ndarray, after: np.ndarray,
                         before_label: str = "Before", 
                         after_label: str = "After") -> np.ndarray:
        """
        Create a side-by-side comparison of two images.
        
        Args:
            before: First image
            after: Second image
            before_label: Label for first image
            after_label: Label for second image
            
        Returns:
            Side-by-side comparison image
        """
        entries = [
            ImageEntry(before, before_label),
            ImageEntry(after, after_label)
        ]
        return self.concatenate(entries, direction=LayoutDirection.HORIZONTAL)
    
    def concatenate(self, images: List[Union[np.ndarray, ImageEntry]], 
                   direction: LayoutDirection = LayoutDirection.HORIZONTAL,
                   labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Concatenate images in specified direction.
        
        Args:
            images: List of images or ImageEntry objects
            direction: Direction to concatenate
            labels: Optional labels for images
            
        Returns:
            Concatenated image
        """
        # Convert to ImageEntry objects
        entries = self._prepare_entries(images, labels)
        
        if not entries:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        if direction == LayoutDirection.GRID:
            return self.create_grid(entries)
        
        # Group images by name for better organization
        grouped = self._group_by_name(entries)
        
        if direction == LayoutDirection.HORIZONTAL:
            return self._concat_horizontal(grouped)
        else:
            return self._concat_vertical(grouped)
    
    def _prepare_entries(self, images: List[Union[np.ndarray, ImageEntry]], 
                        labels: Optional[List[str]] = None) -> List[ImageEntry]:
        """Convert mixed input to ImageEntry objects"""
        entries = []
        for i, img in enumerate(images):
            if isinstance(img, ImageEntry):
                entries.append(img)
            elif isinstance(img, np.ndarray):
                label = labels[i] if labels and i < len(labels) else f"image_{i}"
                entries.append(ImageEntry(img, label))
        return entries
    
    def _group_by_name(self, entries: List[ImageEntry]) -> Dict[str, List[ImageEntry]]:
        """Group images by their name"""
        grouped = {}
        for entry in entries:
            if entry.name not in grouped:
                grouped[entry.name] = []
            grouped[entry.name].append(entry)
        return grouped
    
    def _process_image(self, entry: ImageEntry) -> np.ndarray:
        """Process a single image (add borders, labels, etc.)"""
        img = entry.image.copy()
        
        # Ensure image is BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Add border if requested
        if self.style.show_borders:
            img = cv2.copyMakeBorder(
                img, 
                self.style.border_thickness,
                self.style.border_thickness,
                self.style.border_thickness,
                self.style.border_thickness,
                cv2.BORDER_CONSTANT,
                value=self.style.border_color
            )
        
        # Add label if requested
        if self.style.show_labels and entry.label:
            # Add space for label
            label_height = 30
            labeled = np.full(
                (img.shape[0] + label_height, img.shape[1], 3),
                self.style.background_color,
                dtype=np.uint8
            )
            labeled[label_height:] = img
            
            # Draw label
            label_color = self.style.label_color or self.style.border_color
            cv2.putText(
                labeled, entry.label,
                (5, label_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.label_font_size,
                label_color,
                self.style.label_font_thickness,
                cv2.LINE_AA
            )
            img = labeled
        
        return img
    
    def _concat_horizontal(self, grouped: Dict[str, List[ImageEntry]]) -> np.ndarray:
        """Concatenate groups horizontally"""
        group_images = []
        
        for name, entries in grouped.items():
            # Process all images in group
            processed = [self._process_image(entry) for entry in entries]
            
            # Stack vertically within group
            if len(processed) > 1:
                max_width = max(img.shape[1] for img in processed)
                padded = []
                for img in processed:
                    if img.shape[1] < max_width:
                        pad_left = (max_width - img.shape[1]) // 2
                        pad_right = max_width - img.shape[1] - pad_left
                        img = cv2.copyMakeBorder(
                            img, 0, 0, pad_left, pad_right,
                            cv2.BORDER_CONSTANT, value=self.style.background_color
                        )
                    padded.append(img)
                
                # Add vertical spacing
                spaced = []
                for i, img in enumerate(padded):
                    spaced.append(img)
                    if i < len(padded) - 1:
                        spacer = np.full(
                            (self.style.vertical_spacing, img.shape[1], 3),
                            self.style.background_color, dtype=np.uint8
                        )
                        spaced.append(spacer)
                
                group_img = np.vstack(spaced)
            else:
                group_img = processed[0]
            
            group_images.append(group_img)
        
        # Now concatenate groups horizontally
        if not group_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Pad to same height
        max_height = max(img.shape[0] for img in group_images)
        padded_groups = []
        for img in group_images:
            if img.shape[0] < max_height:
                pad_top = (max_height - img.shape[0]) // 2
                pad_bottom = max_height - img.shape[0] - pad_top
                img = cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, 0, 0,
                    cv2.BORDER_CONSTANT, value=self.style.background_color
                )
            padded_groups.append(img)
        
        # Add horizontal spacing and separators
        final_images = []
        for i, img in enumerate(padded_groups):
            final_images.append(img)
            if i < len(padded_groups) - 1:
                # Add spacing
                spacer = np.full(
                    (img.shape[0], self.style.horizontal_spacing, 3),
                    self.style.background_color, dtype=np.uint8
                )
                
                # Add separator line in middle of spacer if requested
                if self.style.show_separators:
                    cv2.line(
                        spacer,
                        (self.style.horizontal_spacing // 2, 0),
                        (self.style.horizontal_spacing // 2, spacer.shape[0]),
                        self.style.separator_color,
                        self.style.separator_thickness
                    )
                
                final_images.append(spacer)
        
        # Add padding
        result = np.hstack(final_images)
        result = cv2.copyMakeBorder(
            result,
            self.style.padding, self.style.padding,
            self.style.padding, self.style.padding,
            cv2.BORDER_CONSTANT, value=self.style.background_color
        )
        
        return result
    
    def _concat_vertical(self, grouped: Dict[str, List[ImageEntry]]) -> np.ndarray:
        """Concatenate groups vertically"""
        group_images = []
        
        for name, entries in grouped.items():
            # Process all images in group
            processed = [self._process_image(entry) for entry in entries]
            
            # Stack horizontally within group
            if len(processed) > 1:
                max_height = max(img.shape[0] for img in processed)
                padded = []
                for img in processed:
                    if img.shape[0] < max_height:
                        pad_top = (max_height - img.shape[0]) // 2
                        pad_bottom = max_height - img.shape[0] - pad_top
                        img = cv2.copyMakeBorder(
                            img, pad_top, pad_bottom, 0, 0,
                            cv2.BORDER_CONSTANT, value=self.style.background_color
                        )
                    padded.append(img)
                
                # Add horizontal spacing
                spaced = []
                for i, img in enumerate(padded):
                    spaced.append(img)
                    if i < len(padded) - 1:
                        spacer = np.full(
                            (img.shape[0], self.style.horizontal_spacing, 3),
                            self.style.background_color, dtype=np.uint8
                        )
                        spaced.append(spacer)
                
                group_img = np.hstack(spaced)
            else:
                group_img = processed[0]
            
            group_images.append(group_img)
        
        # Now concatenate groups vertically
        if not group_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Pad to same width
        max_width = max(img.shape[1] for img in group_images)
        padded_groups = []
        for img in group_images:
            if img.shape[1] < max_width:
                pad_left = (max_width - img.shape[1]) // 2
                pad_right = max_width - img.shape[1] - pad_left
                img = cv2.copyMakeBorder(
                    img, 0, 0, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=self.style.background_color
                )
            padded_groups.append(img)
        
        # Add vertical spacing and separators
        final_images = []
        for i, img in enumerate(padded_groups):
            final_images.append(img)
            if i < len(padded_groups) - 1:
                # Add spacing
                spacer = np.full(
                    (self.style.vertical_spacing, img.shape[1], 3),
                    self.style.background_color, dtype=np.uint8
                )
                
                # Add separator line in middle of spacer if requested
                if self.style.show_separators:
                    cv2.line(
                        spacer,
                        (0, self.style.vertical_spacing // 2),
                        (spacer.shape[1], self.style.vertical_spacing // 2),
                        self.style.separator_color,
                        self.style.separator_thickness
                    )
                
                final_images.append(spacer)
        
        # Add padding
        result = np.vstack(final_images)
        result = cv2.copyMakeBorder(
            result,
            self.style.padding, self.style.padding,
            self.style.padding, self.style.padding,
            cv2.BORDER_CONSTANT, value=self.style.background_color
        )
        
        return result