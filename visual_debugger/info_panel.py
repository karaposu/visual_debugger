import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any
from enum import Enum


class PanelPosition(Enum):
    """Enumeration of possible panel positions on the image"""
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    TOP_CENTER = "top-center"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_CENTER = "bottom-center"
    CENTER_LEFT = "center-left"
    CENTER_RIGHT = "center-right"
    CENTER = "center"


@dataclass
class PanelStyle:
    """Visual style configuration for the info panel"""
    background_color: Tuple[int, int, int] = (0, 0, 0)
    background_alpha: float = 0.7
    text_color: Tuple[int, int, int] = (255, 255, 255)
    title_color: Tuple[int, int, int] = (255, 255, 100)
    border_color: Optional[Tuple[int, int, int]] = (255, 255, 255)
    border_thickness: int = 2
    padding: int = 15
    line_spacing: int = 5
    font_scale: float = 0.5
    font_thickness: int = 1
    min_width: int = 200
    show_background: bool = True
    font: int = cv2.FONT_HERSHEY_SIMPLEX


class InfoPanel:
    """A clean, organized panel for displaying information on images"""
    
    def __init__(self, 
                 position: Union[str, PanelPosition] = PanelPosition.TOP_LEFT,
                 title: Optional[str] = None,
                 style: Optional[PanelStyle] = None,
                 margin: int = 10):
        """
        Initialize an InfoPanel.
        
        Args:
            position: Where to place the panel on the image
            title: Optional title for the panel
            style: Visual style configuration
            margin: Distance from image edges
        """
        self.position = PanelPosition(position) if isinstance(position, str) else position
        self.title = title
        self.style = style or PanelStyle()
        self.margin = margin
        self._entries: List[Tuple[Optional[str], Any]] = []
        
    def add(self, key: Optional[str] = None, value: Any = None):
        """
        Add a key-value pair or just a value to the panel.
        
        Args:
            key: Optional key/label for the value
            value: The value to display
            
        Returns:
            self for method chaining
        """
        if key is None and value is None:
            self._entries.append((None, ""))  # Empty line
        elif key is None:
            self._entries.append((None, str(value)))  # Just value
        else:
            self._entries.append((str(key), str(value)))  # Key-value pair
        return self
    
    def add_line(self, text: str = ""):
        """
        Add a simple text line to the panel.
        
        Args:
            text: Text to display
            
        Returns:
            self for method chaining
        """
        self._entries.append((None, text))
        return self
    
    def add_separator(self, char: str = "─", width: int = 20):
        """
        Add a separator line to the panel.
        
        Args:
            char: Character to use for the separator
            width: Width of the separator in characters
            
        Returns:
            self for method chaining
        """
        self._entries.append((None, char * width))
        return self
    
    def add_metrics(self, metrics: dict, format_spec: str = ".2f"):
        """
        Add multiple metrics with automatic formatting.
        
        Args:
            metrics: Dictionary of metric names and values
            format_spec: Format specification for float values
            
        Returns:
            self for method chaining
        """
        for key, value in metrics.items():
            if isinstance(value, float):
                self.add(key, f"{value:{format_spec}}")
            else:
                self.add(key, value)
        return self
    
    def add_table(self, headers: List[str], rows: List[List[Any]]):
        """
        Add a formatted table to the panel.
        
        Args:
            headers: Table column headers
            rows: Table data rows
            
        Returns:
            self for method chaining
        """
        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:len(col_widths)]):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add headers
        header_line = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        self.add_line(header_line)
        self.add_line("─" * len(header_line))
        
        # Add rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(w) 
                                 for cell, w in zip(row[:len(col_widths)], col_widths))
            self.add_line(row_line)
        
        return self
    
    def add_progress(self, label: str, value: float, max_value: float = 1.0, width: int = 15):
        """
        Add a text-based progress bar to the panel.
        
        Args:
            label: Label for the progress bar
            value: Current value
            max_value: Maximum value
            width: Width of the progress bar in characters
            
        Returns:
            self for method chaining
        """
        percentage = min(1.0, max(0.0, value / max_value))
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        self.add(label, f"{bar} {percentage:.1%}")
        return self
    
    def clear(self):
        """
        Clear all entries from the panel.
        
        Returns:
            self for method chaining
        """
        self._entries.clear()
        return self
    
    def calculate_dimensions(self):
        """
        Calculate the required panel dimensions based on content.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        font = self.style.font
        max_width = self.style.min_width
        total_height = self.style.padding * 2
        
        # Add title dimensions if present
        if self.title:
            title_scale = self.style.font_scale * 1.2
            (title_w, title_h), _ = cv2.getTextSize(
                self.title, font, title_scale, 
                self.style.font_thickness + 1)
            max_width = max(max_width, title_w + self.style.padding * 2)
            total_height += title_h + self.style.line_spacing * 3  # Extra space for separator
        
        # Calculate dimensions for each entry
        for key, value in self._entries:
            if key:
                text = f"{key}: {value}"
            else:
                text = value
                
            (text_w, text_h), _ = cv2.getTextSize(
                text, font, self.style.font_scale, 
                self.style.font_thickness)
            
            max_width = max(max_width, text_w + self.style.padding * 2)
            total_height += text_h + self.style.line_spacing
        
        return max_width, total_height
    
    def calculate_position(self, image_shape, panel_width, panel_height):
        """
        Calculate the top-left corner position based on anchor.
        
        Args:
            image_shape: Shape of the target image
            panel_width: Width of the panel
            panel_height: Height of the panel
            
        Returns:
            Tuple of (x, y) coordinates for top-left corner
        """
        img_h, img_w = image_shape[:2]
        
        positions = {
            PanelPosition.TOP_LEFT: (self.margin, self.margin),
            PanelPosition.TOP_RIGHT: (img_w - panel_width - self.margin, self.margin),
            PanelPosition.TOP_CENTER: ((img_w - panel_width) // 2, self.margin),
            PanelPosition.BOTTOM_LEFT: (self.margin, img_h - panel_height - self.margin),
            PanelPosition.BOTTOM_RIGHT: (img_w - panel_width - self.margin, 
                                        img_h - panel_height - self.margin),
            PanelPosition.BOTTOM_CENTER: ((img_w - panel_width) // 2, 
                                          img_h - panel_height - self.margin),
            PanelPosition.CENTER_LEFT: (self.margin, (img_h - panel_height) // 2),
            PanelPosition.CENTER_RIGHT: (img_w - panel_width - self.margin, 
                                         (img_h - panel_height) // 2),
            PanelPosition.CENTER: ((img_w - panel_width) // 2, 
                                   (img_h - panel_height) // 2),
        }
        
        return positions.get(self.position, (self.margin, self.margin))
    
