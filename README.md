
# Visual Debugger for Image Processing

The Visual Debugger module provides tools for visually debugging image processing workflows, allowing annotations like points, labels, rectangles, and more on images. It supports a variety of annotation types and includes functionality for merging multiple images into a composite image for comprehensive review.

## Features

- **Multiple Annotation Types**: Support for a variety of annotations such as points, labeled points, rectangles, circles, and orientation vectors based on pitch, yaw, and roll.
- **Image Concatenation**: Capability to concatenate multiple debugged images into a single composite image, facilitating easier visualization of sequential image processing steps.
- **Dynamic Image Handling**: Handles a wide range of image inputs including file paths, in-memory image arrays, base64 encoded images, and images from web links, integrating seamlessly with OpenCV.
- **Customizable Debugging**: Debugging can be turned on or off, and the module supports generating merged debug images for a sequence of operations.
- **Orientation Drawing**: Special functionality for visualizing orientation in 3D space, projecting pitch, yaw, and roll onto a 2D image plane.
 **Automatic Naming**: Automatically handles the naming of output images based on a customizable tag, the sequence number of the operation, and specific conditions of the debugging process. This ensures a clear and organized structure for saved images, making it easy to track and review the progress of image processing.
## Installation

To use this module in your projects, install the pypi package by:

```bash
pip install visual_debugger
```


## Usage

Here is a basic example of how to use the VisualDebugger class within your image processing workflow:

```python
#import VisualDebugger and annotations
from visual_debugger import VisualDebugger, Annotation, AnnotationType
# (not necessary) import UniversalImageInputHandler to read the image file
from image_input_handler import  UniversalImageInputHandler

# Initialize the debugger
vd = VisualDebugger(tag="Example", debug_folder_path="./visualdebug",  active=True)

# Load your image  (this should be replaced with your actual image path or np.array image)
image_path = "path/to/your/image.jpg" 
uih = UniversalImageInputHandler(image_path, debug=False)
COMPATIBLE, img = uih.COMPATIBLE, uih.img

# Create annotations you want to show on the image
annotations = [
    Annotation(type=AnnotationType.RECTANGLE, coordinates=(10, 10, 50, 50)),
    Annotation(type=AnnotationType.POINT_AND_LABEL, coordinates=(100, 100), labels="Center Point")
]

# Process and debug the image.
# This will use "tag" and "process_step" variables to create a file name 
# And save the image into "debug_folder_path" 
vd.visual_debug(img, annotations=annotations, process_step="initial_check")

# Generate a merged image of all processed steps
vd.cook_merged_img()
```

## Documentation

For more details on the methods and their parameters, please refer to the inline documentation within the module files.

## Contributing

Contributions to enhance the functionality or improve the documentation are welcome. Please follow the standard GitHub pull request process to submit your contributions.


---
