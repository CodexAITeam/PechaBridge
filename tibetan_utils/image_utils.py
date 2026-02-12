"""
Image processing utilities for the TibetanOCR project.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from typing import Tuple, List, Union, Dict, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Image as a numpy array
    """
    return cv2.imread(image_path)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a cv2 image (numpy array).
    
    Args:
        pil_image: PIL Image
        
    Returns:
        np.ndarray: cv2 image
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert a cv2 image (numpy array) to a PIL Image.
    
    Args:
        cv2_image: cv2 image
        
    Returns:
        Image.Image: PIL Image
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image.
    
    Args:
        image: Image to resize
        size: Target size (width, height)
        
    Returns:
        np.ndarray: Resized image
    """
    return cv2.resize(image, size)


def extract_text_region(image: np.ndarray, box: List[float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a text region from an image based on a bounding box.
    
    Args:
        image: Image as a numpy array
        box: Bounding box [x, y, w, h, conf, class] (normalized coordinates)
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int, int, int]]: Extracted region and absolute coordinates (x_min, y_min, x_max, y_max)
    """
    # Extract coordinates
    x, y, w, h = box[:4]
    
    # Convert relative coordinates to absolute pixel coordinates
    height, width = image.shape[:2]
    x_min = int((x - w/2) * width)
    y_min = int((y - h/2) * height)
    x_max = int((x + w/2) * width)
    y_max = int((y + h/2) * height)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    # Extract the region
    region = image[y_min:y_max, x_min:x_max]
    
    return region, (x_min, y_min, x_max, y_max)


def draw_boxes(image: np.ndarray, boxes: List[List[float]], labels: List[str] = None, 
               colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Image to draw on
        boxes: List of bounding boxes [x, y, w, h, conf, class]
        labels: List of labels for each box
        colors: List of colors for each box
        
    Returns:
        np.ndarray: Image with boxes drawn
    """
    result = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)  # Default to green
    
    for i, box in enumerate(boxes):
        x, y, w, h = box[:4]
        
        # Convert relative coordinates to absolute pixel coordinates
        height, width = image.shape[:2]
        x_min = int((x - w/2) * width)
        y_min = int((y - h/2) * height)
        x_max = int((x + w/2) * width)
        y_max = int((y + h/2) * height)
        
        # Draw box
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), colors[i], 2)
        
        # Draw label if provided
        if labels is not None and i < len(labels):
            cv2.putText(result, labels[i], (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    
    return result


def save_image(image: np.ndarray, filepath: str) -> None:
    """
    Save an image to a file.
    
    Args:
        image: Image to save
        filepath: Path to save the image
    """
    cv2.imwrite(filepath, image)


def apply_augmentation(image: np.ndarray, augmentation_type: str, **kwargs) -> np.ndarray:
    """
    Apply augmentation to an image.
    
    Args:
        image: Image to augment
        augmentation_type: Type of augmentation ('rotate' or 'noise')
        **kwargs: Additional arguments for the augmentation
        
    Returns:
        np.ndarray: Augmented image
    """
    result = image.copy()
    
    if augmentation_type == 'rotate':
        angle = kwargs.get('angle', np.random.randint(-15, 15))
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    elif augmentation_type == 'noise':
        noise_level = kwargs.get('noise_level', 25)
        noise = np.random.uniform(-noise_level, noise_level, image.shape).astype(np.float32)
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return result


def normalize_box(box: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> List[float]:
    """
    Normalize a bounding box to YOLO format.
    
    Args:
        box: Bounding box (x_min, y_min, x_max, y_max)
        image_size: Image size (width, height)
        
    Returns:
        List[float]: Normalized box [x, y, w, h]
    """
    x_min, y_min, x_max, y_max = box
    width, height = image_size
    
    x = (x_min + x_max) / (2 * width)
    y = (y_min + y_max) / (2 * height)
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    
    return [x, y, w, h]


def denormalize_box(box: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Denormalize a YOLO format bounding box.
    
    Args:
        box: Normalized box [x, y, w, h]
        image_size: Image size (width, height)
        
    Returns:
        Tuple[int, int, int, int]: Bounding box (x_min, y_min, x_max, y_max)
    """
    x, y, w, h = box
    width, height = image_size
    
    x_min = int((x - w/2) * width)
    y_min = int((y - h/2) * height)
    x_max = int((x + w/2) * width)
    y_max = int((y + h/2) * height)
    
    return (x_min, y_min, x_max, y_max)


class BoundingBoxCalculator:
    """
    Utility class for calculating bounding boxes and font sizes for text rendering.
    """
    
    @staticmethod
    def fit(text: str, box_size: Tuple[int, int], font_size: int = 24, font_path: str = 'ext/Microsoft Himalaya.ttf', debug: bool = False) -> Tuple[int, int]:
        """
        Calculate the true bounding box size for the specified text when it is wrapped and terminated to fit a given box size.
        Enhanced with timeout protection and iteration limits.

        Args:
            text: Text to be measured
            box_size: Tuple (width, height) specifying the size of the box to fit the text
            font_size: Size of the font
            font_path: Path to the font file
            debug: Enable debug output

        Returns:
            Tuple (width, height) representing the actual bounding box size of the wrapped and terminated text
        """
        import time
        start_time = time.time()
        timeout_seconds = 5  # 5 second timeout for fit operation
        max_lines = 100      # Maximum lines to process
        max_chars_per_line = 1000  # Maximum characters per line to prevent infinite loops
        
        # Validate inputs
        if not text or not text.strip():
            return (0, 0)
            
        if box_size[0] <= 0 or box_size[1] <= 0:
            if debug:
                print(f"Warning: Invalid box size {box_size}")
            return (0, 0)
        
        # Create a dummy image to get a drawing context
        dummy_image = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_image)

        # Define the font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
            if debug:
                print("Warning: Default font used, may not accurately measure text.")

        box_w, box_h = box_size
        actual_text_width, actual_text_height = 0, 0
        y_offset = 0
        lines_processed = 0

        # Process each line with safety limits
        for line in text.split('\n'):
            if lines_processed >= max_lines:
                if debug:
                    print(f"Warning: Reached maximum line limit ({max_lines})")
                break
                
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                if debug:
                    print(f"Warning: fit() timed out after {timeout_seconds}s")
                break
            
            char_iterations = 0
            while line and char_iterations < max_chars_per_line:
                char_iterations += 1
                
                # Find the breakpoint for wrapping with safety limit
                i = 0
                try:
                    for i in range(min(len(line), max_chars_per_line)):
                        if draw.textlength(line[:i + 1], font=font) > box_w:
                            break
                    else:
                        i = len(line)
                except Exception as e:
                    if debug:
                        print(f"Error in textlength calculation: {e}")
                    i = min(10, len(line))  # Fallback to small chunk

                # Ensure we make progress
                if i == 0:
                    i = 1  # Take at least one character to avoid infinite loop

                # Add the line to wrapped text
                wrapped_line = line[:i]

                try:
                    left, top, right, bottom = font.getbbox(wrapped_line)
                    line_width, line_height = right - left, bottom - top
                except Exception as e:
                    if debug:
                        print(f"Error in getbbox calculation: {e}")
                    # Fallback estimation
                    line_width = len(wrapped_line) * font_size // 2
                    line_height = font_size

                actual_text_width = max(actual_text_width, line_width)
                y_offset += line_height

                # Check if the next line exceeds the box height
                if y_offset > box_h:
                    y_offset -= line_height  # Remove the last line's height if it exceeds
                    break

                line = line[i:]

            lines_processed += 1
            if y_offset > box_h:
                break

        elapsed = time.time() - start_time
        if debug and elapsed > 1.0:
            print(f"fit() took {elapsed:.2f}s for text length {len(text)}, font size {font_size}")

        return actual_text_width, y_offset + 10

    @staticmethod
    def find_max_font(text: str, box_size: Tuple[int, int], font_path: str, max_size: int = 100, debug: bool = False) -> int:
        """
        Find maximum font size where text fits in box using binary search with timeout protection.
        
        Args:
            text: Text to fit
            box_size: Target box size (width, height)
            font_path: Path to font file
            max_size: Maximum font size to try
            debug: Enable debug output
            
        Returns:
            int: Maximum font size that fits
        """
        import time
        start_time = time.time()
        timeout_seconds = 10  # 10 second timeout
        max_iterations = 50   # Maximum iterations to prevent infinite loops
        
        # Validate inputs
        if not text or not text.strip():
            if debug:
                print("Warning: Empty text provided to find_max_font, returning minimum font size")
            return 1
            
        if box_size[0] <= 0 or box_size[1] <= 0:
            if debug:
                print(f"Warning: Invalid box size {box_size}, returning minimum font size")
            return 1
        
        low, high = 1, min(max_size, 200)  # Cap maximum size to prevent extreme values
        best = 1
        iterations = 0
        
        if debug:
            print(f"Starting font size search for text: '{text[:50]}...' in box {box_size}")
        
        while low <= high and iterations < max_iterations:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                if debug:
                    print(f"find_max_font timed out after {timeout_seconds}s, returning best so far: {best}")
                break
                
            iterations += 1
            mid = (low + high) // 2
            
            try:
                fit_start = time.time()
                w, h = BoundingBoxCalculator.fit(text, box_size, mid, font_path)
                fit_time = time.time() - fit_start
                
                if debug and fit_time > 1.0:  # Log slow fit operations
                    print(f"Slow fit operation: {fit_time:.2f}s for font size {mid}")
                
                if w <= box_size[0] and h <= box_size[1]:
                    best = mid
                    low = mid + 1
                    if debug:
                        print(f"Font size {mid} fits ({w}x{h} <= {box_size})")
                else:
                    high = mid - 1
                    if debug:
                        print(f"Font size {mid} too large ({w}x{h} > {box_size})")
                        
            except Exception as e:
                if debug:
                    print(f"Error in fit calculation for font size {mid}: {e}")
                # If fit fails, assume font is too large
                high = mid - 1
        
        elapsed = time.time() - start_time
        if debug:
            print(f"find_max_font completed in {elapsed:.2f}s after {iterations} iterations, best font size: {best}")
            
        return best
