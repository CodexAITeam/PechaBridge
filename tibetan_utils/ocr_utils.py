"""
OCR utilities for the TibetanOCR project.
"""

import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Union

from .image_utils import extract_text_region, cv2_to_pil
from .io_utils import save_json, ensure_dir


def apply_ocr(image: Union[np.ndarray, Image.Image], lang: str = 'eng+deu', 
             config: str = '') -> str:
    """
    Apply OCR to an image.
    
    Args:
        image: Image to process (numpy array or PIL Image)
        lang: Language for Tesseract OCR
        config: Additional Tesseract configuration
        
    Returns:
        str: Recognized text
    """
    # Convert to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = cv2_to_pil(image)
    
    # Apply OCR
    text = pytesseract.image_to_string(image, lang=lang, config=config)
    
    return text.strip()


def apply_ocr_to_detection(image: np.ndarray, box: List[float], lang: str = 'eng+deu', 
                          config: str = '') -> str:
    """
    Apply OCR to a detected text region.
    
    Args:
        image: Image as a numpy array
        box: Bounding box [x, y, w, h, conf, class]
        lang: Language for Tesseract OCR
        config: Additional Tesseract configuration
        
    Returns:
        str: Recognized text
    """
    # Extract the text region
    text_region, _ = extract_text_region(image, box)
    
    # Apply OCR
    text = apply_ocr(text_region, lang=lang, config=config)
    
    return text


def process_detections(image: np.ndarray, results: List[Any], lang: str = 'eng+deu', 
                      config: str = '') -> List[Dict[str, Any]]:
    """
    Process detection results and apply OCR.
    
    Args:
        image: Image as a numpy array
        results: Detection results from YOLO
        lang: Language for Tesseract OCR
        config: Additional Tesseract configuration
        
    Returns:
        List[Dict[str, Any]]: List of detections with OCR results
    """
    detections = []
    
    for i, result in enumerate(results):
        boxes = result.boxes
        
        for j, box in enumerate(boxes.data):
            # Extract box information
            x, y, w, h, conf, cls = box.tolist()
            
            # Apply OCR
            text = apply_ocr_to_detection(image, box, lang=lang, config=config)
            
            # Add detection
            detection = {
                "id": j,
                "box": {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h)
                },
                "confidence": float(conf),
                "class": int(cls),
                "text": text
            }
            detections.append(detection)
    
    return detections


def save_ocr_results(results: List[Dict[str, Any]], image_name: str, output_dir: str) -> str:
    """
    Save OCR results to a JSON file.
    
    Args:
        results: OCR results
        image_name: Name of the image
        output_dir: Output directory
        
    Returns:
        str: Path to the saved JSON file
    """
    # Create output data
    output_data = {
        "image_name": image_name,
        "detections": results
    }
    
    # Create output path
    json_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_ocr.json")
    
    # Save results
    save_json(output_data, json_path)
    
    return json_path


def save_text_regions(image: np.ndarray, results: List[Dict[str, Any]], 
                     image_name: str, output_dir: str) -> List[str]:
    """
    Save text regions as images.
    
    Args:
        image: Image as a numpy array
        results: OCR results
        image_name: Name of the image
        output_dir: Output directory
        
    Returns:
        List[str]: Paths to the saved images
    """
    # Create output directory
    crop_dir = os.path.join(output_dir, "crops")
    ensure_dir(crop_dir)
    
    saved_paths = []
    
    for i, detection in enumerate(results):
        # Get box
        box = [
            detection["box"]["x"],
            detection["box"]["y"],
            detection["box"]["width"],
            detection["box"]["height"]
        ]
        
        # Extract region
        region, _ = extract_text_region(image, box)
        
        # Save region
        base_name = os.path.splitext(image_name)[0]
        crop_path = os.path.join(crop_dir, f"{base_name}_crop_{i}.jpg")
        cv2.imwrite(crop_path, region)
        
        saved_paths.append(crop_path)
    
    return saved_paths


def process_image_with_ocr(image_path: Union[str, np.ndarray], model, output_dir: str = None,
                          lang: str = 'eng+deu', conf: float = 0.25,
                          tesseract_config: str = '', save_crops: bool = False) -> Dict[str, Any]:
    """
    Process an image with YOLO and apply OCR to detected text blocks.
    
    Args:
        image_path: Path to the image or image as a numpy array
        model: YOLO model
        output_dir: Output directory
        lang: Language for Tesseract OCR
        conf: Confidence threshold for detections
        save_crops: Whether to save cropped text regions
        
    Returns:
        Dict[str, Any]: OCR results
    """
    # Load the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
    else:
        # If image_path is already a numpy array
        image = image_path
        image_name = f"image_{hash(str(image.shape))}.jpg"
    
    # Run YOLO inference
    results = model.predict(source=image, conf=conf)
    
    # Process detections
    ocr_results = process_detections(image, results, lang=lang, config=tesseract_config)
    
    # Save results if output directory is provided
    if output_dir:
        ensure_dir(output_dir)
        json_path = save_ocr_results(ocr_results, image_name, output_dir)
        
        # Save cropped text regions if requested
        if save_crops:
            crop_paths = save_text_regions(image, ocr_results, image_name, output_dir)
    
    # Create output data
    output_data = {
        "image_name": image_name,
        "detections": ocr_results
    }
    
    return output_data
