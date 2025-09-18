"""
I/O utilities for the TibetanOCR project.
"""

import os
import re
import json
import yaml
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Union, Any


def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        str: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def extract_filename(path_or_url: str, with_extension: bool = True) -> str:
    """
    Extract filename from a path or URL.
    
    Args:
        path_or_url: File path or URL
        with_extension: Whether to include the file extension
        
    Returns:
        str: Extracted filename
    """
    # Extract PPN pattern from SBB URLs
    ppn_match = re.search(r'PPN(\d{10})-(\d{8})', path_or_url)
    if ppn_match:
        if with_extension:
            return f"PPN{ppn_match.group(1)}-{ppn_match.group(2)}.jpg"
        else:
            return f"PPN{ppn_match.group(1)}-{ppn_match.group(2)}"
    
    # Extract filename from path
    filename = os.path.basename(path_or_url)
    if not with_extension:
        filename = os.path.splitext(filename)[0]
    
    return filename


def save_json(data: Dict[str, Any], filepath: str, ensure_directory: bool = True) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the JSON file
        ensure_directory: Whether to ensure the directory exists
    """
    if ensure_directory:
        ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: str, ensure_directory: bool = True) -> None:
    """
    Save data as YAML file.
    
    Args:
        data: Data to save
        filepath: Path to save the YAML file
        ensure_directory: Whether to ensure the directory exists
    """
    if ensure_directory:
        ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file.
    
    Args:
        filepath: Path to the YAML file
        
    Returns:
        dict: Loaded data
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def find_images(directory: str, recursive: bool = False) -> List[str]:
    """
    Find image files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if recursive:
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and
                any(f.lower().endswith(ext) for ext in image_extensions)]


def get_output_path(base_dir: str, name: str, filename: str, create_dir: bool = True) -> str:
    """
    Get output path for a file.
    
    Args:
        base_dir: Base directory
        name: Experiment name
        filename: Filename
        create_dir: Whether to create the directory
        
    Returns:
        str: Output path
    """
    output_dir = os.path.join(base_dir, name)
    if create_dir:
        ensure_dir(output_dir)
    
    return os.path.join(output_dir, filename)


def hash_current_time() -> str:
    """
    Generate a hash based on current time for unique identifiers.
    
    Returns:
        str: SHA256 hash of current time in nanoseconds
    """
    # Get the current time
    current_time = time.time_ns()

    # Convert the current time to a string
    time_str = str(current_time)

    # Create a hash object (using SHA256)
    hash_object = hashlib.sha256()

    # Update the hash object with the time string
    hash_object.update(time_str.encode())

    # Get the hexadecimal digest of the hash
    time_hash = hash_object.hexdigest()

    return time_hash
