"""
Utilities for interacting with the Staatsbibliothek zu Berlin (SBB) digital collections.
"""

import os
import re
import urllib.request
import ssl
import xml.etree.ElementTree as ET
import hashlib
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import io
import cv2

from .io_utils import ensure_dir, extract_filename


def get_images_from_sbb(ppn: str, verify_ssl: bool = True) -> List[str]:
    """
    Retrieve image URLs from the Staatsbibliothek zu Berlin.
    
    Args:
        ppn: PPN (Pica Production Number) of the document
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        List[str]: List of image URLs
    """
    print(f"Retrieving metadata for PPN {ppn}...")
    files = []
    
    try:
        metadata_url = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"
        
        # Create SSL context
        if not verify_ssl:
            print("SSL verification disabled")
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None
            
        # Open URL with or without SSL verification
        with urllib.request.urlopen(metadata_url, context=ssl_context) as response:
            metadata = ET.parse(response).getroot()
            
            # Namespace for METS XML
            ns = {
                'mets': 'http://www.loc.gov/METS/',
                'xlink': 'http://www.w3.org/1999/xlink'
            }
            
            # Search for fileGrp with USE="DEFAULT"
            for fileGrp in metadata.findall('.//mets:fileGrp[@USE="DEFAULT"]', ns):
                for file in fileGrp.findall('.//mets:file', ns):
                    flocat = file.find('.//mets:FLocat', ns)
                    if flocat is not None:
                        url = flocat.get('{http://www.w3.org/1999/xlink}href')
                        files.append(url)
                        
            print(f"Found: {len(files)} images")
            
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
    
    return files


def download_image(url: str, output_dir: Optional[str] = None, verify_ssl: bool = True, 
                  return_array: bool = False) -> Union[str, np.ndarray, Image.Image]:
    """
    Download an image from a URL.
    
    Args:
        url: URL of the image
        output_dir: Optional directory to save the image
        verify_ssl: Whether to verify SSL certificates
        return_array: Whether to return a numpy array instead of a PIL Image
        
    Returns:
        Union[str, np.ndarray, Image.Image]: Path to the downloaded image, numpy array, or PIL Image
    """
    try:
        # Create SSL context
        if not verify_ssl:
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None
            
        # Open URL with or without SSL verification
        with urllib.request.urlopen(url, context=ssl_context) as response:
            image_data = response.read()
            
            # Extract filename from URL
            filename = extract_filename(url)
            
            if output_dir:
                # Save image to disk
                ensure_dir(output_dir)
                base, ext = os.path.splitext(filename)
                if not ext:
                    ext = ".jpg"

                # SBB URLs often end in generic names like default.jpg/png.
                # Use a stable URL hash suffix in that case, or whenever a
                # file collision would overwrite a different page.
                generic_names = {"default", "image", "download"}
                is_generic = base.lower() in generic_names
                image_path = os.path.join(output_dir, f"{base}{ext}")
                if is_generic or os.path.exists(image_path):
                    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
                    image_path = os.path.join(output_dir, f"{base}_{url_hash}{ext}")
                    i = 2
                    while os.path.exists(image_path):
                        image_path = os.path.join(output_dir, f"{base}_{url_hash}_{i}{ext}")
                        i += 1

                with open(image_path, 'wb') as f:
                    f.write(image_data)
                return image_path
            else:
                # Return image as numpy array or PIL Image
                if return_array:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return img
                else:
                    return Image.open(io.BytesIO(image_data))
                
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None


def process_sbb_images(ppn: str, processor_func, max_images: int = 0, download: bool = False, 
                      output_dir: str = 'sbb_images', verify_ssl: bool = True, **kwargs) -> List[dict]:
    """
    Process images from the Staatsbibliothek zu Berlin.
    
    Args:
        ppn: PPN (Pica Production Number) of the document
        processor_func: Function to process each image
        max_images: Maximum number of images to process (0 = all)
        download: Whether to download images before processing
        output_dir: Directory to save downloaded images
        verify_ssl: Whether to verify SSL certificates
        **kwargs: Additional arguments for the processor function
        
    Returns:
        List[dict]: Processing results
    """
    # Retrieve image URLs
    image_urls = get_images_from_sbb(ppn, verify_ssl=verify_ssl)
    
    if not image_urls:
        print("No images found. Exiting.")
        return []
    
    # Limit the number of images if requested
    if max_images > 0 and len(image_urls) > max_images:
        print(f"Limiting to {max_images} images (out of {len(image_urls)})")
        image_urls = image_urls[:max_images]
    
    results = []
    
    # Process images
    if download:
        # Download images and process local files
        temp_dir = ensure_dir(output_dir)
        print(f"Downloading images to directory: {temp_dir}")
        
        image_paths = []
        for url in image_urls:
            image_path = download_image(url, temp_dir, verify_ssl=verify_ssl)
            if image_path:
                image_paths.append(image_path)
        
        if not image_paths:
            print("No images could be downloaded. Exiting.")
            return []
            
        # Process each image
        for i, img_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            result = processor_func(img_path, **kwargs)
            results.append(result)
    else:
        # Process images directly from the web
        print(f"Processing {len(image_urls)} images directly from the web...")
        
        for i, url in enumerate(image_urls):
            print(f"Processing image {i+1}/{len(image_urls)}: {url}")
            try:
                # Download image (without saving)
                image = download_image(url, verify_ssl=verify_ssl, return_array=True)
                if image is None:
                    continue
                
                # Process the image
                result = processor_func(image, **kwargs)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    return results


def get_sbb_metadata(ppn: str, verify_ssl: bool = True) -> dict:
    """
    Get metadata for a document from the Staatsbibliothek zu Berlin.
    
    Args:
        ppn: PPN (Pica Production Number) of the document
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        dict: Document metadata
    """
    metadata = {
        'ppn': ppn,
        'title': None,
        'author': None,
        'date': None,
        'publisher': None,
        'language': None,
        'pages': 0,
        'url': f"https://digital.staatsbibliothek-berlin.de/werkansicht?PPN={ppn}"
    }
    
    try:
        metadata_url = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"
        
        # Create SSL context
        if not verify_ssl:
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None
            
        # Open URL with or without SSL verification
        with urllib.request.urlopen(metadata_url, context=ssl_context) as response:
            root = ET.parse(response).getroot()
            
            # Namespace for METS XML
            ns = {
                'mets': 'http://www.loc.gov/METS/',
                'mods': 'http://www.loc.gov/mods/v3'
            }
            
            # Extract metadata
            mods = root.find('.//mods:mods', ns)
            if mods is not None:
                # Title
                title_info = mods.find('.//mods:titleInfo/mods:title', ns)
                if title_info is not None:
                    metadata['title'] = title_info.text
                
                # Author
                name = mods.find('.//mods:name[@type="personal"]/mods:namePart', ns)
                if name is not None:
                    metadata['author'] = name.text
                
                # Date
                date = mods.find('.//mods:originInfo/mods:dateIssued', ns)
                if date is not None:
                    metadata['date'] = date.text
                
                # Publisher
                publisher = mods.find('.//mods:originInfo/mods:publisher', ns)
                if publisher is not None:
                    metadata['publisher'] = publisher.text
                
                # Language
                language = mods.find('.//mods:language/mods:languageTerm', ns)
                if language is not None:
                    metadata['language'] = language.text
            
            # Count pages
            files = root.findall('.//mets:fileGrp[@USE="DEFAULT"]/mets:file', ns)
            metadata['pages'] = len(files)
            
    except Exception as e:
        print(f"Error retrieving metadata: {e}")
    
    return metadata
