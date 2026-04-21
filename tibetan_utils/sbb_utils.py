"""
Utilities for interacting with the Staatsbibliothek zu Berlin (SBB) digital collections.
"""

import os
import re
import urllib.request
import ssl
import xml.etree.ElementTree as ET
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
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
                      output_dir: str = 'sbb_images', verify_ssl: bool = True,
                      download_workers: int = 8, **kwargs) -> List[dict]:
    """
    Process images from the Staatsbibliothek zu Berlin.
    
    Args:
        ppn: PPN (Pica Production Number) of the document
        processor_func: Function to process each image
        max_images: Maximum number of images to process (0 = all)
        download: Whether to download images before processing
        output_dir: Directory to save downloaded images
        verify_ssl: Whether to verify SSL certificates
        download_workers: Number of worker threads for parallel downloads
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
        worker_count = max(1, int(download_workers))
        if worker_count == 1 or len(image_urls) <= 1:
            for url in image_urls:
                image_path = download_image(url, temp_dir, verify_ssl=verify_ssl)
                if image_path:
                    image_paths.append(image_path)
        else:
            max_workers = min(worker_count, len(image_urls))
            print(f"Parallel download enabled (workers={max_workers})")

            def _download_one(idx_url: Tuple[int, str]) -> Tuple[int, Optional[str], str]:
                idx, url = idx_url
                image_path = download_image(url, temp_dir, verify_ssl=verify_ssl)
                return idx, image_path, url

            ordered_paths: List[Optional[str]] = [None] * len(image_urls)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_download_one, pair) for pair in enumerate(image_urls)]
                for future in as_completed(futures):
                    idx, image_path, _url = future.result()
                    if image_path:
                        ordered_paths[idx] = image_path
                    completed += 1
                    if completed % 25 == 0 or completed == len(image_urls):
                        print(f"Downloaded {completed}/{len(image_urls)}")

            image_paths = [p for p in ordered_paths if p]
        
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

    Extracts all commonly available MODS fields from the METS XML, including
    title, subtitle, authors/contributors, origin info (place, publisher, date),
    physical description, identifiers (URN, etc.), subjects/classifications,
    abstract, notes, language, and record info.

    Args:
        ppn: PPN (Pica Production Number) of the document
        verify_ssl: Whether to verify SSL certificates

    Returns:
        dict: Document metadata with the following keys (None if not available):
            ppn, title, subtitle, title_part, authors, author (first author),
            contributors, date, date_other, place, publisher, edition,
            physical_description, extent, identifiers, subjects, classifications,
            abstract, notes, language, languages, record_origin, record_creation_date,
            pages, url
    """
    metadata = {
        'ppn': ppn,
        'title': None,
        'subtitle': None,
        'title_part': None,
        'authors': [],
        'author': None,
        'contributors': [],
        'date': None,
        'date_other': None,
        'place': None,
        'publisher': None,
        'edition': None,
        'physical_description': None,
        'extent': None,
        'identifiers': {},
        'subjects': [],
        'classifications': [],
        'abstract': None,
        'notes': [],
        'language': None,
        'languages': [],
        'record_origin': None,
        'record_creation_date': None,
        'pages': 0,
        'url': f"https://digital.staatsbibliothek-berlin.de/werkansicht?PPN={ppn}"
    }

    def _text(el) -> Optional[str]:
        return el.text.strip() if el is not None and el.text else None

    try:
        metadata_url = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"

        if not verify_ssl:
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None

        with urllib.request.urlopen(metadata_url, context=ssl_context) as response:
            root = ET.parse(response).getroot()

            ns = {
                'mets': 'http://www.loc.gov/METS/',
                'mods': 'http://www.loc.gov/mods/v3'
            }

            mods = root.find('.//mods:mods', ns)
            if mods is not None:
                # --- Title ---
                title_info = mods.find('.//mods:titleInfo', ns)
                if title_info is not None:
                    metadata['title'] = _text(title_info.find('mods:title', ns))
                    metadata['subtitle'] = _text(title_info.find('mods:subTitle', ns))
                    metadata['title_part'] = _text(title_info.find('mods:partName', ns))

                # --- Authors / contributors ---
                authors = []
                contributors = []
                for name_el in mods.findall('.//mods:name', ns):
                    role_el = name_el.find('.//mods:roleTerm', ns)
                    role = _text(role_el) or ""
                    name_parts = [_text(p) for p in name_el.findall('mods:namePart', ns) if p.text]
                    full_name = " ".join(filter(None, name_parts)).strip() or None
                    if full_name:
                        if role.lower() in ("", "aut", "author", "creator"):
                            authors.append(full_name)
                        else:
                            contributors.append({"name": full_name, "role": role})
                metadata['authors'] = authors
                metadata['author'] = authors[0] if authors else None
                metadata['contributors'] = contributors

                # --- Origin info ---
                origin = mods.find('.//mods:originInfo', ns)
                if origin is not None:
                    metadata['publisher'] = _text(origin.find('mods:publisher', ns))
                    metadata['edition'] = _text(origin.find('mods:edition', ns))
                    # Prefer dateIssued, fall back to dateCreated / copyrightDate
                    for date_tag in ('mods:dateIssued', 'mods:dateCreated', 'mods:copyrightDate'):
                        d = _text(origin.find(date_tag, ns))
                        if d:
                            metadata['date'] = d
                            break
                    metadata['date_other'] = _text(origin.find('mods:dateOther', ns))
                    place_el = origin.find('.//mods:placeTerm[@type="text"]', ns)
                    if place_el is None:
                        place_el = origin.find('.//mods:placeTerm', ns)
                    metadata['place'] = _text(place_el)

                # --- Physical description ---
                phys = mods.find('.//mods:physicalDescription', ns)
                if phys is not None:
                    metadata['physical_description'] = _text(phys.find('mods:form', ns))
                    metadata['extent'] = _text(phys.find('mods:extent', ns))

                # --- Identifiers (URN, ISBN, ISSN, etc.) ---
                identifiers: dict = {}
                for id_el in mods.findall('.//mods:identifier', ns):
                    id_type = id_el.get('type', 'unknown')
                    id_val = _text(id_el)
                    if id_val:
                        identifiers[id_type] = id_val
                metadata['identifiers'] = identifiers

                # --- Subjects ---
                subjects = []
                for subj in mods.findall('.//mods:subject', ns):
                    parts = []
                    for child in subj:
                        t = _text(child)
                        if t:
                            parts.append(t)
                    if parts:
                        subjects.append(" -- ".join(parts))
                metadata['subjects'] = subjects

                # --- Classifications ---
                classifications = []
                for cls in mods.findall('.//mods:classification', ns):
                    authority = cls.get('authority', '')
                    val = _text(cls)
                    if val:
                        classifications.append({"authority": authority, "value": val})
                metadata['classifications'] = classifications

                # --- Abstract ---
                metadata['abstract'] = _text(mods.find('.//mods:abstract', ns))

                # --- Notes ---
                notes = [_text(n) for n in mods.findall('.//mods:note', ns) if n.text]
                metadata['notes'] = [n for n in notes if n]

                # --- Language ---
                languages = []
                for lang_el in mods.findall('.//mods:language/mods:languageTerm', ns):
                    lang = _text(lang_el)
                    if lang:
                        languages.append(lang)
                metadata['languages'] = languages
                metadata['language'] = languages[0] if languages else None

                # --- Record info ---
                rec = mods.find('.//mods:recordInfo', ns)
                if rec is not None:
                    metadata['record_origin'] = _text(rec.find('mods:recordOrigin', ns))
                    metadata['record_creation_date'] = _text(rec.find('mods:recordCreationDate', ns))

            # Count pages
            files = root.findall('.//mets:fileGrp[@USE="DEFAULT"]/mets:file', ns)
            metadata['pages'] = len(files)

    except Exception as e:
        print(f"Error retrieving metadata: {e}")

    return metadata
