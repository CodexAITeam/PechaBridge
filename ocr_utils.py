#!/usr/bin/env python3
"""
Hilfsfunktionen für die Verarbeitung von Bildern und OCR.
Enthält gemeinsam genutzte Funktionen für inference_sbb.py und ocr_on_detections.py.
"""

import os
import re
import urllib.request
import ssl
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import json
import cv2
import numpy as np
from PIL import Image
import io


def get_images_from_sbb(ppn, verify_ssl=True):
    """
    Ruft Bilddaten von der Staatsbibliothek zu Berlin ab.
    
    Args:
        ppn: Die PPN (Pica Production Number) des Dokuments
        verify_ssl: SSL-Zertifikate überprüfen
        
    Returns:
        Eine Liste von URLs zu den Bildern
    """
    print(f"Rufe Metadaten für PPN {ppn} ab...")
    files = []
    
    try:
        metadata_url = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"
        
        # SSL-Kontext erstellen
        if not verify_ssl:
            print("SSL-Verifizierung deaktiviert")
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None
            
        # URL öffnen mit oder ohne SSL-Verifizierung
        with urllib.request.urlopen(metadata_url, context=ssl_context) as response:
            metadata = ET.parse(response).getroot()
            
            # Namespace für METS XML
            ns = {
                'mets': 'http://www.loc.gov/METS/',
                'xlink': 'http://www.w3.org/1999/xlink'
            }
            
            # Suche nach der fileGrp mit USE="DEFAULT"
            for fileGrp in metadata.findall('.//mets:fileGrp[@USE="DEFAULT"]', ns):
                for file in fileGrp.findall('.//mets:file', ns):
                    flocat = file.find('.//mets:FLocat', ns)
                    if flocat is not None:
                        url = flocat.get('{http://www.w3.org/1999/xlink}href')
                        files.append(url)
                        
            print(f"Gefunden: {len(files)} Bilder")
            
    except Exception as e:
        print(f"Fehler beim Abrufen der Metadaten: {e}")
    
    return files


def download_image(url, output_dir=None, verify_ssl=True, return_array=False):
    """
    Lädt ein Bild von einer URL herunter und speichert es optional.
    
    Args:
        url: Die URL des Bildes
        output_dir: Optional. Verzeichnis zum Speichern des Bildes
        verify_ssl: SSL-Zertifikate überprüfen
        return_array: Wenn True, gibt ein numpy-Array zurück, sonst PIL Image oder Dateipfad
        
    Returns:
        Pfad zum heruntergeladenen Bild, PIL Image oder numpy-Array
    """
    try:
        # SSL-Kontext erstellen
        if not verify_ssl:
            ssl_context = ssl._create_unverified_context()
        else:
            ssl_context = None
            
        # URL öffnen mit oder ohne SSL-Verifizierung
        with urllib.request.urlopen(url, context=ssl_context) as response:
            image_data = response.read()
            
            # Extrahiere Dateinamen aus URL
            match = re.search(r'PPN(\d{10})-(\d{8})', url)
            if match:
                filename = f"PPN{match.group(1)}-{match.group(2)}.jpg"
            else:
                filename = f"image_{hash(url)}.jpg"
            
            if output_dir:
                # Speichere Bild auf Festplatte
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, filename)
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                return image_path
            else:
                # Gib Bild als numpy-Array oder PIL Image zurück
                if return_array:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return img
                else:
                    return Image.open(io.BytesIO(image_data))
                
    except Exception as e:
        print(f"Fehler beim Herunterladen des Bildes {url}: {e}")
        return None


def extract_text_region(image, box):
    """
    Extrahiert einen Textbereich aus einem Bild basierend auf einer Bounding-Box.
    
    Args:
        image: Das Bild als numpy-Array
        box: Die Bounding-Box [x, y, w, h, conf, class] (normalisierte Koordinaten)
        
    Returns:
        Der ausgeschnittene Bildbereich als numpy-Array und die absoluten Koordinaten
    """
    # Extrahiere Koordinaten
    x, y, w, h = box[:4]
    
    # Konvertiere relative Koordinaten in absolute Pixel-Koordinaten
    height, width = image.shape[:2]
    x_min = int((x - w/2) * width)
    y_min = int((y - h/2) * height)
    x_max = int((x + w/2) * width)
    y_max = int((y + h/2) * height)
    
    # Stelle sicher, dass die Koordinaten innerhalb des Bildes liegen
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    # Schneide den Textblock aus
    text_region = image[y_min:y_max, x_min:x_max]
    
    return text_region, (x_min, y_min, x_max, y_max)


def add_common_arguments(parser):
    """
    Fügt gemeinsame Kommandozeilenargumente zu einem ArgumentParser hinzu.
    
    Args:
        parser: Der ArgumentParser, zu dem die Argumente hinzugefügt werden sollen
        
    Returns:
        Der aktualisierte ArgumentParser
    """
    # Modelloptionen
    model_group = parser.add_argument_group('Modelloptionen')
    model_group.add_argument('--model', type=str, required=True,
                        help='Pfad zum trainierten Modell (z.B. runs/detect/train/weights/best.pt)')
    model_group.add_argument('--conf', type=float, default=0.25,
                        help='Konfidenz-Schwellenwert für Detektionen')
    model_group.add_argument('--imgsz', type=int, default=1024,
                        help='Bildgröße für die Inferenz')
    model_group.add_argument('--device', type=str, default='',
                        help='Gerät für die Inferenz (z.B. cpu, 0, 0,1,2,3 für mehrere GPUs)')
    
    # SBB-spezifische Optionen
    sbb_group = parser.add_argument_group('SBB-spezifische Optionen')
    sbb_group.add_argument('--ppn', type=str, help='PPN (Pica Production Number) des Dokuments in der Staatsbibliothek zu Berlin')
    sbb_group.add_argument('--download', action='store_true',
                        help='Lade Bilder herunter anstatt sie direkt zu verarbeiten')
    sbb_group.add_argument('--no-ssl-verify', action='store_true',
                        help='Deaktiviere SSL-Zertifikatsverifizierung (nicht empfohlen für Produktionsumgebungen)')
    sbb_group.add_argument('--max-images', type=int, default=0,
                        help='Maximale Anzahl an Bildern für die Inferenz (0 = alle)')
    
    # Ausgabeoptionen
    output_group = parser.add_argument_group('Ausgabeoptionen')
    output_group.add_argument('--output', type=str, default='results',
                        help='Verzeichnis zum Speichern der Ergebnisse')
    output_group.add_argument('--source', type=str, help='Pfad zu Bildern oder Verzeichnis für die Inferenz')
    
    return parser
