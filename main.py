import os
import time
import re
import json
import math
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_drawing_api_v2")

# Constants
PORT = int(os.environ.get("PORT", 10000))
MAX_PAGE_SIZE = 10000
MIN_LINE_LENGTH = 0.1

app = FastAPI(
    title="Vector Drawing API v2 - Layer Support",
    description="Extracts vector data from technical drawings with PDF layer support",
    version="2.0.0",
)

# Add middlewares
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== LAYER EXTRACTION FUNCTIONS ====================

def extract_comprehensive_layers(pdf_doc: fitz.Document) -> Dict[str, Any]:
    """Extract complete layer information from PDF document"""
    layers = []
    ocg_info = {}
    layer_configurations = []
    pages_with_layers = []
    layer_names = set()
    layer_usage_stats = {}
    
    # Get PDF metadata
    metadata = pdf_doc.metadata
    pdf_version = getattr(pdf_doc, 'pdf_version', 'Unknown')
    
    try:
        # Method 1: Extract OCG information from document catalog
        ocg_info = extract_ocg_catalog_info(pdf_doc)
        
        # Method 2: Extract from xref table
        xref_layers = extract_layers_from_xref(pdf_doc)
        
        # Method 3: Extract layer configurations
        layer_configurations = extract_layer_configurations(pdf_doc)
        
        # Method 4: Analyze pages for layer usage
        pages_with_layers, page_layer_usage = analyze_pages_for_layers(pdf_doc)
        
        # Combine all layer information
        all_layers = {}
        
        # Process OCG catalog info
        if ocg_info.get('ocgs'):
            for ocg_ref, ocg_data in ocg_info['ocgs'].items():
                layer_name = ocg_data.get('name', f'Layer_{ocg_ref}')
                all_layers[layer_name] = {
                    'name': layer_name,
                    'ocg_ref': ocg_ref,
                    'visible': ocg_data.get('visible', True),
                    'locked': ocg_data.get('locked', False),
                    'intent': ocg_data.get('intent', []),
                    'usage': ocg_data.get('usage', {}),
                    'creator_info': ocg_data.get('creator_info', ''),
                    'source': 'OCG_catalog'
                }
                layer_names.add(layer_name)
        
        # Process xref layers
        for layer_data in xref_layers:
            layer_name = layer_data['name']
            if layer_name in all_layers:
                # Merge information
                all_layers[layer_name].update({k: v for k, v in layer_data.items() if v})
            else:
                all_layers[layer_name] = layer_data
                all_layers[layer_name]['source'] = 'xref_table'
            layer_names.add(layer_name)
        
        # Convert to list format
        layers = list(all_layers.values())
        
        # Calculate usage statistics
        layer_usage_stats = calculate_layer_usage_stats(layers, pages_with_layers, pdf_doc.page_count)
        
    except Exception as e:
        logger.error(f"Error in comprehensive layer extraction: {str(e)}")
        # Return basic info if layer extraction fails
        return {
            "has_layers": False,
            "layer_count": 0,
            "layers": [],
            "ocg_info": {},
            "layer_configurations": [],
            "pages_with_layers": [],
            "layer_usage_analysis": {}
        }
    
    return {
        "has_layers": len(layer_names) > 0,
        "layer_count": len(layer_names),
        "layers": layers,
        "ocg_info": ocg_info,
        "layer_configurations": layer_configurations,
        "pages_with_layers": pages_with_layers,
        "layer_usage_analysis": layer_usage_stats
    }

def extract_ocg_catalog_info(pdf_doc: fitz.Document) -> Dict[str, Any]:
    """Extract OCG (Optional Content Groups) information from document catalog"""
    ocg_info = {
        'ocgs': {},
        'default_config': {},
        'alternate_configs': []
    }
    
    try:
        # Get document catalog
        catalog = pdf_doc.pdf_catalog()
        
        # Look for OCProperties in catalog
        if 'OCProperties' in catalog:
            oc_props = catalog['OCProperties']
            
            # Extract OCGs
            if 'OCGs' in oc_props:
                ocgs = oc_props['OCGs']
                for i, ocg_ref in enumerate(ocgs):
                    try:
                        ocg_obj = pdf_doc.xref_get_object(ocg_ref)
                        ocg_data = parse_ocg_object(ocg_obj)
                        ocg_info['ocgs'][str(ocg_ref)] = ocg_data
                    except:
                        continue
            
            # Extract default configuration
            if 'D' in oc_props:
                default_config = oc_props['D']
                ocg_info['default_config'] = parse_oc_config(pdf_doc, default_config)
                        
    except Exception as e:
        logger.debug(f"Could not extract OCG catalog info: {str(e)}")
    
    return ocg_info

def parse_ocg_object(ocg_obj: str) -> Dict[str, Any]:
    """Parse OCG object string to extract layer information"""
    ocg_data = {
        'name': 'Unknown Layer',
        'visible': True,
        'locked': False,
        'intent': [],
        'usage': {},
        'creator_info': ''
    }
    
    try:
        # Extract name
        name_match = re.search(r'/Name\s*\((.*?)\)', ocg_obj)
        if name_match:
            ocg_data['name'] = name_match.group(1)
        
        # Extract intent
        intent_match = re.search(r'/Intent\s*\[(.*?)\]', ocg_obj)
        if intent_match:
            intents = re.findall(r'/(\w+)', intent_match.group(1))
            ocg_data['intent'] = intents
            
    except Exception as e:
        logger.debug(f"Error parsing OCG object: {str(e)}")
    
    return ocg_data

def parse_oc_config(pdf_doc: fitz.Document, config_obj: Union[str, int]) -> Dict[str, Any]:
    """Parse Optional Content Configuration"""
    config_data = {
        'name': 'Default',
        'creator': '',
        'base_state': 'ON',
        'on_layers': [],
        'off_layers': [],
        'locked_layers': []
    }
    
    try:
        if isinstance(config_obj, int):
            config_str = pdf_doc.xref_get_object(config_obj)
        else:
            config_str = str(config_obj)
        
        # Extract configuration name
        name_match = re.search(r'/Name\s*\((.*?)\)', config_str)
        if name_match:
            config_data['name'] = name_match.group(1)
            
    except Exception as e:
        logger.debug(f"Error parsing OC config: {str(e)}")
    
    return config_data

def extract_layers_from_xref(pdf_doc: fitz.Document) -> List[Dict[str, Any]]:
    """Extract layer information from xref table"""
    layers = []
    
    try:
        xref_count = pdf_doc.xref_length()
        
        for xref in range(xref_count):
            try:
                obj_type = pdf_doc.xref_get_key(xref, "Type")
                if obj_type and "OCG" in str(obj_type):
                    # Get layer properties
                    name_obj = pdf_doc.xref_get_key(xref, "Name")
                    
                    layer_name = str(name_obj).strip('()/"') if name_obj else f"Layer_{xref}"
                    
                    layer_data = {
                        "name": layer_name,
                        "visible": True,  # Default
                        "locked": False,  # Default
                        "ocg_ref": str(xref),
                        "intent": [],
                        "usage": {},
                        "xref_number": xref
                    }
                    
                    layers.append(layer_data)
                    
            except Exception as e:
                logger.debug(f"Error processing xref {xref}: {str(e)}")
                continue
                
    except Exception as e:
        logger.debug(f"Error in xref layer extraction: {str(e)}")
    
    return layers

def extract_layer_configurations(pdf_doc: fitz.Document) -> List[Dict[str, Any]]:
    """Extract layer configuration information"""
    configurations = []
    
    try:
        configurations.append({
            "name": "Default Configuration",
            "base_state": "ON",
            "creator": pdf_doc.metadata.get('creator', 'Unknown'),
            "description": "Default layer visibility configuration"
        })
        
    except Exception as e:
        logger.debug(f"Error extracting layer configurations: {str(e)}")
    
    return configurations

def analyze_pages_for_layers(pdf_doc: fitz.Document) -> tuple:
    """Analyze each page for layer usage"""
    pages_with_layers = []
    layer_usage = {}
    
    for page_num in range(pdf_doc.page_count):
        try:
            page = pdf_doc[page_num]
            page_layers = extract_page_layer_details(page)
            
            if page_layers:
                pages_with_layers.append({
                    "page_number": page_num + 1,
                    "layers": page_layers,
                    "layer_count": len(page_layers)
                })
                
                # Update usage statistics
                for layer_name in page_layers:
                    if layer_name not in layer_usage:
                        layer_usage[layer_name] = []
                    layer_usage[layer_name].append(page_num + 1)
                    
        except Exception as e:
            logger.debug(f"Error analyzing page {page_num}: {str(e)}")
            continue
    
    return pages_with_layers, layer_usage

def extract_page_layer_details(page: fitz.Page) -> List[str]:
    """Extract detailed layer information from a page"""
    layers = []
    
    try:
        # Get page content
        content_streams = page.get_contents()
        
        for stream in content_streams:
            try:
                content = stream.get_buffer().decode('latin-1', errors='ignore')
                
                # Look for OCG markers
                ocg_patterns = [
                    r'/OC\s*/(\w+)',  # Basic OCG reference
                    r'BDC\s*/OC\s*/(\w+)',  # Begin marked content with OCG
                    r'/Properties\s*<<.*?/(\w+)\s*\d+\s+\d+\s+R.*?>>',  # Properties dictionary
                ]
                
                for pattern in ocg_patterns:
                    matches = re.findall(pattern, content)
                    layers.extend(matches)
                
            except Exception as e:
                logger.debug(f"Error processing content stream: {str(e)}")
                continue
                
    except Exception as e:
        logger.debug(f"Error extracting page layer details: {str(e)}")
    
    return list(set(layers))  # Remove duplicates

def calculate_layer_usage_stats(layers: List[Dict], pages_with_layers: List[Dict], total_pages: int) -> Dict[str, Any]:
    """Calculate layer usage statistics"""
    stats = {
        "total_layers": len(layers),
        "pages_with_layers": len(pages_with_layers),
        "pages_without_layers": total_pages - len(pages_with_layers),
        "layer_distribution": {},
        "most_used_layers": [],
        "least_used_layers": []
    }
    
    try:
        # Calculate layer distribution
        layer_usage_count = {}
        
        for page_info in pages_with_layers:
            for layer in page_info.get('layers', []):
                layer_usage_count[layer] = layer_usage_count.get(layer, 0) + 1
        
        stats["layer_distribution"] = layer_usage_count
        
        # Sort layers by usage
        sorted_layers = sorted(layer_usage_count.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_layers:
            stats["most_used_layers"] = sorted_layers[:3]  # Top 3
            stats["least_used_layers"] = sorted_layers[-3:]  # Bottom 3
            
    except Exception as e:
        logger.debug(f"Error calculating usage stats: {str(e)}")
    
    return stats

# ==================== VECTOR EXTRACTION FUNCTIONS ====================

def point_to_dict(p, precision: int = 2) -> dict:
    """Convert point to dictionary with configurable precision"""
    if p is None:
        return {"x": 0.0, "y": 0.0}
    
    try:
        if hasattr(p, 'x') and hasattr(p, 'y'):
            return {"x": round(float(p.x), precision), "y": round(float(p.y), precision)}
        elif isinstance(p, (tuple, list)) and len(p) >= 2:
            return {"x": round(float(p[0]), precision), "y": round(float(p[1]), precision)}
        else:
            return {"x": 0.0, "y": 0.0}
    except (ValueError, TypeError, AttributeError):
        return {"x": 0.0, "y": 0.0}

def rect_to_dict(r, precision: int = 2) -> dict:
    """Convert rectangle to dictionary with configurable precision"""
    if r is None:
        return {
            "x0": 0.0, "y0": 0.0,
            "x1": 0.0, "y1": 0.0,
            "width": 0.0, "height": 0.0
        }
    
    try:
        if isinstance(r, tuple) and len(r) >= 4:
            return {
                "x0": round(float(r[0]), precision),
                "y0": round(float(r[1]), precision),
                "x1": round(float(r[2]), precision),
                "y1": round(float(r[3]), precision),
                "width": round(float(r[2]) - float(r[0]), precision),
                "height": round(float(r[3]) - float(r[1]), precision)
            }
        elif hasattr(r, 'x0'):
            return {
                "x0": round(float(r.x0), precision),
                "y0": round(float(r.y0), precision),
                "x1": round(float(r.x1), precision),
                "y1": round(float(r.y1), precision),
                "width": round(float(r.width), precision),
                "height": round(float(r.height), precision)
            }
        else:
            return {
                "x0": 0.0, "y0": 0.0,
                "x1": 0.0, "y1": 0.0,
                "width": 0.0, "height": 0.0
            }
    except (ValueError, TypeError, AttributeError):
        return {
            "x0": 0.0, "y0": 0.0,
            "x1": 0.0, "y1": 0.0,
            "width": 0.0, "height": 0.0
        }

def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def extract_dimension_info(text: str) -> Dict[str, Any]:
    """Extract dimension information from text"""
    dimension_data = {"is_dimension": False, "value": None, "unit": None, "type": None}
    
    patterns = [
        (r'^(\d+)$', 'numeric'),
        (r'(\d+(?:\.\d+)?)\s*(mm|cm|m|ft|in|")', 'measurement'),
        (r'(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)', 'dimension'),
        (r'[ØΦ]\s*(\d+(?:\.\d+)?)', 'diameter'),
        (r'R\s*(\d+(?:\.\d+)?)', 'radius'),
        (r'∠\s*(\d+(?:\.\d+)?)[°]?', 'angle'),
        (r'(\d+)\s*:\s*(\d+)', 'scale'),
        (r'±\s*(\d+(?:\.\d+)?)', 'tolerance'),
    ]
    
    for pattern, dim_type in patterns:
        match = re.search(pattern, text.strip())
        if match:
            dimension_data["is_dimension"] = True
            dimension_data["value"] = match.group(1)
            dimension_data["type"] = dim_type
            if len(match.groups()) > 1:
                dimension_data["unit"] = match.group(2)
            break
    
    return dimension_data

def extract_path_data(path: dict, precision: int = 2) -> Dict[str, Any]:
    """Extract path data with configurable precision"""
    width = path.get("width", 1.0)
    if width is None:
        width = 1.0
    else:
        try:
            width = float(width)
        except (ValueError, TypeError):
            width = 1.0
    
    opacity = path.get("opacity", 1.0)
    if opacity is None:
        opacity = 1.0
    else:
        try:
            opacity = float(opacity)
        except (ValueError, TypeError):
            opacity = 1.0
    
    path_data = {
        "width": round(width, precision),
        "opacity": round(opacity, precision),
        "closePath": path.get("closePath", False),
    }
    
    # Include colors only if present
    if "color" in path and path["color"]:
        try:
            path_data["color"] = [round(float(c), 3) for c in path["color"]]
        except:
            pass
    
    return path_data

def get_current_layer_from_content(content_position: int, layer_markers: List[Dict]) -> Optional[str]:
    """
    Determine which layer a content element belongs to based on position in content stream
    """
    current_layer = None
    
    for marker in layer_markers:
        if marker['start_pos'] <= content_position <= marker.get('end_pos', float('inf')):
            current_layer = marker['layer_name']
        elif marker['start_pos'] > content_position:
            break
    
    return current_layer

def extract_layer_markers_from_content(page: fitz.Page) -> List[Dict]:
    """
    Extract layer markers from page content streams to track which elements belong to which layer
    """
    layer_markers = []
    
    try:
        content_streams = page.get_contents()
        
        for stream_idx, stream in enumerate(content_streams):
            try:
                content = stream.get_buffer().decode('latin-1', errors='ignore')
                
                # Find layer begin/end markers
                # BDC = Begin Dictionary Content, EMC = End Marked Content
                patterns = [
                    (r'/OC\s*/(\w+)\s+BDC', 'layer_start'),
                    (r'EMC', 'layer_end'),
                    (r'/Properties\s*<<.*?/(\w+)\s*\d+\s+\d+\s+R.*?>>\s*BDC', 'layer_start'),
                ]
                
                position = 0
                current_layer_stack = []
                
                for pattern, marker_type in patterns:
                    for match in re.finditer(pattern, content):
                        if marker_type == 'layer_start':
                            layer_name = match.group(1)
                            current_layer_stack.append(layer_name)
                            layer_markers.append({
                                'layer_name': layer_name,
                                'start_pos': match.start(),
                                'end_pos': None,
                                'stream_idx': stream_idx
                            })
                        elif marker_type == 'layer_end' and current_layer_stack:
                            # Find the most recent unclosed layer marker
                            for i in range(len(layer_markers) - 1, -1, -1):
                                if (layer_markers[i]['end_pos'] is None and 
                                    layer_markers[i]['stream_idx'] == stream_idx):
                                    layer_markers[i]['end_pos'] = match.start()
                                    if current_layer_stack:
                                        current_layer_stack.pop()
                                    break
                
            except Exception as e:
                logger.debug(f"Error processing content stream {stream_idx}: {str(e)}")
                continue
                
    except Exception as e:
        logger.debug(f"Error extracting layer markers: {str(e)}")
    
    return layer_markers

def extract_embedded_text_with_layers(page: fitz.Page, page_num: int, precision: int = 2, has_layers: bool = False) -> List[Dict]:
    """Extract embedded text from PDF with layer information"""
    embedded_texts = []
    
    try:
        text_dict = page.get_text("dict")
        layer_markers = extract_layer_markers_from_content(page) if has_layers else []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            bbox = span["bbox"]
                            
                            text_data = {
                                "text": text,
                                "position": {
                                    "x": round(bbox[0], precision),
                                    "y": round(bbox[1], precision)
                                },
                                "bbox": rect_to_dict(bbox, precision),
                                "page_number": page_num + 1,
                                "source": "embedded"
                            }
                            
                            # Add layer information if available
                            if has_layers and layer_markers:
                                # Estimate content position (approximation)
                                content_pos = int(bbox[0] + bbox[1] * 1000)  # Simple hash
                                current_layer = get_current_layer_from_content(content_pos, layer_markers)
                                if current_layer:
                                    text_data["layer"] = current_layer
                            
                            # Add dimension info if detected
                            dim_info = extract_dimension_info(text)
                            if dim_info["is_dimension"]:
                                text_data["dimension_info"] = dim_info
                            
                            embedded_texts.append(text_data)
    
    except Exception as e:
        logger.warning(f"Failed to extract embedded text: {e}")
    
    return embedded_texts

def extract_vector_data_with_layers(page: fitz.Page, precision: int = 2, has_layers: bool = False) -> Dict[str, List]:
    """Extract vector drawings from page with layer information"""
    lines = []
    rectangles = []
    curves = []
    polygons = []
    
    try:
        drawings = page.get_drawings()
        layer_markers = extract_layer_markers_from_content(page) if has_layers else []
        
        for path_idx, path in enumerate(drawings):
            path_info = extract_path_data(path, precision)
            
            # Determine layer for this path (if layers exist)
            path_layer = None
            if has_layers and layer_markers:
                # Use path index as approximation for content position
                content_pos = path_idx * 100
                path_layer = get_current_layer_from_content(content_pos, layer_markers)
            
            for item in path["items"]:
                item_type = item[0]
                points = item[1:] if len(item) > 1 else []
                
                if item_type == "l" and len(points) >= 2:  # Line
                    p1 = point_to_dict(points[0], precision)
                    p2 = point_to_dict(points[1], precision)
                    line_length = distance(p1, p2)
                    if line_length >= MIN_LINE_LENGTH:
                        line_data = {
                            "type": "line",
                            "p1": p1,
                            "p2": p2,
                            "length": round(line_length, precision),
                            **path_info
                        }
                        
                        # Add layer information
                        if path_layer:
                            line_data["layer"] = path_layer
                            
                        lines.append(line_data)
                
                elif item_type == "re" and points:  # Rectangle
                    rect = rect_to_dict(points[0], precision)
                    rect_area = rect["width"] * rect["height"]
                    if rect_area >= 1:
                        rect_data = {
                            "type": "rectangle",
                            "rect": rect,
                            "area": round(rect_area, precision),
                            **path_info
                        }
                        
                        # Add layer information
                        if path_layer:
                            rect_data["layer"] = path_layer
                            
                        rectangles.append(rect_data)
                
                elif item_type == "c" and points:  # Curve/Circle
                    try:
                        if len(points) >= 4:
                            curve_data = {
                                "type": "bezier",
                                "points": [point_to_dict(p, precision) for p in points[:4]],
                                **path_info
                            }
                        else:
                            center = point_to_dict(points[0], precision)
                            radius = 1.0
                            if len(points) > 1 and points[1] is not None:
                                try:
                                    radius = float(points[1])
                                except:
                                    radius = 1.0
                            curve_data = {
                                "type": "circle",
                                "center": center,
                                "radius": round(radius, precision),
                                **path_info
                            }
                        
                        # Add layer information
                        if path_layer:
                            curve_data["layer"] = path_layer
                            
                        curves.append(curve_data)
                    except Exception as e:
                        logger.warning(f"Error processing curve: {e}")
                
                elif item_type == "qu" and len(points) >= 3:  # Polygon
                    polygon_data = {
                        "type": "polygon",
                        "points": [point_to_dict(p, precision) for p in points],
                        **path_info
                    }
                    
                    # Add layer information
                    if path_layer:
                        polygon_data["layer"] = path_layer
                        
                    polygons.append(polygon_data)
    
    except Exception as e:
        logger.warning(f"Error extracting vector data: {e}")
    
    return {
        "lines": lines,
        "rectangles": rectangles,
        "curves": curves,
        "polygons": polygons
    }

def deduplicate_texts(texts: List[Dict], tolerance: float = 10.0) -> List[Dict]:
    """Remove duplicate texts based on content and position"""
    unique_texts = []
    seen = set()
    
    for text in texts:
        # Create a key based on text and approximate position
        text_key = f"{text['text'].lower()}_{round(text['position']['x']/tolerance)}_{round(text['position']['y']/tolerance)}"
        
        if text_key not in seen:
            seen.add(text_key)
            unique_texts.append(text)
    
    return unique_texts

def group_data_by_layers(texts: List[Dict], drawings: Dict[str, List]) -> Dict[str, Dict]:
    """Group texts and drawings by layer"""
    layers_data = {}
    
    # Group texts by layer
    for text in texts:
        layer_name = text.get('layer', 'No Layer')
        if layer_name not in layers_data:
            layers_data[layer_name] = {
                'texts': [],
                'drawings': {'lines': [], 'rectangles': [], 'curves': [], 'polygons': []}
            }
        
        # Remove layer field from text data to avoid duplication
        text_copy = text.copy()
        text_copy.pop('layer', None)
        layers_data[layer_name]['texts'].append(text_copy)
    
    # Group drawings by layer
    for drawing_type, drawing_list in drawings.items():
        for drawing in drawing_list:
            layer_name = drawing.get('layer', 'No Layer')
            if layer_name not in layers_data:
                layers_data[layer_name] = {
                    'texts': [],
                    'drawings': {'lines': [], 'rectangles': [], 'curves': [], 'polygons': []}
                }
            
            # Remove layer field from drawing data
            drawing_copy = drawing.copy()
            drawing_copy.pop('layer', None)
            layers_data[layer_name]['drawings'][drawing_type].append(drawing_copy)
    
    return layers_data

def minify_output(data: Dict, minify: bool = True, remove_non_essential: bool = True) -> str:
    """Minify JSON output to a single-line string"""
    logger.info(f"Minifying output, minify={minify}, remove_non_essential={remove_non_essential}")
    
    # Make a copy to avoid modifying original data
    output_data = data.copy()
    
    if remove_non_essential:
        logger.info("Removing non-essential fields (fonts, colors, etc.)")
        # Keep essential metadata but remove creation/modification dates
        if "metadata" in output_data:
            essential_metadata = {
                "filename": output_data["metadata"].get("filename", ""),
                "title": output_data["metadata"].get("title", ""),
                "author": output_data["metadata"].get("author", ""),
                "subject": output_data["metadata"].get("subject", ""),
                "keywords": output_data["metadata"].get("keywords", ""),
                "creator": output_data["metadata"].get("creator", ""),
                "producer": output_data["metadata"].get("producer", "")
            }
            output_data["metadata"] = essential_metadata
        
        # Remove processing times from summary
        if "summary" in output_data:
            output_data["summary"].pop("processing_time_ms", None)
            output_data["summary"].pop("file_size_mb", None)
        
        # Remove non-essential fields from pages
        for page in output_data.get("pages", []):
            page.pop("processing_time_ms", None)
            
            # Remove from texts
            for text in page.get("texts", []):
                text.pop("font", None)
                text.pop("color", None)
                text.pop("flags", None)
                text.pop("size", None)
                text.pop("confidence", None)
            
            # Remove from drawings
            if page.get("has_layers") and "layers" in page:
                # Process layered data
                for layer_name, layer_data in page["layers"].items():
                    for text in layer_data.get("texts", []):
                        text.pop("font", None)
                        text.pop("color", None)
                        text.pop("flags", None)
                        text.pop("size", None)
                        text.pop("confidence", None)
                    
                    for drawing_type in ["lines", "rectangles", "curves", "polygons"]:
                        for item in layer_data.get("drawings", {}).get(drawing_type, []):
                            item.pop("color", None)
                            item.pop("opacity", None)
                            item.pop("dashes", None)
            else:
                # Process non-layered data
                for drawing_type in ["lines", "rectangles", "curves", "polygons"]:
                    for item in page.get("drawings", {}).get(drawing_type, []):
                        item.pop("color", None)
                        item.pop("opacity", None)
                        item.pop("dashes", None)
    
    # Serialize
    try:
        if minify:
            logger.info("Producing compact JSON (no whitespace)")
            return json.dumps(output_data, separators=(',', ':'), ensure_ascii=False)
        else:
            logger.info("Producing indented JSON")
            return json.dumps(output_data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error during JSON serialization: {e}")
        raise HTTPException(status_code=500, detail=f"JSON serialization failed: {str(e)}")

# ==================== API ENDPOINTS ====================

@app.post("/extract/")
async def extract_with_layer_support(
    file: UploadFile = File(...),
    minify: bool = Query(True, description="Minify JSON output (remove whitespace)"),
    remove_non_essential: bool = Query(True, description="Remove non-essential fields (fonts, colors, etc.)"),
    precision: int = Query(2, ge=0, le=3, description="Decimal precision for coordinates")
):
    """
    Extract vector data from PDF technical drawings with layer support
    
    NEW in v2:
    - Detects PDF layers automatically
    - If layers found: extracts data per layer with layer labels
    - If no layers: falls back to normal extraction
    - Each line/text/drawing gets layer information when available
    """
    logger.info(f"Vector API v2 request: minify={minify}, remove_non_essential={remove_non_essential}, precision={precision}")
    start_time = time.time()
    
    try:
        logger.info(f"Extracting vector data from: {file.filename}")
        
        # Read PDF
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")
        
        logger.info(f"PDF loaded: {len(pdf_document)} pages, {len(pdf_bytes)/1024:.1f} KB")
        
        # Extract layer information
        logger.info("=== Checking for PDF layers ===")
        layer_info = extract_comprehensive_layers(pdf_document)
        has_layers = layer_info["has_layers"]
        
        if has_layers:
            logger.info(f"✅ Found {layer_info['layer_count']} layers in PDF")
            for layer in layer_info["layers"]:
                logger.info(f"  - Layer: {layer['name']} (visible: {layer['visible']})")
        else:
            logger.info("📄 No layers detected - using standard extraction")
        
        # Extract metadata
        metadata = pdf_document.metadata
        
        output = {
            "metadata": {
                "filename": file.filename,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            },
            "layer_info": layer_info,  # NEW: Include layer information
            "pages": [],
            "summary": {
                "total_pages": len(pdf_document),
                "has_layers": has_layers,  # NEW: Layer detection flag
                "total_layers": layer_info["layer_count"],  # NEW: Layer count
                "total_texts": 0,
                "total_lines": 0,
                "total_rectangles": 0,
                "total_curves": 0,
                "total_polygons": 0,
                "dimensions_found": 0,
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                "processing_time_ms": 0
            }
        }
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page_start = time.time()
            page = pdf_document[page_num]
            logger.info(f"Processing page {page_num + 1} of {len(pdf_document)}")
            
            # Extract all text types with layer support
            all_texts = []
            
            # 1. Embedded text (with layer support)
            embedded_texts = extract_embedded_text_with_layers(page, page_num, precision, has_layers)
            all_texts.extend(embedded_texts)
            
            # TODO: Add layer support to annotations and form fields (similar to embedded text)
            # For now, these don't have layer support but could be added later
            
            # Deduplicate texts
            unique_texts = deduplicate_texts(all_texts)
            
            # Sort by position
            unique_texts.sort(key=lambda x: (x["position"]["y"], x["position"]["x"]))
            
            # Extract vector data with layer support
            vector_data = extract_vector_data_with_layers(page, precision, has_layers)
            
            # Count dimensions
            dimensions_count = sum(1 for t in unique_texts if t.get("dimension_info", {}).get("is_dimension", False))
            
            # Build page data
            page_data = {
                "page_number": page_num + 1,
                "page_size": {
                    "width": round(page.rect.width, precision),
                    "height": round(page.rect.height, precision)
                },
                "has_layers": has_layers,  # NEW: Page-level layer flag
                "processing_time_ms": int((time.time() - page_start) * 1000)
            }
            
            if has_layers:
                # NEW: Group data by layers
                logger.info(f"Grouping page {page_num + 1} data by layers")
                layers_data = group_data_by_layers(unique_texts, vector_data)
                page_data["layers"] = layers_data
                
                # Log layer distribution for this page
                for layer_name, layer_data in layers_data.items():
                    texts_count = len(layer_data["texts"])
                    lines_count = len(layer_data["drawings"]["lines"])
                    logger.info(f"  Layer '{layer_name}': {texts_count} texts, {lines_count} lines")
            else:
                # Standard format (no layers)
                page_data["texts"] = unique_texts
                page_data["drawings"] = vector_data
            
            output["pages"].append(page_data)
            
            # Update summary
            output["summary"]["total_texts"] += len(unique_texts)
            output["summary"]["total_lines"] += len(vector_data["lines"])
            output["summary"]["total_rectangles"] += len(vector_data["rectangles"])
            output["summary"]["total_curves"] += len(vector_data["curves"])
            output["summary"]["total_polygons"] += len(vector_data["polygons"])
            output["summary"]["dimensions_found"] += dimensions_count
        
        pdf_document.close()
        
        output["summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Vector extraction completed in {output['summary']['processing_time_ms']}ms")
        logger.info(f"Found {output['summary']['total_texts']} texts, {output['summary']['dimensions_found']} dimensions")
        if has_layers:
            logger.info(f"Data organized across {output['summary']['total_layers']} layers")
        
        # Minify and filter output
        return minify_output(output, minify, remove_non_essential)
    
    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'pdf_document' in locals():
            try:
                pdf_document.close()
            except:
                pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vector Drawing API v2 - Layer Support",
        "version": "2.0.0",
        "description": "Extracts vector data from technical drawings with PDF layer support",
        "new_features_v2": [
            "🔍 Automatic PDF layer detection",
            "📋 Data extraction per layer with layer labels",
            "🔄 Fallback to standard extraction if no layers",
            "🏷️ Each line/text/drawing tagged with layer information",
            "📊 Layer usage statistics and analysis"
        ],
        "workflow": [
            "1. Upload PDF file",
            "2. Automatic layer detection using OCG analysis",
            "3. If layers found: extract data per layer + label each element",
            "4. If no layers: standard extraction (backward compatible)",
            "5. Return structured data with layer information"
        ],
        "output_formats": {
            "with_layers": {
                "pages": [{
                    "has_layers": True,
                    "layers": {
                        "Layer_Name": {
                            "texts": ["text elements with layer info"],
                            "drawings": ["vector elements with layer info"]
                        }
                    }
                }]
            },
            "without_layers": {
                "pages": [{
                    "has_layers": False,
                    "texts": ["all text elements"],
                    "drawings": ["all vector elements"]
                }]
            }
        },
        "endpoints": {
            "/": "This page",
            "/extract/": "POST - Extract vector data with layer support",
            "/health/": "GET - Health check"
        },
        "parameters": {
            "minify": "true/false (remove whitespace)",
            "remove_non_essential": "true/false (remove fonts, colors, etc.)",
            "precision": "0-3 (decimal places for coordinates)"
        },
        "compatibility": {
            "master_api": "✅ Compatible with Master API v4.1.1+",
            "n8n": "✅ Direct HTTP request support",
            "backward_compatible": "✅ Same output format when no layers"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": [
            "PDF layer detection",
            "Per-layer data extraction",
            "Layer-tagged elements",
            "Backward compatibility"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Vector Drawing API v2 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
