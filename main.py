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
import fitz  # PyMuPDF
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("merged_extraction_api")

# Constants
PORT = int(os.environ.get("PORT", 10000))
MAX_PAGE_SIZE = 10000
MIN_LINE_LENGTH = 0.1

app = FastAPI(
    title="Merged Extraction API",
    description="Extracts all text and vector data from technical drawings with minification and coordinate info",
    version="1.0.0",
)

# Add GZip middleware for automatic compression if client supports it
app.add_middleware(GZipMiddleware, minimum_size=1000)

class MyPdfModel:
    """Enhanced PDF model that calculates real dimensions and coordinate systems"""
    
    def __init__(self, pdf_document: fitz.Document):
        self.pdf_document = pdf_document
        self.page_dimensions = self._calculate_page_dimensions()
        self.total_dimensions = self._calculate_total_dimensions()
    
    def _calculate_page_dimensions(self) -> List[Dict]:
        """Calculate dimensions for each page"""
        pages_info = []
        
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document[page_num]
            rect = page.rect
            
            # Page dimensions in points (PDF native)
            page_width = rect.width
            page_height = rect.height
            
            # Convert to pixels (assuming 150 DPI)
            page_width_pixels = int((page_width * 150) / 72)
            page_height_pixels = int((page_height * 150) / 72)
            
            # Convert to millimeters
            page_width_mm = page_width * 0.352778
            page_height_mm = page_height * 0.352778
            
            page_info = {
                "page_number": page_num + 1,
                "dimensions_points": {
                    "width": round(page_width, 2),
                    "height": round(page_height, 2),
                    "x0": round(rect.x0, 2),
                    "y0": round(rect.y0, 2),
                    "x1": round(rect.x1, 2),
                    "y1": round(rect.y1, 2)
                },
                "dimensions_pixels": {
                    "width": page_width_pixels,
                    "height": page_height_pixels,
                    "dpi": 150
                },
                "dimensions_mm": {
                    "width": round(page_width_mm, 2),
                    "height": round(page_height_mm, 2)
                }
            }
            
            pages_info.append(page_info)
        
        return pages_info
    
    def _calculate_total_dimensions(self) -> Dict:
        """Calculate total PDF dimensions"""
        if not self.page_dimensions:
            return {}
        
        total_width_points = sum(p["dimensions_points"]["width"] for p in self.page_dimensions)
        total_height_points = sum(p["dimensions_points"]["height"] for p in self.page_dimensions)
        max_width_points = max(p["dimensions_points"]["width"] for p in self.page_dimensions)
        max_height_points = max(p["dimensions_points"]["height"] for p in self.page_dimensions)
        
        return {
            "total_pages": len(self.page_dimensions),
            "total_width_points": round(total_width_points, 2),
            "total_height_points": round(total_height_points, 2),
            "max_width_points": round(max_width_points, 2),
            "max_height_points": round(max_height_points, 2),
            "total_width_pixels": int((total_width_points * 150) / 72),
            "total_height_pixels": int((total_height_points * 150) / 72),
            "max_width_pixels": int((max_width_points * 150) / 72),
            "max_height_pixels": int((max_height_points * 150) / 72),
            "total_width_mm": round(total_width_points * 0.352778, 2),
            "total_height_mm": round(total_height_points * 0.352778, 2),
            "max_width_mm": round(max_width_points * 0.352778, 2),
            "max_height_mm": round(max_height_points * 0.352778, 2)
        }

# Shared utility functions (keeping original functionality)
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

def extract_embedded_text(page: fitz.Page, page_num: int, precision: int = 2) -> List[Dict]:
    """Extract embedded text from PDF"""
    embedded_texts = []
    
    try:
        text_dict = page.get_text("dict")
        
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
                                "source": "embedded",
                                # NEW: Add coordinates for filter API
                                "coordinates": {
                                    "x": round(bbox[0], precision),
                                    "y": round(bbox[1], precision),
                                    "center_x": round((bbox[0] + bbox[2]) / 2, precision),
                                    "center_y": round((bbox[1] + bbox[3]) / 2, precision)
                                }
                            }
                            
                            # Add dimension info if detected
                            dim_info = extract_dimension_info(text)
                            if dim_info["is_dimension"]:
                                text_data["dimension_info"] = dim_info
                            
                            embedded_texts.append(text_data)
    
    except Exception as e:
        logger.warning(f"Failed to extract embedded text: {e}")
    
    return embedded_texts

def extract_annotations(page: fitz.Page, page_num: int, precision: int = 2) -> List[Dict]:
    """Extract annotations from PDF"""
    annotation_texts = []
    
    try:
        for annot in page.annots():
            if annot:
                content = annot.info.get("content", "")
                if not content and hasattr(annot, "get_text"):
                    try:
                        content = annot.get_text()
                    except:
                        pass
                
                if content and content.strip():
                    rect = annot.rect
                    
                    text_data = {
                        "text": content.strip(),
                        "position": {
                            "x": round(rect.x0, precision),
                            "y": round(rect.y0, precision)
                        },
                        "bbox": rect_to_dict(rect, precision),
                        "page_number": page_num + 1,
                        "source": "annotation",
                        # NEW: Add coordinates for filter API
                        "coordinates": {
                            "x": round(rect.x0, precision),
                            "y": round(rect.y0, precision),
                            "center_x": round((rect.x0 + rect.x1) / 2, precision),
                            "center_y": round((rect.y0 + rect.y1) / 2, precision)
                        }
                    }
                    
                    # Add dimension info if detected
                    dim_info = extract_dimension_info(content)
                    if dim_info["is_dimension"]:
                        text_data["dimension_info"] = dim_info
                    
                    annotation_texts.append(text_data)
    
    except Exception as e:
        logger.warning(f"Failed to extract annotations: {e}")
    
    return annotation_texts

def extract_form_fields(page: fitz.Page, page_num: int, precision: int = 2) -> List[Dict]:
    """Extract form fields from PDF"""
    fields = []
    
    try:
        widgets = page.widgets()
        if widgets:
            for widget in widgets:
                text = widget.field_value or widget.field_display or ""
                if text and text.strip():
                    field_data = {
                        "text": text.strip(),
                        "position": {
                            "x": round(widget.rect.x0, precision),
                            "y": round(widget.rect.y0, precision)
                        },
                        "bbox": rect_to_dict(widget.rect, precision),
                        "page_number": page_num + 1,
                        "source": "form_field",
                        # NEW: Add coordinates for filter API
                        "coordinates": {
                            "x": round(widget.rect.x0, precision),
                            "y": round(widget.rect.y0, precision),
                            "center_x": round((widget.rect.x0 + widget.rect.x1) / 2, precision),
                            "center_y": round((widget.rect.y0 + widget.rect.y1) / 2, precision)
                        }
                    }
                    
                    # Add dimension info if detected
                    dim_info = extract_dimension_info(text)
                    if dim_info["is_dimension"]:
                        field_data["dimension_info"] = dim_info
                    
                    fields.append(field_data)
    
    except Exception as e:
        logger.warning(f"Error extracting form fields: {e}")
    
    return fields

def extract_vector_data(page: fitz.Page, precision: int = 2) -> Dict[str, List]:
    """Extract vector drawings from page"""
    lines = []
    rectangles = []
    curves = []
    polygons = []
    
    try:
        drawings = page.get_drawings()
        for path in drawings:
            path_info = extract_path_data(path, precision)
            
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
                            # NEW: Add coordinates for filter API
                            "coordinates": {
                                "start_x": p1["x"],
                                "start_y": p1["y"],
                                "end_x": p2["x"],
                                "end_y": p2["y"],
                                "center_x": round((p1["x"] + p2["x"]) / 2, precision),
                                "center_y": round((p1["y"] + p2["y"]) / 2, precision)
                            },
                            **path_info
                        }
                        lines.append(line_data)
                
                elif item_type == "re" and points:  # Rectangle
                    rect = rect_to_dict(points[0], precision)
                    rect_area = rect["width"] * rect["height"]
                    if rect_area >= 1:
                        rect_data = {
                            "type": "rectangle",
                            "rect": rect,
                            "area": round(rect_area, precision),
                            # NEW: Add coordinates for filter API
                            "coordinates": {
                                "x": rect["x0"],
                                "y": rect["y0"],
                                "center_x": round((rect["x0"] + rect["x1"]) / 2, precision),
                                "center_y": round((rect["y0"] + rect["y1"]) / 2, precision),
                                "width": rect["width"],
                                "height": rect["height"]
                            },
                            **path_info
                        }
                        rectangles.append(rect_data)
                
                elif item_type == "c" and points:  # Curve/Circle
                    try:
                        if len(points) >= 4:
                            curve_points = [point_to_dict(p, precision) for p in points[:4]]
                            # Calculate center point for curves
                            center_x = sum(p["x"] for p in curve_points) / len(curve_points)
                            center_y = sum(p["y"] for p in curve_points) / len(curve_points)
                            
                            curve_data = {
                                "type": "bezier",
                                "points": curve_points,
                                # NEW: Add coordinates for filter API
                                "coordinates": {
                                    "center_x": round(center_x, precision),
                                    "center_y": round(center_y, precision),
                                    "control_points": curve_points
                                },
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
                                # NEW: Add coordinates for filter API
                                "coordinates": {
                                    "center_x": center["x"],
                                    "center_y": center["y"],
                                    "radius": round(radius, precision),
                                    "bounds_x0": round(center["x"] - radius, precision),
                                    "bounds_y0": round(center["y"] - radius, precision),
                                    "bounds_x1": round(center["x"] + radius, precision),
                                    "bounds_y1": round(center["y"] + radius, precision)
                                },
                                **path_info
                            }
                        curves.append(curve_data)
                    except Exception as e:
                        logger.warning(f"Error processing curve: {e}")
                
                elif item_type == "qu" and len(points) >= 3:  # Polygon
                    polygon_points = [point_to_dict(p, precision) for p in points]
                    # Calculate center point for polygon
                    center_x = sum(p["x"] for p in polygon_points) / len(polygon_points)
                    center_y = sum(p["y"] for p in polygon_points) / len(polygon_points)
                    
                    polygon_data = {
                        "type": "polygon",
                        "points": polygon_points,
                        # NEW: Add coordinates for filter API
                        "coordinates": {
                            "center_x": round(center_x, precision),
                            "center_y": round(center_y, precision),
                            "vertices": polygon_points,
                            "vertex_count": len(polygon_points)
                        },
                        **path_info
                    }
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

def calculate_coordinate_bounds(data: Dict) -> Dict[str, Any]:
    """Calculate actual min/max coordinates from extracted data"""
    all_x_coords = []
    all_y_coords = []
    
    for page in data.get("pages", []):
        # Collect coordinates from texts
        for text in page.get("texts", []):
            if "coordinates" in text:
                all_x_coords.extend([text["coordinates"]["x"], text["coordinates"]["center_x"]])
                all_y_coords.extend([text["coordinates"]["y"], text["coordinates"]["center_y"]])
            # Also from bbox
            if "bbox" in text:
                bbox = text["bbox"]
                all_x_coords.extend([bbox["x0"], bbox["x1"]])
                all_y_coords.extend([bbox["y0"], bbox["y1"]])
        
        # Collect coordinates from drawings
        drawings = page.get("drawings", {})
        
        # Lines
        for line in drawings.get("lines", []):
            if "coordinates" in line:
                coords = line["coordinates"]
                all_x_coords.extend([coords["start_x"], coords["end_x"], coords["center_x"]])
                all_y_coords.extend([coords["start_y"], coords["end_y"], coords["center_y"]])
        
        # Rectangles
        for rect in drawings.get("rectangles", []):
            if "coordinates" in rect:
                coords = rect["coordinates"]
                all_x_coords.extend([coords["x"], coords["center_x"], coords["x"] + coords["width"]])
                all_y_coords.extend([coords["y"], coords["center_y"], coords["y"] + coords["height"]])
        
        # Curves/Circles
        for curve in drawings.get("curves", []):
            if "coordinates" in curve:
                coords = curve["coordinates"]
                all_x_coords.append(coords["center_x"])
                all_y_coords.append(coords["center_y"])
                if "bounds_x0" in coords:
                    all_x_coords.extend([coords["bounds_x0"], coords["bounds_x1"]])
                    all_y_coords.extend([coords["bounds_y0"], coords["bounds_y1"]])
        
        # Polygons
        for polygon in drawings.get("polygons", []):
            if "coordinates" in polygon:
                coords = polygon["coordinates"]
                all_x_coords.append(coords["center_x"])
                all_y_coords.append(coords["center_y"])
                for vertex in coords.get("vertices", []):
                    all_x_coords.append(vertex["x"])
                    all_y_coords.append(vertex["y"])
    
    if all_x_coords and all_y_coords:
        return {
            "coordinate_bounds": {
                "min_x": round(min(all_x_coords), 2),
                "max_x": round(max(all_x_coords), 2),
                "min_y": round(min(all_y_coords), 2),
                "max_y": round(max(all_y_coords), 2),
                "width": round(max(all_x_coords) - min(all_x_coords), 2),
                "height": round(max(all_y_coords) - min(all_y_coords), 2),
                "total_elements": len(all_x_coords)
            }
        }
    else:
        return {
            "coordinate_bounds": {
                "min_x": 0.0,
                "max_x": 0.0,
                "min_y": 0.0,
                "max_y": 0.0,
                "width": 0.0,
                "height": 0.0,
                "total_elements": 0
            }
        }

def minify_output(data: Dict, minify: bool = True, remove_non_essential: bool = True) -> str:
    """
    Minify JSON output to a single-line string, removing all whitespace.
    Optionally removes non-essential fields (fonts, colors, etc.).
    
    Args:
        data (Dict): The JSON data to minify.
        minify (bool): If True, produce single-line JSON (no whitespace).
        remove_non_essential (bool): If True, remove fonts, colors, etc.
    
    Returns:
        str: Minified JSON string.
    """
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
                "producer": output_data["metadata"].get("producer", ""),
                # NEW: Keep PDF dimensions for filter API
                "pdf_dimensions": output_data["metadata"].get("pdf_dimensions", {}),
                "page_dimensions": output_data["metadata"].get("page_dimensions", [])
                # Skip creation_date and modification_date to save space
            }
            output_data["metadata"] = essential_metadata
        
        # Keep coordinate bounds in summary (important for filtering!)
        if "summary" in output_data:
            # Remove processing times but keep coordinate bounds
            output_data["summary"].pop("processing_time_ms", None)
            output_data["summary"].pop("file_size_mb", None)
            # Keep coordinate_bounds for filter API!
        
        # Remove non-essential fields from pages
        for page in output_data.get("pages", []):
            page.pop("processing_time_ms", None)
            page.pop("fonts", None)
            page.pop("layers", None)
            
            # Remove from texts (but keep coordinates!)
            for text in page.get("texts", []):
                text.pop("font", None)      # Remove fonts
                text.pop("color", None)     # Remove colors
                text.pop("flags", None)
                text.pop("size", None)
                text.pop("confidence", None)
                # Keep coordinates field for filter API
            
            # Remove from drawings (but keep coordinates!)
            for drawing_type in ["lines", "rectangles", "curves", "polygons"]:
                for item in page.get("drawings", {}).get(drawing_type, []):
                    item.pop("color", None)  # Remove colors
                    item.pop("opacity", None)
                    item.pop("dashes", None)
                    # Keep coordinates field for filter API
    
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

@app.post("/extract/")
async def extract_all(
    file: UploadFile = File(...),
    minify: bool = Query(True, description="Minify JSON output (remove whitespace)"),
    remove_non_essential: bool = Query(True, description="Remove non-essential fields (fonts, colors, etc.)"),
    precision: int = Query(2, ge=0, le=3, description="Decimal precision for coordinates")
):
    """
    Extract all data from PDF technical drawings
    Combines text extraction and vector extraction with configurable output minification
    NOW WITH: Enhanced coordinate information for filter API usage
    """
    logger.info(f"Received request with minify={minify}, remove_non_essential={remove_non_essential}, precision={precision}")
    start_time = time.time()
    
    try:
        logger.info(f"Extracting data from: {file.filename}")
        
        # Read PDF
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")
        
        logger.info(f"PDF loaded: {len(pdf_document)} pages, {len(pdf_bytes)/1024:.1f} KB")
        
        # NEW: Initialize MyPdfModel to get real dimensions
        pdf_model = MyPdfModel(pdf_document)
        
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
                "modification_date": metadata.get("modDate", ""),
                # NEW: Add PDF dimensions for filter API
                "pdf_dimensions": pdf_model.total_dimensions,
                "page_dimensions": pdf_model.page_dimensions
            },
            "pages": [],
            "summary": {
                "total_pages": len(pdf_document),
                "total_texts": 0,
                "total_lines": 0,
                "total_rectangles": 0,
                "total_curves": 0,
                "total_polygons": 0,
                "dimensions_found": 0,
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                "processing_time_ms": 0,
                # NEW: Add coordinate system info
                "coordinate_system": "pdf_points_with_filter_coords"
            }
        }
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page_start = time.time()
            page = pdf_document[page_num]
            logger.info(f"Processing page {page_num + 1} of {len(pdf_document)}")
            
            # Extract all text types
            all_texts = []
            
            # 1. Embedded text
            embedded_texts = extract_embedded_text(page, page_num, precision)
            all_texts.extend(embedded_texts)
            
            # 2. Annotations
            annotation_texts = extract_annotations(page, page_num, precision)
            all_texts.extend(annotation_texts)
            
            # 3. Form fields
            form_texts = extract_form_fields(page, page_num, precision)
            all_texts.extend(form_texts)
            
            # Deduplicate texts
            unique_texts = deduplicate_texts(all_texts)
            
            # Sort by position
            unique_texts.sort(key=lambda x: (x["position"]["y"], x["position"]["x"]))
            
            # Extract vector data
            vector_data = extract_vector_data(page, precision)
            
            # Count dimensions
            dimensions_count = sum(1 for t in unique_texts if t.get("dimension_info", {}).get("is_dimension", False))
            
            # Build page data
            page_data = {
                "page_number": page_num + 1,
                "page_size": {
                    "width": round(page.rect.width, precision),
                    "height": round(page.rect.height, precision)
                },
                # NEW: Add page dimensions from MyPdfModel
                "page_dimensions": pdf_model.page_dimensions[page_num] if page_num < len(pdf_model.page_dimensions) else {},
                "texts": unique_texts,
                "drawings": vector_data,
                "processing_time_ms": int((time.time() - page_start) * 1000)
            }
            
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
        
        # NEW: Calculate actual coordinate bounds from extracted data
        coordinate_bounds = calculate_coordinate_bounds(output)
        output["summary"].update(coordinate_bounds)
        
        logger.info(f"Extraction completed in {output['summary']['processing_time_ms']}ms")
        logger.info(f"Found {output['summary']['total_texts']} texts, {output['summary']['dimensions_found']} dimensions")
        logger.info(f"PDF total dimensions: {pdf_model.total_dimensions}")
        logger.info(f"Actual coordinate bounds: {coordinate_bounds['coordinate_bounds']}")
        
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
        "message": "Merged Extraction API with Enhanced Coordinates",
        "version": "1.0.0",
        "description": "Extracts text and vector data from technical drawings with minification and coordinate info for filter API",
        "endpoints": {
            "/": "This page",
            "/extract/": "POST - Extract all data from PDF with coordinates",
            "/health/": "GET - Health check"
        },
        "parameters": {
            "minify": "true/false (remove whitespace)",
            "remove_non_essential": "true/false (remove fonts, colors, etc.)",
            "precision": "0-3 (decimal places for coordinates)"
        },
        "new_features": {
            "coordinates": "All elements now include x,y coordinates for filter API",
            "pdf_dimensions": "Real PDF dimensions in points, pixels, and millimeters",
            "page_dimensions": "Individual page dimensions for each page",
            "coordinate_bounds": "Actual min/max X,Y coordinates from extracted data",
            "filter_ready": "Output optimized for coordinate-based filtering"
        },
        "coordinate_fields": {
            "texts": "coordinates.x, coordinates.y, coordinates.center_x, coordinates.center_y",
            "lines": "coordinates.start_x, coordinates.start_y, coordinates.end_x, coordinates.end_y, coordinates.center_x, coordinates.center_y",
            "rectangles": "coordinates.x, coordinates.y, coordinates.center_x, coordinates.center_y, coordinates.width, coordinates.height",
            "circles": "coordinates.center_x, coordinates.center_y, coordinates.radius, coordinates.bounds_x0, coordinates.bounds_y0, coordinates.bounds_x1, coordinates.bounds_y1",
            "polygons": "coordinates.center_x, coordinates.center_y, coordinates.vertices, coordinates.vertex_count"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": [
            "Original functionality preserved",
            "Enhanced with coordinate information",
            "PDF dimension extraction",
            "Filter API ready output"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {PORT}")
    logger.info("Enhanced with coordinate information for filter API usage")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
