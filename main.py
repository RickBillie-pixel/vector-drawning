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
import pytesseract
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("merged_extraction_api")

# Constants
PORT = int(os.environ.get("PORT", 10000))
MAX_PAGE_SIZE = 10000
MIN_LINE_LENGTH = 0.1
DPI_HIGH = 300
DPI_LOW = 150
ENABLE_OCR = os.environ.get("ENABLE_OCR", "true").lower() == "true"

app = FastAPI(
    title="Merged Extraction API",
    description="Extracts all text and vector data from technical drawings with minification",
    version="1.0.0",
)

# Add GZip middleware for automatic compression if client supports it
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Shared utility functions
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
                                "source": "embedded"
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
                        "source": "annotation"
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
                        "source": "form_field"
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
                            **path_info
                        }
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
                        curves.append(curve_data)
                    except Exception as e:
                        logger.warning(f"Error processing curve: {e}")
                
                elif item_type == "qu" and len(points) >= 3:  # Polygon
                    polygon_data = {
                        "type": "polygon",
                        "points": [point_to_dict(p, precision) for p in points],
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

def perform_ocr(page: fitz.Page, page_num: int, precision: int = 2) -> List[Dict]:
    """Perform OCR on page if enabled"""
    if not ENABLE_OCR:
        return []
    
    ocr_texts = []
    
    try:
        mat = fitz.Matrix(DPI_HIGH/72, DPI_HIGH/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to grayscale
        img_gray = img.convert('L')
        
        # Perform OCR
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Scale factors
        scale_x = page.rect.width / img.width
        scale_y = page.rect.height / img.height
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text and data['conf'][i] > 20:
                x = data['left'][i] * scale_x
                y = data['top'][i] * scale_y
                w = data['width'][i] * scale_x
                h = data['height'][i] * scale_y
                
                text_data = {
                    "text": text,
                    "position": {
                        "x": round(x, precision),
                        "y": round(y, precision)
                    },
                    "bbox": {
                        "x0": round(x, precision),
                        "y0": round(y, precision),
                        "x1": round(x + w, precision),
                        "y1": round(y + h, precision),
                        "width": round(w, precision),
                        "height": round(h, precision)
                    },
                    "page_number": page_num + 1,
                    "source": "ocr",
                    "confidence": data['conf'][i]
                }
                
                # Add dimension info if detected
                dim_info = extract_dimension_info(text)
                if dim_info["is_dimension"]:
                    text_data["dimension_info"] = dim_info
                
                ocr_texts.append(text_data)
    
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
    
    return ocr_texts

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
        # Remove non-critical metadata
        if "metadata" in output_data:
            critical_metadata = {
                "filename": output_data["metadata"].get("filename", ""),
                "total_pages": output_data["summary"].get("total_pages", 0)
            }
            output_data["metadata"] = critical_metadata
        
        # Remove processing times from summary
        if "summary" in output_data:
            output_data["summary"].pop("processing_time_ms", None)
            output_data["summary"].pop("file_size_mb", None)
        
        # Remove non-essential fields from pages
        for page in output_data.get("pages", []):
            page.pop("processing_time_ms", None)
            page.pop("fonts", None)
            page.pop("layers", None)
            
            # Remove from texts
            for text in page.get("texts", []):
                text.pop("font", None)      # Remove fonts
                text.pop("color", None)     # Remove colors
                text.pop("flags", None)
                text.pop("size", None)
                text.pop("confidence", None)
            
            # Remove from drawings
            for drawing_type in ["lines", "rectangles", "curves", "polygons"]:
                for item in page.get("drawings", {}).get(drawing_type, []):
                    item.pop("color", None)  # Remove colors
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

@app.post("/extract/")
async def extract_all(
    file: UploadFile = File(...),
    minify: bool = Query(True, description="Minify JSON output (remove whitespace)"),
    remove_non_essential: bool = Query(True, description="Remove non-essential fields (fonts, colors, etc.)"),
    enable_ocr: bool = Query(ENABLE_OCR, description="Enable OCR processing"),
    precision: int = Query(2, ge=0, le=3, description="Decimal precision for coordinates")
):
    """
    Extract all data from PDF technical drawings
    Combines text extraction, vector extraction, and OCR with configurable output minification
    """
    logger.info(f"Received request with minify={minify}, remove_non_essential={remove_non_essential}, enable_ocr={enable_ocr}, precision={precision}")
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
                "processing_time_ms": 0
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
            
            # 4. OCR (if enabled)
            if enable_ocr:
                ocr_texts = perform_ocr(page, page_num, precision)
                all_texts.extend(ocr_texts)
            
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
        
        logger.info(f"Extraction completed in {output['summary']['processing_time_ms']}ms")
        logger.info(f"Found {output['summary']['total_texts']} texts, {output['summary']['dimensions_found']} dimensions")
        
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
        "message": "Merged Extraction API",
        "version": "1.0.0",
        "description": "Extracts text and vector data from technical drawings with minification",
        "endpoints": {
            "/": "This page",
            "/extract/": "POST - Extract all data from PDF",
            "/health/": "GET - Health check"
        },
        "parameters": {
            "minify": "true/false (remove whitespace)",
            "remove_non_essential": "true/false (remove fonts, colors, etc.)",
            "enable_ocr": "true/false (enable OCR processing)",
            "precision": "0-3 (decimal places for coordinates)"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "ocr_enabled": ENABLE_OCR
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
