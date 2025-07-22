import os
import json
import logging
import uuid
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", 10000))

app = FastAPI(
    title="Pre-Scale API",
    description="Filters dimension lines and associates them with OCR text for scale calculation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Point(BaseModel):
    x: float
    y: float

class TextData(BaseModel):
    text: str
    position: Dict[str, float]
    bbox: Dict[str, float]
    page_number: Optional[int] = None
    source: Optional[str] = None

class DimensionLine(BaseModel):
    line: Dict[str, Any]
    orientation: str
    midpoint: Point
    associated_text: Optional[TextData] = None
    distance_to_text: Optional[float] = None
    confidence_score: float = 0.0

class PreScaleOutput(BaseModel):
    metadata: Dict[str, Any]
    dimension_lines: List[DimensionLine]
    summary: Dict[str, Any]

# Constants
MIN_LINE_LENGTH = 40  # Minimum length for dimension lines
MAX_TEXT_DISTANCE = 50  # Maximum distance between line midpoint and text
DIMENSION_PATTERNS = [
    r'^\d+$',  # Simple numbers: 2500, 3000
    r'^\d+\.\d+$',  # Decimals: 25.5, 3.6
    r'^\d+(?:\.\d+)?\s*(?:mm|cm|m|MM|CM|M)$',  # With units: 2500mm, 3.6m
    r'^\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?$',  # Dimensions: 2500x3000
    r'^[\d\s]+$',  # Numbers with spaces: 2 500
]

def parse_input_data(contents: str) -> Dict:
    """Parse the input JSON string, handling potential nested JSON."""
    try:
        data = json.loads(contents)
        if isinstance(data, str):
            logger.warning("Input is a string, attempting to parse again")
            data = json.loads(data)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {str(e)}")

def calculate_orientation(p1: Dict[str, float], p2: Dict[str, float]) -> str:
    """Calculate line orientation based on endpoints."""
    dx = abs(p2['x'] - p1['x'])
    dy = abs(p2['y'] - p1['y'])
    
    if dx > dy:
        return "horizontal"
    elif dy > dx:
        return "vertical"
    else:
        return "diagonal"

def calculate_midpoint(p1: Dict[str, float], p2: Dict[str, float]) -> Point:
    """Calculate midpoint of a line."""
    return Point(
        x=(p1['x'] + p2['x']) / 2,
        y=(p1['y'] + p2['y']) / 2
    )

def calculate_distance(point1: Point, point2: Dict[str, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2['x'])**2 + (point1.y - point2['y'])**2)

def get_text_center(bbox: Dict[str, float]) -> Dict[str, float]:
    """Get center point of a bounding box."""
    return {
        'x': (bbox.get('x0', 0) + bbox.get('x1', 0)) / 2,
        'y': (bbox.get('y0', 0) + bbox.get('y1', 0)) / 2
    }

def is_dimension_text(text: str) -> bool:
    """Check if text matches dimension patterns."""
    cleaned_text = text.strip()
    for pattern in DIMENSION_PATTERNS:
        if re.match(pattern, cleaned_text, re.IGNORECASE):
            return True
    return False

def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from dimension text."""
    # Remove units and spaces
    cleaned = re.sub(r'[^\d.,]', '', text)
    cleaned = cleaned.replace(',', '.')
    try:
        return float(cleaned)
    except:
        return None

def find_associated_text(midpoint: Point, texts: List[Dict], max_distance: float = MAX_TEXT_DISTANCE) -> Tuple[Optional[Dict], Optional[float]]:
    """Find the closest dimension text to a line's midpoint."""
    best_text = None
    best_distance = float('inf')
    
    for text_data in texts:
        # Skip if not a dimension text
        if not is_dimension_text(text_data.get('text', '')):
            continue
        
        # Calculate distance to text center
        text_center = get_text_center(text_data.get('bbox', {}))
        distance = calculate_distance(midpoint, text_center)
        
        # Update if this is the closest dimension text
        if distance < max_distance and distance < best_distance:
            best_text = text_data
            best_distance = distance
    
    return (best_text, best_distance) if best_text else (None, None)

def calculate_confidence(line_length: float, orientation: str, has_text: bool, text_distance: Optional[float]) -> float:
    """Calculate confidence score for a dimension line."""
    score = 0.0
    
    # Length factor (longer lines are more likely to be dimensions)
    if line_length > 100:
        score += 0.3
    elif line_length > 60:
        score += 0.2
    
    # Orientation factor (horizontal/vertical preferred)
    if orientation in ["horizontal", "vertical"]:
        score += 0.3
    
    # Text association factor
    if has_text:
        score += 0.3
        # Distance factor
        if text_distance and text_distance < 20:
            score += 0.1
    
    return min(score, 1.0)

@app.post("/pre-filter/")
async def pre_scale_filter(file: UploadFile):
    """Filter dimension lines and associate them with OCR text for scale calculation"""
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Read and parse the uploaded JSON file
        contents = (await file.read()).decode('utf-8')
        input_data = parse_input_data(contents)
        
        # Save input for debugging
        debug_path = None
        try:
            debug_path = f"/tmp/input_{uuid.uuid4()}.json"
            with open(debug_path, 'w') as f:
                json.dump(input_data, f, indent=2)
            logger.info(f"Saved input for debugging to {debug_path}")
        except Exception as e:
            logger.warning(f"Failed to save debug input: {e}")
        
        # Validate structure
        if not input_data.get('pages') or not input_data.get('metadata'):
            logger.error("Invalid structure: missing pages or metadata")
            raise HTTPException(status_code=400, detail="Invalid input structure")
        
        # Process dimension lines
        all_dimension_lines = []
        total_lines_processed = 0
        total_texts_processed = 0
        
        for page in input_data['pages']:
            page_number = page.get('page_number', 1)
            
            # Get lines and texts
            drawings = page.get('drawings', {})
            lines = drawings.get('lines', [])
            texts = page.get('texts', [])
            
            if not isinstance(lines, list):
                lines = []
            if not isinstance(texts, list):
                texts = []
            
            total_texts_processed += len(texts)
            
            # Process each line
            for line in lines:
                total_lines_processed += 1
                
                # Skip if not a proper line
                if line.get('type') != 'line' or 'p1' not in line or 'p2' not in line:
                    continue
                
                # Check minimum length
                line_length = line.get('length', 0)
                if line_length < MIN_LINE_LENGTH:
                    continue
                
                # Calculate orientation
                orientation = calculate_orientation(line['p1'], line['p2'])
                
                # Calculate midpoint
                midpoint = calculate_midpoint(line['p1'], line['p2'])
                
                # Find associated text
                associated_text, distance = find_associated_text(midpoint, texts)
                
                # Calculate confidence
                confidence = calculate_confidence(
                    line_length,
                    orientation,
                    associated_text is not None,
                    distance
                )
                
                # Create dimension line entry
                dimension_line = {
                    "line": line,
                    "orientation": orientation,
                    "midpoint": midpoint.dict(),
                    "associated_text": associated_text,
                    "distance_to_text": round(distance, 2) if distance else None,
                    "confidence_score": round(confidence, 2),
                    "page_number": page_number
                }
                
                # Only include lines with reasonable confidence
                if confidence > 0.3 or (associated_text and orientation in ["horizontal", "vertical"]):
                    all_dimension_lines.append(dimension_line)
        
        # Sort by confidence score
        all_dimension_lines.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Prepare output
        output = {
            "metadata": input_data.get('metadata', {}),
            "dimension_lines": all_dimension_lines,
            "summary": {
                "total_dimension_lines": len(all_dimension_lines),
                "total_lines_processed": total_lines_processed,
                "total_texts_processed": total_texts_processed,
                "lines_with_text": sum(1 for d in all_dimension_lines if d['associated_text']),
                "horizontal_lines": sum(1 for d in all_dimension_lines if d['orientation'] == 'horizontal'),
                "vertical_lines": sum(1 for d in all_dimension_lines if d['orientation'] == 'vertical'),
                "diagonal_lines": sum(1 for d in all_dimension_lines if d['orientation'] == 'diagonal'),
                "avg_confidence": round(sum(d['confidence_score'] for d in all_dimension_lines) / len(all_dimension_lines), 2) if all_dimension_lines else 0
            }
        }
        
        logger.info(f"Pre-Scale filtering complete: {len(all_dimension_lines)} dimension lines found from {total_lines_processed} total lines")
        logger.info(f"Lines with associated text: {output['summary']['lines_with_text']}")
        
        return output
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during pre-scale filtering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health/")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "api_type": "pre-scale",
        "min_line_length": MIN_LINE_LENGTH,
        "max_text_distance": MAX_TEXT_DISTANCE
    }

@app.get("/")
async def root():
    return {
        "title": "Pre-Scale API",
        "description": "Filters dimension lines and associates them with OCR text for scale calculation",
        "version": "2.0.0",
        "endpoints": {
            "/": "This page",
            "/pre-filter/": "POST - Process vector and OCR data",
            "/health/": "GET - Health check"
        },
        "rules": {
            "min_line_length": MIN_LINE_LENGTH,
            "max_text_distance": MAX_TEXT_DISTANCE,
            "orientations": ["horizontal", "vertical", "diagonal"],
            "dimension_patterns": DIMENSION_PATTERNS
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
