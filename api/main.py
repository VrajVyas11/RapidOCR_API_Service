import time
import gc
import psutil
import math
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict, Any

# Register support for additional formats like AVIF and HEIF
try:
    from pillow_heif import register_heif_opener, register_avif_opener
    register_heif_opener()
    register_avif_opener()
    print("[INFO] pillow_heif registered for AVIF/HEIF support")
except ImportError:
    print("[WARNING] pillow_heif not installed, AVIF and HEIF support may be limited")

app = FastAPI(title="Vercel RapidOCR API")

# Global OCR instance
_reader = None

class InitRequest(BaseModel):
    languages: Optional[str] = "en"

def log_memory_usage(stage: str = ""):
    if stage:
        print(f"[MEMORY {stage}]")
    try:
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"[MEMORY] Current process: {current_mem:.2f} MB")
    except:
        print(f"[MEMORY] Could not get memory info")

def get_reader():
    global _reader
    if _reader is None:
        log_memory_usage("Creating Reader")

        from rapidocr_onnxruntime import RapidOCR
        try:
            import wordninja
        except ImportError:
            print("[WARNING] wordninja not available, using basic splitting")
            wordninja = None

        _reader = {
            "ocr": RapidOCR(),
            "splitter": wordninja
        }

        log_memory_usage("Reader Created")
    return _reader

def process_image_bytes(image_bytes):
    """Convert uploaded image bytes to numpy array for RapidOCR"""
    try:
        # Open image with PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles AVIF, WebP, etc.)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB format)
        numpy_image = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV/RapidOCR
        bgr_image = numpy_image[:, :, ::-1]
        
        return bgr_image
        
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")

def calculate_bbox_properties(bbox):
    """Calculate center, width, height from bbox coordinates"""
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    width = right - left
    height = bottom - top
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'left': left,
        'right': right,
        'top': top,
        'bottom': bottom,
        'width': width,
        'height': height
    }

def calculate_distance(bbox1_props, bbox2_props):
    """Calculate distance between two bboxes"""
    dx = bbox1_props['center_x'] - bbox2_props['center_x']
    dy = bbox1_props['center_y'] - bbox2_props['center_y']
    return math.sqrt(dx*dx + dy*dy)

def is_horizontally_aligned(bbox1_props, bbox2_props, tolerance_factor=0.3):
    """Check if two bboxes are roughly horizontally aligned"""
    overlap_top = max(bbox1_props['top'], bbox2_props['top'])
    overlap_bottom = min(bbox1_props['bottom'], bbox2_props['bottom'])
    overlap_height = max(0, overlap_bottom - overlap_top)
    
    min_height = min(bbox1_props['height'], bbox2_props['height'])
    return overlap_height > (min_height * tolerance_factor)

def is_vertically_aligned(bbox1_props, bbox2_props, tolerance_factor=0.3):
    """Check if two bboxes are roughly vertically aligned"""
    overlap_left = max(bbox1_props['left'], bbox2_props['left'])
    overlap_right = min(bbox1_props['right'], bbox2_props['right'])
    overlap_width = max(0, overlap_right - overlap_left)
    
    min_width = min(bbox1_props['width'], bbox2_props['width'])
    return overlap_width > (min_width * tolerance_factor)

def should_merge_bubbles(bbox1_props, bbox2_props, max_distance_factor=2.0):
    """Determine if two text bubbles should be merged"""
    avg_width = (bbox1_props['width'] + bbox2_props['width']) / 2
    avg_height = (bbox1_props['height'] + bbox2_props['height']) / 2
    avg_size = (avg_width + avg_height) / 2
    
    distance = calculate_distance(bbox1_props, bbox2_props)
    relative_distance = distance / avg_size if avg_size > 0 else float('inf')
    
    if relative_distance > max_distance_factor:
        return False
    
    h_aligned = is_horizontally_aligned(bbox1_props, bbox2_props)
    v_aligned = is_vertically_aligned(bbox1_props, bbox2_props)
    
    if relative_distance < 1.5 and (h_aligned or v_aligned):
        return True
    
    if relative_distance < 0.8:
        return True
    
    return False

def manga_reading_order_sort(results):
    """Sort results in manga reading order (right-to-left, top-to-bottom)"""
    def sort_key(item):
        bbox, text, score = item
        props = calculate_bbox_properties(bbox)
        return (props['center_y'], -props['center_x'])
    
    return sorted(results, key=sort_key)

def calculate_collective_bbox(paragraph_items):
    """Calculate collective bbox for paragraph"""
    if not paragraph_items:
        return None
    
    all_xs = []
    all_ys = []
    
    for item in paragraph_items:
        bbox = item['bbox']
        for point in bbox:
            all_xs.append(point[0])
            all_ys.append(point[1])
    
    min_x = min(all_xs)
    max_x = max(all_xs)
    min_y = min(all_ys)
    max_y = max(all_ys)
    
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

def group_paragraphs(results, max_distance_factor=2.0):
    """Advanced paragraph grouping for manga"""
    if not results:
        return []
    
    sorted_results = manga_reading_order_sort(results)
    
    items_with_props = []
    for bbox, text, score in sorted_results:
        props = calculate_bbox_properties(bbox)
        items_with_props.append({
            'bbox': bbox,
            'text': text,
            'score': score,
            'props': props,
            'used': False
        })
    
    paragraphs = []
    
    for i, current_item in enumerate(items_with_props):
        if current_item['used']:
            continue
            
        paragraph_items = [current_item]
        current_item['used'] = True
        
        search_expanded = True
        while search_expanded:
            search_expanded = False
            
            for j, candidate_item in enumerate(items_with_props):
                if candidate_item['used']:
                    continue
                
                should_merge = False
                for para_item in paragraph_items:
                    if should_merge_bubbles(
                        para_item['props'], 
                        candidate_item['props'], 
                        max_distance_factor
                    ):
                        should_merge = True
                        break
                
                if should_merge:
                    paragraph_items.append(candidate_item)
                    candidate_item['used'] = True
                    search_expanded = True
        
        paragraph_items.sort(key=lambda x: (x['props']['center_y'], -x['props']['center_x']))
        paragraph_text = ' '.join([item['text'] for item in paragraph_items])
        
        collective_bbox = calculate_collective_bbox(paragraph_items)
        avg_score = sum([item['score'] for item in paragraph_items]) / len(paragraph_items)
        
        individual_items = []
        for item in paragraph_items:
            individual_items.append({
                'bbox': item['bbox'],
                'text': item['text'],
                'score': item['score']
            })
        
        paragraph_data = {
            'text': paragraph_text,
            'bbox': collective_bbox,
            'score': avg_score,
            'item_count': len(paragraph_items),
            'individual_items': individual_items
        }
        
        paragraphs.append(paragraph_data)
    
    return paragraphs

@app.get("/")
async def root():
    return {"message": "RapidOCR API for Vercel", "status": "online"}

@app.post("/init")
async def api_init(req: InitRequest):
    """Initialize OCR"""
    log_memory_usage("Before Init")
    start_time = time.time()
    try:
        reader = get_reader()
        end_time = time.time()
        print(f"[TIME] RapidOCR Initialization: {end_time - start_time:.3f}s")
        log_memory_usage("After Init")
        return {
            "status": "success",
            "message": "RapidOCR initialized for Vercel deployment",
            "langs": req.languages,
            "note": "Enhanced grouping for manga dialogue bubbles."
        }
    except Exception as e:
        log_memory_usage("Init Failed")
        raise HTTPException(status_code=500, detail=f"Failed to initialize OCR: {str(e)}")

@app.post("/read_text")
async def api_read_text(image: UploadFile = File(...), languages: str = "en"):
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        log_memory_usage("Before Processing")
        start_time = time.time()

        gc.collect()
        reader = get_reader()
        ocr, splitter = reader["ocr"], reader["splitter"]

        # Read image bytes directly
        image_bytes = await image.read()
        print(f"[INFO] Processing image: {image.filename} ({len(image_bytes)} bytes)")
        
        # Convert to numpy array for RapidOCR
        numpy_image = process_image_bytes(image_bytes)
        
        # Run OCR on numpy array
        results, _ = ocr(numpy_image)

        if not results:
            return JSONResponse(content={
                "status": "success",
                "results": [],
                "paragraphs": [],
                "message": "No text detected in image"
            })

        # Fix spaces in each result
        fixed_results = []
        for box, text, score in results:
            # Try to fix glued words if wordninja is available
            if splitter and " " not in text:
                text = " ".join(splitter.split(text))
            fixed_results.append({
                "bbox": box,
                "text": text,
                "score": float(score)
            })

        # Group into paragraphs
        grouped_paragraphs = group_paragraphs(
            [(res["bbox"], res["text"], res["score"]) for res in fixed_results],
            max_distance_factor=2.0
        )

        total_time = time.time() - start_time
        print(f"[TIME] Total read_text: {total_time:.3f}s")
        print(f"[INFO] Processed {len(fixed_results)} lines into {len(grouped_paragraphs)} paragraphs")

        gc.collect()
        log_memory_usage("After Processing")

        return JSONResponse(content={
            "status": "success",
            "results": fixed_results,
            "paragraphs": grouped_paragraphs,
            "stats": {
                "total_lines": len(fixed_results),
                "total_paragraphs": len(grouped_paragraphs),
                "processing_time": f"{total_time:.3f}s",
                "image_size": len(image_bytes)
            },
            "memory_usage": f"~{psutil.Process().memory_info().rss / 1024 / 1024:.0f}MB"
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log_memory_usage("Processing Failed")
        raise HTTPException(status_code=500, detail=f"Error reading text: {str(e)}")
    finally:
        gc.collect()

@app.post("/close")
async def api_close():
    """Close and cleanup everything"""
    global _reader
    log_memory_usage("Before Close")

    if _reader is not None:
        del _reader
        _reader = None

    gc.collect()
    log_memory_usage("After Close")
    return JSONResponse(content={"status": "success", "message": "Memory cleanup complete"})

# For Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)