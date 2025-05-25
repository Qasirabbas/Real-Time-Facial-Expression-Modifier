from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import logging
import uvicorn
import httpx
import copy
from pathlib import Path

# Import your Advance_Live_Portrait modules
from Advance_Live_Portrait_2 import (
    AdvancedLivePortrait_execution,
    ExpressionEditor,
    LP_Engine,
    Create_gif,
    AdvancedLivePortrait,
    VideoCombine,
    pil2tensor,
    get_device
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
BASE_FOLDER = '/mnt/disk2/user_data/'
# Configure static files and templates
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/expression/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
user_images: Dict[str, Image.Image] = {}
user_parameters: Dict[str, Dict[str, float]] = {}
user_face_boxes: Dict[str, List[List[int]]] = {}
user_face_states: Dict[str, Dict[int, Dict[str, float]]] = {}
user_motion_links: Dict[str, Dict] = {}
user_last_edited_images: Dict[str, Image.Image] = {}
last_motion_link = None

# Model instances
model = None
engine = None
adv_editor = None
vid_editor = None
executor = ThreadPoolExecutor()

# Constants
TARGET_SIZE = (1024, 1024)
DEFAULT_PARAMETERS = {
    "rotate_pitch": 0.0,
    "rotate_yaw": 0.0,
    "rotate_roll": 0.0,
    "blink": 0.0,
    "eyebrow": 0.0,
    "wink": 0.0,
    "pupil_x": 0.0,
    "pupil_y": 0.0,
    "aaa": 0.0,
    "eee": 0.0,
    "woo": 0.0,
    "smile": 0.0
}
async def check_user_auth(request: Request) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        print("Checking user authentication via Nginx...")

        cookies = request.cookies
        headers = {
            'Authorization': f'Bearer {cookies.get("access_token")}'
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://192.168.1.2:5009/check_auth",
                headers=headers,
                cookies=cookies,
                timeout=5.0
            )

        print(f"Auth URL: {response.url}, Response Code: {response.status_code}")

        if response.status_code == 200:
            auth_data = response.json()
            if auth_data.get('isAuthenticated'):
                return True, auth_data.get('user'), None
            return False, None, "Not authenticated"

        return False, None, f"Auth failed with status code: {response.status_code}"

    except Exception as e:
        print(f"Auth check failed: {e}")
        return False, None, str(e)


# Directory management
def create_user_directories(user_id: str) -> Tuple[str, str]:
    user_folder = os.path.join(BASE_FOLDER, str(user_id))
    uploads_folder = os.path.join(user_folder, 'uploads')
    output_folder = os.path.join(user_folder, 'output')

    os.makedirs(uploads_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    return uploads_folder, output_folder


# Dependency for authentication
async def get_current_user(request: Request) -> str:
    is_auth, user_id, error = await check_user_auth(request)
    if not is_auth:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

# Helper Functions
def resize_image(image: Image.Image, target_size: tuple) -> Image.Image:
    """Resize image maintaining aspect ratio"""
    target_width, target_height = target_size
    original_width, original_height = image.size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

async def process_single_face(
    input_img: Image.Image,
    parameters: list,
    face_box: list,
    sample_image: Optional[Image.Image] = None
) -> Tuple[Image.Image, dict, dict]:
    """Process a single face in the image"""
    global model, engine
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            executor,
            AdvancedLivePortrait_execution,
            model,
            input_img,
            parameters,
            face_box,
            sample_image
        )
    except Exception as e:
        logger.error(f"Error processing single face: {str(e)}", exc_info=True)
        raise

async def process_image_with_states(
    image_id: str,
    face_index: int,
    parameters: dict
) -> Tuple[Image.Image, dict, dict]:
    """Process image applying all face states"""
    if image_id not in user_images or image_id not in user_face_boxes:
        raise HTTPException(status_code=404, detail="Image not found")

    input_img = user_images[image_id]
    face_boxes = user_face_boxes[image_id]
    
    # Update state for the edited face
    if image_id not in user_face_states:
        user_face_states[image_id] = {}
    if face_index not in user_face_states[image_id]:
        user_face_states[image_id][face_index] = DEFAULT_PARAMETERS.copy()
    
    user_face_states[image_id][face_index].update(parameters)

    # Process image
    try:
        current_img = copy.deepcopy(input_img)
        current_motion_links = {}

        # Apply each face's state in sequence
        for idx, face_box in enumerate(face_boxes):
            if idx in user_face_states[image_id]:
                face_params = user_face_states[image_id][idx]
                param_list = list(face_params.values())

                result_img, motion_link, _ = await process_single_face(
                    current_img,
                    param_list,
                    face_box
                )
                
                current_img = result_img
                if idx == face_index:
                    current_motion_links = motion_link

        # Store results
        user_last_edited_images[image_id] = current_img
        if image_id not in user_motion_links:
            user_motion_links[image_id] = {}
        user_motion_links[image_id] = current_motion_links

        return current_img, current_motion_links, None

    except Exception as e:
        logger.error(f"Error processing image with states: {str(e)}", exc_info=True)
        raise

# Model Loading
@app.on_event("startup")
async def load_model():
    """Load models on startup"""
    global model, engine, adv_editor, vid_editor
    try:
        logger.info("Loading models...")
        engine = LP_Engine()
        adv_editor = AdvancedLivePortrait(engine)
        vid_editor = VideoCombine()
        model = ExpressionEditor(engine)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise

# Home page route
@app.get("/expression", response_class=HTMLResponse)
async def read_root(request: Request):
    print("Accessing /expression page...")
    is_auth, user_id, error = await check_user_auth(request)

    if not is_auth or user_id is None:
        print("Redirecting to login page...")
        return RedirectResponse(url=f'/login?next={request.url.path}')

    print(f"Rendering index.html for user: {user_id}")
    create_user_directories(user_id)
    return templates.TemplateResponse("index3.html", {"request": request})

@app.post("/expression/upload")
async def upload_image(file: UploadFile = Form(...)):
    """Handle image upload and initial face detection"""
    logger.debug(f"Uploading file: {file.filename}")
    
    allowed_formats = [
        'image/jpeg', 'image/png', 'image/gif',
        'image/bmp', 'image/tiff', 'image/webp',
        'image/heif', 'image/svg+xml'
    ]

    if file.content_type not in allowed_formats:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if image.size[0] > TARGET_SIZE[0] or image.size[1] > TARGET_SIZE[1]:
            image = resize_image(image, TARGET_SIZE)

        # Generate ID and store image
        image_id = str(uuid.uuid4())
        user_images[image_id] = image

        # Detect faces
        # img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        # img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        src_image = image.convert("RGB")
        src_img_tensor = pil2tensor(src_image).to(get_device())
        face_boxes = engine.detect_face(src_img_tensor, crop_factor=1.3)
        
        # Store face data
        user_face_boxes[image_id] = face_boxes
        user_face_states[image_id] = {
            i: DEFAULT_PARAMETERS.copy() 
            for i in range(len(face_boxes))
        }

        logger.debug(f"Image uploaded successfully. ID: {image_id}")
        return {"image_id": image_id, "message": "Image uploaded successfully"}

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/expression/detect-faces/{image_id}")
async def get_detected_faces(image_id: str):
    """Get detected face boxes for an image"""
    logger.debug(f"Getting face boxes for image: {image_id}")
    
    if image_id not in user_images:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        if image_id not in user_face_boxes:
            input_img = user_images[image_id]
            img_tensor = torch.from_numpy(np.array(input_img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            face_boxes = engine.detect_face(img_tensor, crop_factor=1.3)
            user_face_boxes[image_id] = face_boxes
            adjusted_boxes = []
            for box in face_boxes:
                # Calculate center of the face
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                # Calculate width and height
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                # Make boxes smaller by reducing size to 85% of original
                scale_factor = 0.85
                new_width = width * scale_factor
                new_height = height * scale_factor
                
                # Calculate new coordinates while maintaining center
                new_box = [
                    center_x - (new_width / 2),  # new x1
                    center_y - (new_height / 2), # new y1
                    center_x + (new_width / 2),  # new x2
                    center_y + (new_height / 2)  # new y2
                ]
                
                adjusted_boxes.append(new_box)
            
            user_face_boxes[image_id] = adjusted_boxes
        return user_face_boxes[image_id]

    except Exception as e:
        logger.error(f"Face detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/expression/get-face-state/{image_id}/{face_index}")
async def get_face_state(image_id: str, face_index: int):
    """Get the current state of a specific face"""
    try:
        if image_id not in user_face_states:
            raise HTTPException(status_code=404, detail="Image not found")
        
        face_index = int(face_index)
        if face_index not in user_face_states[image_id]:
            return DEFAULT_PARAMETERS
        
        return user_face_states[image_id][face_index]
    except Exception as e:
        logger.error(f"Error getting face state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expression/edit")
async def edit_image(
    image_id: str = Form(...),
    face_index: int = Form(...),
    rotate_pitch: float = Form(0.0),
    rotate_yaw: float = Form(0.0),
    rotate_roll: float = Form(0.0),
    blink: float = Form(0.0),
    eyebrow: float = Form(0.0),
    wink: float = Form(0.0),
    pupil_x: float = Form(0.0),
    pupil_y: float = Form(0.0),
    aaa: float = Form(0.0),
    eee: float = Form(0.0),
    woo: float = Form(0.0),
    smile: float = Form(0.0)
):
    """Edit image expression for a specific face"""
    logger.debug(f"Editing image {image_id}, face index {face_index}")
    
    try:
        # Process parameters
        parameters = {
            "rotate_pitch": rotate_pitch,
            "rotate_yaw": rotate_yaw,
            "rotate_roll": rotate_roll,
            "blink": blink,
            "eyebrow": eyebrow,
            "wink": wink,
            "pupil_x": pupil_x,
            "pupil_y": pupil_y,
            "aaa": aaa,
            "eee": eee,
            "woo": woo,
            "smile": smile
        }

        # Process image with all face states
        result_img, motion_links, _ = await process_image_with_states(
            image_id,
            face_index,
            parameters
        )

        # Convert to bytes
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Edit error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expression/reset-face")
async def reset_face(
    image_id: str = Form(...),
    face_index: int = Form(...)
):
    """Reset a specific face to its default state"""
    logger.debug(f"Resetting face {face_index} for image {image_id}")
    
    try:
        if image_id not in user_face_states:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Reset the face state to default
        user_face_states[image_id][face_index] = DEFAULT_PARAMETERS.copy()
        
        # Process image with updated states
        result_img, motion_links, _ = await process_image_with_states(
            image_id,
            face_index,
            DEFAULT_PARAMETERS
        )

        # Convert to bytes
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error resetting face: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expression/reset-all-faces")
async def reset_all_faces(image_id: str = Form(...)):
    """Reset all faces in an image to their default states"""
    logger.debug(f"Resetting all faces for image {image_id}")
    
    try:
        if image_id not in user_images:
            raise HTTPException(status_code=404, detail="Image not found")

        if image_id not in user_face_boxes:
            raise HTTPException(status_code=400, detail="No face detection data")

        # Reset states for all faces
        if image_id in user_face_states:
            face_boxes = user_face_boxes[image_id]
            user_face_states[image_id] = {
                i: DEFAULT_PARAMETERS.copy() 
                for i in range(len(face_boxes))
            }

        # Process image with reset states
        input_img = user_images[image_id]
        result_img = input_img

        for face_index, face_box in enumerate(user_face_boxes[image_id]):
            parameters = list(DEFAULT_PARAMETERS.values())
            result_img, motion_link, _ = await process_single_face(
                result_img,
                parameters,
                face_box
            )
            if face_index == 0:  # Store motion link for the first face
                user_motion_links[image_id] = motion_link

        # Store the result
        user_last_edited_images[image_id] = result_img

        # Convert to bytes
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Error resetting all faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expression/process-imitation")
async def process_imitation(
    source_image: UploadFile = File(...),
    sample_image: UploadFile = File(...),
    face_index: int = Form(0)
):
    """Process imitation with face selection"""
    try:
        logger.debug("Starting imitation process")
        
        # Load and process images
        source_contents = await source_image.read()
        sample_contents = await sample_image.read()
        
        source_img = Image.open(io.BytesIO(source_contents)).convert("RGB")
        sample_img = Image.open(io.BytesIO(sample_contents)).convert("RGB")
        
        # Resize if needed
        if source_img.size[0] > TARGET_SIZE[0] or source_img.size[1] > TARGET_SIZE[1]:
            source_img = resize_image(source_img, TARGET_SIZE)
        if sample_img.size[0] > TARGET_SIZE[0] or sample_img.size[1] > TARGET_SIZE[1]:
            sample_img = resize_image(sample_img, TARGET_SIZE)
        global last_motion_link
        # Detect faces in source image
        # source_tensor = torch.from_numpy(np.array(source_img)).float() / 255.0
        # source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0)
        # src_image = source_img.convert("RGB")
        src_img_tensor = pil2tensor(source_img).to(get_device())
        face_boxes = engine.detect_face(src_img_tensor, crop_factor=1.3)
        # samp_img_tensor = pil2tensor(sample_img).to(get_device())
        if face_index >= len(face_boxes):
            raise HTTPException(status_code=400, detail="Invalid face index")
        
        # Process imitation
        result_img, motion_link, _ = await process_single_face(
            source_img,
            list(DEFAULT_PARAMETERS.values()),
            face_boxes[face_index],
            sample_img
        )
        last_motion_link = motion_link
        # Store motion link for potential GIF creation
        image_id = str(uuid.uuid4())
        user_motion_links[image_id] = motion_link
        
        # Convert result to bytes
        buf = io.BytesIO()
        result_img.save(buf, format='PNG')
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Imitation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/expression/create-gif")
async def create_gif():
    global last_motion_link
    if last_motion_link is None:
        raise HTTPException(status_code=400, detail="No motion link available. Please edit an image first.")

    try:
        # Create GIF
        gif_path = await asyncio.get_event_loop().run_in_executor(
            executor,
            Create_gif,
            adv_editor,  # This should be your ExpressionEditor instance
            vid_editor,  # This should be your VideoCombine instance
            last_motion_link
        )
        # Read the GIF file
        with open(gif_path, 'rb') as f:
            gif_data = f.read()
        return StreamingResponse(io.BytesIO(gif_data), media_type="image/gif")
    except Exception as e:
        print(f"Error in create_gif: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating GIF: {str(e)}")

@app.delete("/expression/delete/{image_id}")
async def delete_image(image_id: str):
    """Delete image and associated data"""
    try:
        # Remove from all storage dictionaries
        for storage in [
            user_images,
            user_parameters,
            user_face_boxes,
            user_face_states,
            user_motion_links,
            user_last_edited_images
        ]:
            if image_id in storage:
                del storage[image_id]
        
        return {"message": "Image deleted successfully"}
    except Exception as e:
        logger.error(f"Delete error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting image: {str(e)}"
        )

# Error Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "error": str(exc)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP exception: {str(exc.detail)}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail
        }
    )

# Health Check
@app.get("/expression/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([model, engine, adv_editor, vid_editor]),
        "active_sessions": len(user_images)
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        # Clear all stored data
        user_images.clear()
        user_parameters.clear()
        user_face_boxes.clear()
        user_face_states.clear()
        user_motion_links.clear()
        user_last_edited_images.clear()
        
        # Shutdown executor
        executor.shutdown(wait=True)
        
        # Clean up temporary files
        temp_dir = OUTPUT_DIR / "temp"
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                except:
                    pass
            try:
                temp_dir.rmdir()
            except:
                pass
        
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4001,
        log_level="debug",
        access_log=True
    )