from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.models.tensorflow_model import process_image_with_model, process_video_with_model
import shutil
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload_image_tensorflow")
async def upload_image_tensorflow(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "tensorflow", "images")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Image uploaded successfully: {file_path}")

        # Process the image with the TensorFlow model
        result = process_image_with_model(file_path)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@router.post("/upload_video_tensorflow")
async def upload_video_tensorflow(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "tensorflow", "videos")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video uploaded successfully: {file_path}")

        # Process the video with the TensorFlow model
        result = process_video_with_model(file_path)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
