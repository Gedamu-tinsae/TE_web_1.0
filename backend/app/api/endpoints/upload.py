from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.models.opencv_tensorflow import process_image
from app.models.haze_removal import HazeRemoval
import shutil
import os
import logging
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...), low_visibility: bool = Form(False)):
    try:
        # Save the uploaded file
        upload_dir = os.path.join("uploads", "opencv", "images")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded successfully: {file_path}")

        # Apply dehazing if low_visibility is True
        if low_visibility:
            logger.info("Applying dehazing to low visibility image")
            # Create a temporary path for dehazed image
            dehazed_path = os.path.join(upload_dir, f"dehazed_{file.filename}")
            
            # Apply dehazing using HazeRemoval class
            hr = HazeRemoval()
            hr.open_image(file_path)
            hr.get_dark_channel()
            hr.get_air_light()
            hr.get_transmission()
            hr.guided_filter()
            hr.recover()
            
            # Save dehazed image
            cv2.imwrite(dehazed_path, hr.dst)
            logger.info(f"Dehazed image saved at: {dehazed_path}")
            
            # Process the dehazed image
            result = process_image(dehazed_path)
            
            # Add dehazing info to result
            result["preprocessing"] = "dehazing_applied"
            result["original_path"] = file_path
        else:
            # Process the image without dehazing
            result = process_image(file_path)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
