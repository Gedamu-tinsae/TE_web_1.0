from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.models.opencv_tensorflow import process_image
from app.models.haze_removal import HazeRemoval
import shutil
import os
import logging
import cv2
import numpy as np
import base64

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
            
            # Load original image for later use
            original_image = cv2.imread(file_path)
            
            # Apply dehazing using HazeRemoval class
            hr = HazeRemoval()
            hr.open_image(file_path)
            hr.get_dark_channel()
            hr.get_air_light()
            hr.get_transmission()
            hr.guided_filter()
            hr.recover()
            
            # Get all intermediate images (except the original which is redundant)
            dehaze_stages = hr.get_all_intermediate_images()
            # Remove the original from stages as it's redundant
            if 'original' in dehaze_stages:
                del dehaze_stages['original']
            
            # Save dehazed image
            cv2.imwrite(dehazed_path, hr.dst)
            logger.info(f"Dehazed image saved at: {dehazed_path}")
            
            # Process the dehazed image with a lower confidence threshold
            result = process_image(dehazed_path, confidence_threshold=0.6)
            
            # Add dehazing info to result
            result["preprocessing"] = "dehazing_applied"
            result["original_path"] = file_path
            
            # Convert all intermediate dehazing images to base64
            dehaze_intermediate_base64 = {}
            for key, img in dehaze_stages.items():
                _, buffer = cv2.imencode('.jpg', img)
                dehaze_intermediate_base64[key] = base64.b64encode(buffer).decode('utf-8')
            
            # Add the dehazing intermediate images to the result
            result["dehaze_stages"] = dehaze_intermediate_base64
            
            # Fix: Restore original colors to the annotated image
            try:
                # Get the annotated image path from the result
                annotated_path = os.path.join("results", "opencv", "images", os.path.basename(result["result_url"]))
                
                # Load the annotated dehazed image
                annotated_image = cv2.imread(annotated_path)
                
                if annotated_image is not None and original_image is not None:
                    # Copy the license plate annotation (green rectangle and text) to the original image
                    
                    # 1. Extract the green annotations (rectangle and text) using color thresholding
                    lower_green = np.array([0, 200, 0])  # BGR lower bound for bright green
                    upper_green = np.array([100, 255, 100])  # BGR upper bound for bright green
                    
                    # Create a mask of green elements
                    mask = cv2.inRange(annotated_image, lower_green, upper_green)
                    
                    # Apply the mask to get only the green elements
                    green_elements = cv2.bitwise_and(annotated_image, annotated_image, mask=mask)
                    
                    # 2. Overlay these green elements onto the original image
                    # Resize original image if dimensions don't match
                    if original_image.shape != annotated_image.shape:
                        original_image = cv2.resize(original_image, (annotated_image.shape[1], annotated_image.shape[0]))
                    
                    # Create a combined image: original background with green annotations
                    combined = cv2.addWeighted(original_image, 1.0, green_elements, 1.0, 0)
                    
                    # 3. Save the result
                    cv2.imwrite(annotated_path, combined)
                    logger.info(f"Restored original colors to annotated image: {annotated_path}")
            except Exception as e:
                logger.error(f"Error restoring original colors: {e}")
            
        else:
            # Process the image without dehazing
            result = process_image(file_path)

        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
