from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from models.schemas import BodyLanguageResponse, BodyLanguageTrainingResponse
from services.body_language import BodyLanguageService
from typing import List, Dict, Any, Optional
import os
import csv
import tempfile
import shutil
import cv2
import numpy as np
import base64
import importlib
import sys
from pydantic import BaseModel

router = APIRouter()
body_language_service = BodyLanguageService()

# Define a model for threshold updates
class ThresholdUpdate(BaseModel):
    detection_confidence: Optional[float] = None
    tracking_confidence: Optional[float] = None
    model_complexity: Optional[int] = None
    prediction_confidence: Optional[float] = None
    pad_missing_features: Optional[bool] = None
    handle_nan_values: Optional[bool] = None
    min_training_samples: Optional[int] = None

@router.post("/update-thresholds")
async def update_thresholds(update: ThresholdUpdate):
    """
    Update detection thresholds at runtime.
    Changes apply to new detections but won't persist after server restart.
    """
    try:
        # Import the threshold module
        import config.detection_thresholds as thresholds
        
        # Update only the provided values
        changes_made = False
        
        if update.detection_confidence is not None:
            if 0 <= update.detection_confidence <= 1:
                thresholds.DETECTION_CONFIDENCE = update.detection_confidence
                changes_made = True
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Detection confidence must be between 0 and 1"}
                )
                
        if update.tracking_confidence is not None:
            if 0 <= update.tracking_confidence <= 1:
                thresholds.TRACKING_CONFIDENCE = update.tracking_confidence
                changes_made = True
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Tracking confidence must be between 0 and 1"}
                )
                
        if update.model_complexity is not None:
            if update.model_complexity in [0, 1, 2]:
                thresholds.MODEL_COMPLEXITY = update.model_complexity
                changes_made = True
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Model complexity must be 0, 1, or 2"}
                )
                
        if update.prediction_confidence is not None:
            if 0 <= update.prediction_confidence <= 1:
                thresholds.PREDICTION_CONFIDENCE = update.prediction_confidence
                changes_made = True
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Prediction confidence must be between 0 and 1"}
                )
                
        if update.pad_missing_features is not None:
            thresholds.PAD_MISSING_FEATURES = update.pad_missing_features
            changes_made = True
            
        if update.handle_nan_values is not None:
            thresholds.HANDLE_NAN_VALUES = update.handle_nan_values
            changes_made = True
            
        if update.min_training_samples is not None:
            if update.min_training_samples > 0:
                thresholds.MIN_TRAINING_SAMPLES = update.min_training_samples
                changes_made = True
            else:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "Minimum training samples must be greater than 0"}
                )
                
        if changes_made:
            # Return current values
            return {
                "success": True,
                "message": "Thresholds updated successfully",
                "current_values": {
                    "detection_confidence": thresholds.DETECTION_CONFIDENCE,
                    "tracking_confidence": thresholds.TRACKING_CONFIDENCE,
                    "model_complexity": thresholds.MODEL_COMPLEXITY,
                    "prediction_confidence": thresholds.PREDICTION_CONFIDENCE,
                    "pad_missing_features": thresholds.PAD_MISSING_FEATURES,
                    "handle_nan_values": thresholds.HANDLE_NAN_VALUES,
                    "min_training_samples": thresholds.MIN_TRAINING_SAMPLES
                }
            }
        else:
            return {
                "success": True, 
                "message": "No changes made to thresholds",
                "current_values": {
                    "detection_confidence": thresholds.DETECTION_CONFIDENCE,
                    "tracking_confidence": thresholds.TRACKING_CONFIDENCE,
                    "model_complexity": thresholds.MODEL_COMPLEXITY,
                    "prediction_confidence": thresholds.PREDICTION_CONFIDENCE,
                    "pad_missing_features": thresholds.PAD_MISSING_FEATURES,
                    "handle_nan_values": thresholds.HANDLE_NAN_VALUES,
                    "min_training_samples": thresholds.MIN_TRAINING_SAMPLES
                }
            }
            
    except Exception as e:
        print(f"Error updating thresholds: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error updating thresholds: {str(e)}"}
        )

@router.get("/get-thresholds")
async def get_thresholds():
    """Get the current detection threshold values"""
    try:
        import config.detection_thresholds as thresholds
        
        return {
            "success": True,
            "thresholds": {
                "detection_confidence": thresholds.DETECTION_CONFIDENCE,
                "tracking_confidence": thresholds.TRACKING_CONFIDENCE,
                "model_complexity": thresholds.MODEL_COMPLEXITY,
                "prediction_confidence": thresholds.PREDICTION_CONFIDENCE,
                "pad_missing_features": thresholds.PAD_MISSING_FEATURES,
                "handle_nan_values": thresholds.HANDLE_NAN_VALUES,
                "min_training_samples": thresholds.MIN_TRAINING_SAMPLES
            }
        }
    except Exception as e:
        print(f"Error getting thresholds: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error getting thresholds: {str(e)}"}
        )

@router.post("/detect", response_model=BodyLanguageResponse)
async def detect_body_language(image: UploadFile = File(...)):
    try:
        # Check if model is loaded
        if not body_language_service.model:
            return BodyLanguageResponse(
                success=False,
                error="Body language model not loaded. Please train a model first."
            )
            
        # Read and process the image
        image_data = await image.read()
        result = body_language_service.process_image(image_data)

        return BodyLanguageResponse(
            success=True,
            body_language_class=result.get("class", "Unknown"),
            confidence=result.get("confidence", 0.0),
            landmarks=result.get("landmarks", {})
        )
    except ValueError as e:
        error_msg = str(e)
        print(f"Body language detection error: {error_msg}")
        
        if "feature" in error_msg.lower() and "dimension" in error_msg.lower():
            print("Feature dimension mismatch detected. This could be due to different landmark counts between training and inference.")
            
        return BodyLanguageResponse(
            success=False,
            error=error_msg
        )
    except Exception as e:
        print(f"Unexpected error in body language detection: {str(e)}")
        return BodyLanguageResponse(
            success=False,
            error=f"Unexpected error: {str(e)}"
        )

@router.post("/process-frame")
async def process_frame(image: UploadFile = File(...)):
    try:
        # Check if mediapipe is initialized
        if not body_language_service.mp_holistic:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "MediaPipe holistic model not initialized"}
            )
            
        image_data = await image.read()
        result = body_language_service.process_image_with_landmarks(image_data)
        
        # Return the image with landmarks as base64 with predictions
        return JSONResponse(
            content={
                "success": True,
                "annotated_image": result.get("annotated_image", ""),
                "body_language_class": result.get("class", "Unknown"),
                "confidence": result.get("confidence", 0.0)
            }
        )
    except ValueError as e:
        error_msg = str(e)
        print(f"Frame processing error: {error_msg}")
        
        if "feature" in error_msg.lower() and "dimension" in error_msg.lower():
            print("Feature dimension mismatch detected during frame processing.")
            
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": error_msg}
        )
    except Exception as e:
        print(f"Unexpected error in frame processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Unexpected error: {str(e)}"}
        )

@router.post("/record-training-data")
async def record_training_data(
    image: UploadFile = File(...),
    class_name: str = Form(...),
):
    try:
        # Read and process the image
        image_data = await image.read()
        success = body_language_service.record_landmark_data(image_data, class_name)
        
        if success:
            return {"success": True, "message": f"Training data recorded for class: {class_name}"}
        else:
            return {"success": False, "message": "Failed to detect landmarks"}
    except Exception as e:
        print(f"Error recording training data: {str(e)}")
        return {"success": False, "error": str(e)}

@router.post("/train-model", response_model=BodyLanguageTrainingResponse)
async def train_model():
    try:
        training_results = body_language_service.train_model()
        return BodyLanguageTrainingResponse(
            success=True,
            accuracy=training_results.get("accuracy", 0.0),
            model_trained=True,
            classes=training_results.get("classes", [])
        )
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return BodyLanguageTrainingResponse(
            success=False,
            error=str(e)
        )

@router.get("/available-classes")
async def get_available_classes():
    try:
        classes = body_language_service.get_available_classes()
        return {"success": True, "classes": classes}
    except Exception as e:
        print(f"Error getting available classes: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/download-model")
async def download_model():
    try:
        model_path = body_language_service.get_model_path()
        if os.path.exists(model_path):
            return FileResponse(
                path=model_path,
                filename="body_language_model.pkl",
                media_type="application/octet-stream"
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Model file not found"}
            )
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@router.post("/fix-feature-dimensions")
async def fix_feature_dimensions():
    """
    Fix feature dimension issues by retraining the model with consistent features.
    This is useful when you encounter 'X has N features, but StandardScaler is expecting M features' errors.
    """
    try:
        # Check if we have training data
        if not os.path.exists(body_language_service.data_path) or os.path.getsize(body_language_service.data_path) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No training data available"}
            )
            
        # First, repair the data file to ensure consistent columns
        success = body_language_service.repair_data_file()
        if not success:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to repair data file"}
            )
        
        # Then retrain the model
        training_results = body_language_service.train_model()
        
        # Update the feature count file
        model_dir = body_language_service.model_dir
        feature_count_file = os.path.join(model_dir, "feature_count.txt")
        feature_count = training_results.get("feature_count", 0)
        
        with open(feature_count_file, 'w') as f:
            f.write(str(feature_count))
        
        return JSONResponse(
            content={
                "success": True,
                "message": "Model retrained with consistent feature dimensions",
                "feature_count": feature_count,
                "accuracy": training_results.get("accuracy", 0.0),
                "classes": training_results.get("classes", [])
            }
        )
    except Exception as e:
        print(f"Error fixing feature dimensions: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error fixing feature dimensions: {str(e)}"}
        )

@router.post("/delete-all-data")
async def delete_all_data():
    """
    Delete all training data, models, and feature count information.
    This will remove all training samples and reset the model.
    """
    try:
        success = body_language_service.delete_all_data()
        
        if success:
            return JSONResponse(
                content={
                    "success": True,
                    "message": "All training data and models deleted successfully"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to delete training data"}
            )
    except Exception as e:
        print(f"Error deleting training data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error deleting training data: {str(e)}"}
        ) 