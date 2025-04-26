from fastapi import FastAPI, File, UploadFile
from preprocess import VideoPreprocessor
from predict import EmotionPredictor
import tempfile
import os
import uvicorn
from typing import Dict
import torch

# Initialize FastAPI application
app = FastAPI()

# Initialize video preprocessor
# This handles all the video processing steps including:
# - Frame extraction
# - OpenPose skeleton detection
# - Feature extraction for the models
preprocessor = VideoPreprocessor(
    openpose_model_path="/models/openpose_model",  # Path to the OpenPose model files
    temp_dir="temp_processing"  # Directory to store temporary processing files
)

# Initialize the emotion prediction model
# This contains the TSN (Temporal Segment Network) and STGCN models
# that analyze visual and motion features to detect emotions
predictor = EmotionPredictor(
    model_dir="pretrained",  # Directory containing pretrained model weights
    device="cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU
)

@app.post("/analyse_depression")
async def analyse_depression(video: UploadFile = File(...)) -> Dict:
    """
    Process video and analyse depression indicators
    
    Parameters:
    - video: Uploaded video file to analyze
    
    Returns:
    - Dictionary containing prediction results or error message
    
    The function performs the following steps:
    1. Save the uploaded video to a temporary file
    2. Process the video to extract features
    3. Run the emotion prediction models
    4. Clean up temporary files
    5. Return the prediction results
    """
    try:
        # Save uploaded video to temporary file
        # Using NamedTemporaryFile to get a secure temporary filename
        # suffix='.mp4' ensures proper file extension for video processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await video.read()  # Asynchronously read the uploaded file
            temp_video.write(content)     # Write content to temporary file
            temp_video_path = temp_video.name  # Store the path for later use
        
        # Preprocess video
        # This extracts frames, detects poses, and prepares data for the models
        processed_data = preprocessor.process_video(temp_video_path)
        
        # Make predictions
        # Run the emotion detection models and calculate depression indicators
        predictions = predictor.predict(processed_data)
        
        # Clean up temporary files
        os.unlink(temp_video_path)  # Delete the temporary video file
        preprocessor.cleanup()      # Clean up any other temporary processing files
        
        # Return successful response with predictions
        return {
            "status": "success",
            "predictions": predictions  # Contains emotion scores and depression indicators
        }
        
    except Exception as e:
        # Handle any errors that occur during processing
        # Return error response with the exception message
        return {
            "status": "error",
            "message": str(e)  # Convert exception to string for the response
        }

# Run the FastAPI application when this script is executed directly
if __name__ == "__main__":
    # Start the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000) 