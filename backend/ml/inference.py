from fastapi import FastAPI, File, UploadFile
from preprocess import VideoPreprocessor
from predict import EmotionPredictor
import tempfile
import os
import uvicorn
from typing import Dict
import torch

app = FastAPI()

# Initialize preprocessor and predictor
preprocessor = VideoPreprocessor(
    openpose_model_path="path_to_openpose_model",
    temp_dir="temp_processing"
)

predictor = EmotionPredictor(
    model_dir="pretrained",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

@app.post("/analyse_depression")
async def analyse_depression(video: UploadFile = File(...)) -> Dict:
    """
    Process video and analyse depression indicators
    """
    try:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            content = await video.read()
            temp_video.write(content)
            temp_video_path = temp_video.name
        
        # Preprocess video
        processed_data = preprocessor.process_video(temp_video_path)
        
        # Make predictions
        predictions = predictor.predict(processed_data)
        
        # Clean up
        os.unlink(temp_video_path)
        preprocessor.cleanup()
        
        return {
            "status": "success",
            "predictions": predictions
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 