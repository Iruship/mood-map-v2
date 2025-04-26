from fastapi import APIRouter, HTTPException, status, File, UploadFile
from models.schemas import EmotionDetectionResponse
from services.emotion_detection import EmotionDetectionService

router = APIRouter()
emotion_service = EmotionDetectionService()

@router.post("/detect", response_model=EmotionDetectionResponse)
async def detect_emotion(image: UploadFile = File(...)):
    try:
        if not emotion_service.is_initialized():
            return EmotionDetectionResponse(
                success=False,
                error="Models not properly loaded"
            )

        # Read and process the image
        image_data = await image.read()
        results = emotion_service.process_image(image_data)

        return EmotionDetectionResponse(
            success=True,
            faces=results
        )

    except ValueError as e:
        return EmotionDetectionResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return EmotionDetectionResponse(
            success=False,
            error=str(e)
        ) 