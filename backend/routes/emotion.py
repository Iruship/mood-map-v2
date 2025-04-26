from fastapi import APIRouter, HTTPException, status, Depends, Query
from models.schemas import EmotionCreate, EmotionResponse, TokenData
from utils.auth import get_current_user
from pymongo import DESCENDING
from config.database import get_db
from typing import List, Annotated
from datetime import datetime

router = APIRouter()
db = get_db()
emotion_collection = db.emotion_analyses

@router.post("/record", response_model=EmotionResponse, status_code=status.HTTP_201_CREATED)
async def record_emotion(
    emotion_data: EmotionCreate,
    current_user: Annotated[TokenData, Depends(get_current_user)]
):
    # Create emotion analysis document
    analysis_dict = {
        "user_id": current_user.username,
        "emotion": emotion_data.emotion,
        "confidence": emotion_data.confidence,
        "created_at": datetime.utcnow()
    }
    
    # Store in database
    result = emotion_collection.insert_one(analysis_dict)
    analysis_dict["_id"] = result.inserted_id
    
    return EmotionResponse(**analysis_dict)

@router.get("/history", response_model=List[EmotionResponse])
async def get_history(
    current_user: Annotated[TokenData, Depends(get_current_user)],
    limit: int = Query(default=10, ge=1, le=100)
):
    # Get user's emotion analyses, sorted by date
    analyses = list(emotion_collection
                   .find({"user_id": current_user.username})
                   .sort("created_at", DESCENDING)
                   .limit(limit))
    
    return [EmotionResponse(**analysis) for analysis in analyses]

@router.get("/latest", response_model=EmotionResponse)
async def get_latest(current_user: Annotated[TokenData, Depends(get_current_user)]):
    # Get user's latest emotion analysis
    analysis = emotion_collection.find_one(
        {"user_id": current_user.username},
        sort=[("created_at", DESCENDING)]
    )

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No emotion analyses found"
        )

    return EmotionResponse(**analysis) 