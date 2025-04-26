from fastapi import APIRouter, HTTPException, status, Depends, Query
from models.schemas import PHQ9Submit, PHQ9Response, PHQ9Questions, PHQ9Question, TokenData, PHQ9HistoryResponse
from utils.auth import get_current_user
from pymongo import DESCENDING
from pymongo.errors import PyMongoError
from config.database import get_db
from typing import List, Annotated
from datetime import datetime
import logging
import uuid

router = APIRouter()
db = get_db()
phq9_collection = db.phq9_responses

logger = logging.getLogger(__name__)

# PHQ-9 Questions for reference
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself"
]

SEVERITY_LEVELS = {
    "minimal": {"min": 0, "max": 4},
    "mild": {"min": 5, "max": 9},
    "moderate": {"min": 10, "max": 14},
    "moderately_severe": {"min": 15, "max": 19},
    "severe": {"min": 20, "max": 27}
}

def calculate_severity(total_score: int) -> str:
    for level, range_ in SEVERITY_LEVELS.items():
        if range_["min"] <= total_score <= range_["max"]:
            return level
    return "severe" 

@router.post("/submit", response_model=PHQ9Response, status_code=status.HTTP_201_CREATED)
async def submit_phq9(
    phq9_data: PHQ9Submit,
    current_user: Annotated[dict, Depends(get_current_user)]
):
    try:
        # Create response document with UUID
        response_dict = {
            "_id": str(uuid.uuid4()),
            "username": current_user["username"],
            "userId": current_user["_id"],
            "score": phq9_data.score,
            "created_at": datetime.utcnow()
        }
        
        # Store in database
        result = phq9_collection.insert_one(response_dict)
        if not result.acknowledged:
            logger.error(f"Failed to insert PHQ9 score for user {current_user['username']}: Insert not acknowledged")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save PHQ-9 score"
            )
            
        logger.info(f"Successfully saved PHQ9 score for user {current_user['username']}")
        return response_dict
        
    except PyMongoError as e:
        logger.error(f"Database error while saving PHQ9 score for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while saving PHQ-9 score"
        )
    except Exception as e:
        logger.error(f"Unexpected error while saving PHQ9 score for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@router.get("/history", response_model=List[PHQ9HistoryResponse])
async def get_history(
    current_user: Annotated[dict, Depends(get_current_user)],
    limit: int = Query(default=10, ge=1, le=100)
):
    try:
        # Get user's PHQ-9 responses, sorted by date, only score and created_at fields
        responses = list(phq9_collection
                        .find(
                            {"userId": current_user["_id"]},
                            {"score": 1, "created_at": 1, "_id": 0}
                        )
                        .sort("created_at", DESCENDING)
                        .limit(limit))
        
        logger.info(f"Successfully retrieved {len(responses)} PHQ9 scores for user {current_user['username']}")
        return responses
        
    except PyMongoError as e:
        logger.error(f"Database error while fetching PHQ9 history for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while fetching history"
        )
    except Exception as e:
        logger.error(f"Unexpected error while fetching PHQ9 history for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@router.get("/latest", response_model=PHQ9Response)
async def get_latest(current_user: Annotated[dict, Depends(get_current_user)]):
    try:
        # Get user's latest PHQ-9 response
        response = phq9_collection.find_one(
            {"userId": current_user["_id"]},
            sort=[("created_at", DESCENDING)]
        )

        if not response:
            logger.info(f"No PHQ9 scores found for user {current_user['username']}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No PHQ-9 responses found"
            )
        
        logger.info(f"Successfully retrieved latest PHQ9 score for user {current_user['username']}")
        return response
        
    except PyMongoError as e:
        logger.error(f"Database error while fetching latest PHQ9 score for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred while fetching latest score"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions (like 404)
    except Exception as e:
        logger.error(f"Unexpected error while fetching latest PHQ9 score for user {current_user['username']}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

@router.get("/questions", response_model=PHQ9Questions)
async def get_questions():
    """Return the list of PHQ-9 questions"""
    try:
        return {
            "questions": [
                {"id": idx, "text": question}
                for idx, question in enumerate(PHQ9_QUESTIONS)
            ],
            "scoring_guide": {
                "0": "Not at all",
                "1": "Several days",
                "2": "More than half the days",
                "3": "Nearly every day"
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error while fetching PHQ9 questions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        ) 