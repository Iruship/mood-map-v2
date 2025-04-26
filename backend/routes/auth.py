from fastapi import APIRouter, HTTPException, status, Depends
from models.schemas import UserCreate, UserResponse, UserLogin, Token
from utils.auth import get_password_hash, verify_password, create_access_token, get_current_user
from config.database import get_db
from datetime import timedelta
from config.config import Config
from typing import Annotated
import logging
import uuid

router = APIRouter()
db = get_db()
users_collection = db.users

logger = logging.getLogger(__name__)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    logger.info(f"Attempting to register new user with username: {user.username} and email: {user.email}")
    
    # Validate passwords match
    if user.password != user.confirm_password:
        logger.warning(f"Password mismatch during registration for username: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )
    
    # Check if username or email already exists
    if users_collection.find_one({"username": user.username}):
        logger.warning(f"Registration failed - username already exists: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    if users_collection.find_one({"email": user.email}):
        logger.warning(f"Registration failed - email already exists: {user.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user document
    user_dict = user.model_dump(exclude={"confirm_password"})
    user_dict["password"] = get_password_hash(user.password)
    user_dict["_id"] = str(uuid.uuid4())  # Generate UUID string
    
    # Insert into database
    try:
        result = users_collection.insert_one(user_dict)
        logger.info(f"Successfully registered new user with username: {user.username}")
        return UserResponse(**user_dict)
    except Exception as e:
        logger.error(f"Failed to register user {user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    logger.info(f"Login attempt for username/email: {user_credentials.username}")
    
    user = users_collection.find_one({
        "$or": [
            {"username": user_credentials.username},
            {"email": user_credentials.username}
        ]
    })
    
    if not user:
        logger.warning(f"Login failed - user not found: {user_credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_password(user_credentials.password, user["password"]):
        logger.warning(f"Login failed - incorrect password for user: {user_credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        access_token_expires = timedelta(seconds=Config.JWT_ACCESS_TOKEN_EXPIRES)
        # Create a copy of user data without sensitive information
        user_data = {
            "username": user["username"],
            "email": user["email"],
            "name": user["name"],
            "_id": str(user["_id"])  # Convert ObjectId to string
        }
        access_token = create_access_token(
            data=user_data,
            expires_delta=access_token_expires
        )
        logger.info(f"Successfully logged in user: {user_credentials.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Failed to create access token for user {user_credentials.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate access token"
        )

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: Annotated[dict, Depends(get_current_user)]):
    logger.info(f"Fetching profile for user: {current_user.username}")
    
    user = users_collection.find_one({"username": current_user.username})
    if not user:
        logger.error(f"User not found in database: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    logger.info(f"Successfully retrieved profile for user: {current_user.username}")
    return UserResponse(**user)