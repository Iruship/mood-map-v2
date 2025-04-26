from pymongo import MongoClient, ASCENDING, DESCENDING
from config.config import Config

def init_db():
    client = MongoClient(Config.MONGODB_URI)
    db = client["mental_health_db"]

    # Create indexes for users collection
    db.users.create_index([('username', ASCENDING)], unique=True)
    db.users.create_index([('email', ASCENDING)], unique=True)

    # Create indexes for PHQ-9 responses
    db.phq9_responses.create_index([('userId', ASCENDING)])
    db.phq9_responses.create_index([('created_at', DESCENDING)])
    db.phq9_responses.create_index([('userId', ASCENDING), ('created_at', DESCENDING)])

    # Create indexes for emotion analyses
    db.emotion_analyses.create_index([('user_id', ASCENDING)])
    db.emotion_analyses.create_index([('created_at', DESCENDING)])
    db.emotion_analyses.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])

    print("Database indexes created successfully") 