from pymongo import MongoClient
from config.config import Config
from models.init_db import init_db

_client = None
_db = None

def get_db():
    global _client, _db
    if _db is None:
        _client = MongoClient(Config.MONGODB_URI)
        _db = _client["mental_health_db"]
    return _db

def initialize_db():
    """Initialize database with indexes and collections"""
    global _client, _db
    if _db is None:
        _client = MongoClient(Config.MONGODB_URI)
        _db = _client["mental_health_db"]
    init_db()

def close_db():
    """Close database connection"""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None 