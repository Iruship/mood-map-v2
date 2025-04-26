from fastapi import FastAPI
from fastapi_cors import CORS
from contextlib import asynccontextmanager
from routes.auth import router as auth_router
from routes.phq9 import router as phq9_router
from routes.emotion import router as emotion_router
from routes.emotion_detection import router as emotion_detection_router
from routes.body_language import router as body_language_router
from config.database import initialize_db, close_db
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        initialize_db()
        print("Successfully initialized database with indexes")
    except Exception as e:
        print(f"Failed to initialize database: {str(e)}")
        raise e
    yield
    # Shutdown
    close_db()

app = FastAPI(
    title="Depression Analysis API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
    allow_credentials=True,  # Important for sending cookies/authorization headers
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(phq9_router, prefix="/api/phq9", tags=["PHQ-9"])
app.include_router(emotion_router, prefix="/api/emotion", tags=["Emotion"])
app.include_router(emotion_detection_router, prefix="/api/emotion-detection", tags=["Emotion Detection"])
app.include_router(body_language_router, prefix="/api/body-language", tags=["Body Language Detection"])

@app.get("/")
async def root():
    return {"message": "Welcome to Depression Analysis API"}

if __name__ == '__main__':
    try:
        port = 5001
        print(f"Starting server on port {port}")
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    except Exception as e:
        print(f"Failed to start server: {e}") 