from datetime import datetime
import uuid

class EmotionAnalysis:
    def __init__(self, user_id, emotion, confidence, image_data=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.emotion = emotion
        self.confidence = confidence
        self.created_at = datetime.utcnow()

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "emotion": self.emotion,
            "confidence": self.confidence,
            "created_at": self.created_at
        }

    @staticmethod
    def from_dict(data):
        analysis = EmotionAnalysis(
            user_id=data.get("user_id"),
            emotion=data.get("emotion"),
            confidence=data.get("confidence")
        )
        if "id" in data:
            analysis.id = data["id"]
        if "created_at" in data:
            analysis.created_at = data["created_at"]
        return analysis 