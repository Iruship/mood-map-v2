import cv2
import numpy as np
import tensorflow as tf

#TODO: ALL MODELS SHOULD BE IN A MODEL FOLDER
class EmotionDetectionService:
    EMOTION_DICT = {
        0: "Angry",
        1: "Disgusted",
        2: "Fearful",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprised"
    }

    def __init__(self):
        self.emotion_model = None
        self.face_detector = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize the emotion detection and face detection models"""
        try:
            self.emotion_model = tf.keras.models.load_model('./ml/emotion.keras')
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
            )
            print("All models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.emotion_model = None
            self.face_detector = None

    def is_initialized(self):
        """Check if models are properly initialized"""
        return self.emotion_model is not None and self.face_detector is not None

    def process_image(self, image_data):
        """Process image data and detect emotions"""
        if not self.is_initialized():
            raise Exception("Models not properly initialized")

        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Invalid image data")

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            raise ValueError("No faces detected in the image")

        # Process each face
        results = []
        for (x, y, w, h) in faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1),
                0
            )
            emotion_prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            
            results.append({
                'emotion': self.EMOTION_DICT[maxindex],
                'confidence': float(emotion_prediction[0][maxindex]),
                'face_location': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })

        return results 