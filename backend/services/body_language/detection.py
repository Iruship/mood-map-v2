import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .utils import utils

class BodyLanguageDetection:
    """Handles body language detection and image processing functionality."""
    
    def __init__(self, service):
        self.service = service
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and detect body language"""
        if not self.service.model:
            raise ValueError("Model not loaded")
            
        # Convert and process image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        with utils.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            # Process the image
            results = holistic.process(image_rgb)
            
            # Get landmarks with consistent dimensions
            landmarks = utils.extract_landmarks(results)
            
            # Flatten landmarks for prediction
            row = utils.flatten_landmarks(landmarks)
            
            try:
                # Ensure feature count matches what the model expects
                row = utils.prepare_features_for_model(row, self.service)
                
                # Create DataFrame and predict
                X = pd.DataFrame([row])
                body_language_class = self.service.model.predict(X)[0]
                body_language_prob = self.service.model.predict_proba(X)[0]
                
                # Get the confidence for the predicted class
                class_idx = list(self.service.model.classes_).index(body_language_class)
                confidence = body_language_prob[class_idx]
                
                # Only return predictions with confidence above threshold
                if confidence < 0.65:
                    body_language_class = "Unknown"
                    
                return {
                    "class": body_language_class,
                    "confidence": float(confidence),
                    "landmarks": landmarks
                }
            except Exception as e:
                raise ValueError(f"Error during prediction: {str(e)}")
    
    def process_image_with_landmarks(self, image_data: bytes) -> Tuple[np.ndarray, bool]:
        """Process image and return it with landmarks drawn on it"""
        # Convert image data
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a copy for drawing
        annotated_image = image.copy()
        
        # Process with MediaPipe
        with utils.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            # Process the image
            results = holistic.process(image_rgb)
            
            # Check if any landmarks were detected
            landmarks_detected = (results.pose_landmarks is not None or 
                               results.face_landmarks is not None or
                               results.left_hand_landmarks is not None or
                               results.right_hand_landmarks is not None)
            
            if not landmarks_detected:
                return annotated_image, False
            
            # Draw landmarks
            mp_drawing = utils.mp_drawing
            mp_holistic = utils.mp_holistic
            
            # Draw face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.face_landmarks, 
                    mp_holistic.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )
            
            # Draw hands landmarks
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, 
                    results.right_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )
            
            # Predict body language if model is loaded
            if self.service.model is not None:
                try:
                    # Get landmarks and predict
                    landmarks = utils.extract_landmarks(results)
                    row = utils.flatten_landmarks(landmarks)
                    row = utils.prepare_features_for_model(row, self.service)
                    
                    # Predict
                    X = pd.DataFrame([row])
                    body_language_class = self.service.model.predict(X)[0]
                    body_language_prob = self.service.model.predict_proba(X)[0]
                    
                    # Get confidence and add text if high enough
                    class_idx = list(self.service.model.classes_).index(body_language_class)
                    confidence = body_language_prob[class_idx]
                    
                    if confidence >= 0.65:
                        # Add text with prediction
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = f"{body_language_class}: {confidence:.2f}"
                        cv2.putText(annotated_image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error adding prediction to image: {str(e)}")
        
        # Return the annotated image and success status
        return annotated_image, landmarks_detected 