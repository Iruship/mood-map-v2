import cv2
import numpy as np
import pandas as pd
import base64
import os
from typing import Dict, Any, Tuple

from .utils import utils

class BodyLanguageDetection:
    """Handles body language detection and image processing functionality."""
    
    def __init__(self, service):
        self.service = service
        self.tf_model = None
        self.label_encoder = None
        self.feature_scaler = None
        # Add RNN-specific variables
        self.rnn_model = None
        self.rnn_label_encoder = None
        self.rnn_feature_scaler = None
    
    def ensure_nn_model_loaded(self):
        """Ensure the neural network model is loaded if needed"""
        if self.service.current_model_type == "neural_network" and self.tf_model is None:
            try:
                import tensorflow as tf
                import pickle
                
                # Load TensorFlow model
                model_path = os.path.join(self.service.nn_model_dir, 'model.h5')
                self.tf_model = tf.keras.models.load_model(model_path)
                
                # Load label encoder
                with open(os.path.join(self.service.nn_model_dir, 'label_encoder.pkl'), 'rb') as f:
                    self.label_encoder = pickle.load(f)
                    
                # Load feature scaler
                with open(os.path.join(self.service.nn_model_dir, 'feature_scaler.pkl'), 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                    
                print("Neural network model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading neural network model: {str(e)}")
                return False
        return True
    
    def ensure_rnn_model_loaded(self):
        """Ensure the RNN model is loaded if needed"""
        if self.service.current_model_type == "rnn" and self.rnn_model is None:
            try:
                import tensorflow as tf
                import pickle
                
                # Load TensorFlow RNN model
                model_path = os.path.join(self.service.rnn_model_dir, 'model.h5')
                self.rnn_model = tf.keras.models.load_model(model_path)
                
                # Load label encoder
                with open(os.path.join(self.service.rnn_model_dir, 'label_encoder.pkl'), 'rb') as f:
                    self.rnn_label_encoder = pickle.load(f)
                    
                # Load feature scaler
                with open(os.path.join(self.service.rnn_model_dir, 'feature_scaler.pkl'), 'rb') as f:
                    self.rnn_feature_scaler = pickle.load(f)
                    
                print("RNN model loaded successfully")
                return True
            except Exception as e:
                print(f"Error loading RNN model: {str(e)}")
                return False
        return True
    
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and detect body language"""
        # Different handling based on model type
        if self.service.current_model_type == "neural_network":
            return self._process_image_nn(image_data)
        elif self.service.current_model_type == "rnn":
            return self._process_image_rnn(image_data)
        else:
            return self._process_image_default(image_data)
    
    def _process_image_default(self, image_data: bytes) -> Dict[str, Any]:
        """Process image using the default ML model"""
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
    
    def _process_image_nn(self, image_data: bytes) -> Dict[str, Any]:
        """Process image using the neural network model"""
        # Load neural network model if not already loaded
        if not self.ensure_nn_model_loaded():
            raise ValueError("Neural network model not loaded or failed to load")
            
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
                
                # Scale features
                X_scaled = self.feature_scaler.transform([row])
                
                # Predict with neural network
                prediction = self.tf_model.predict(X_scaled)[0]
                class_idx = np.argmax(prediction)
                confidence = float(prediction[class_idx])
                
                # Get class name
                body_language_class = self.label_encoder.inverse_transform([class_idx])[0]
                
                # Only return predictions with confidence above threshold
                if confidence < 0.65:
                    body_language_class = "Unknown"
                    
                return {
                    "class": body_language_class,
                    "confidence": confidence,
                    "landmarks": landmarks
                }
            except Exception as e:
                raise ValueError(f"Error during prediction: {str(e)}")
    
    def _process_image_rnn(self, image_data: bytes) -> Dict[str, Any]:
        """Process image using the RNN model"""
        # Load RNN model if not already loaded
        if not self.ensure_rnn_model_loaded():
            raise ValueError("RNN model not loaded or failed to load")
            
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
                
                # Scale features
                X_scaled = self.rnn_feature_scaler.transform([row])
                
                # Reshape for RNN input [samples, timesteps, features]
                X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                
                # Predict with RNN
                prediction = self.rnn_model.predict(X_scaled)[0]
                class_idx = np.argmax(prediction)
                confidence = float(prediction[class_idx])
                
                # Get class name
                body_language_class = self.rnn_label_encoder.inverse_transform([class_idx])[0]
                
                # Only return predictions with confidence above threshold
                if confidence < 0.65:  # You can adjust this threshold
                    body_language_class = "Unknown"
                    
                return {
                    "class": body_language_class,
                    "confidence": confidence,
                    "landmarks": landmarks
                }
            except Exception as e:
                raise ValueError(f"Error during RNN prediction: {str(e)}")
    
    def process_image_with_landmarks(self, image_data: bytes) -> Dict[str, Any]:
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
                # Convert image to base64 for JSON response
                _, buffer = cv2.imencode('.jpg', annotated_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "annotated_image": image_base64,
                    "success": False
                }
            
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
            
            # Predict body language based on the selected model type
            prediction_result = None
            
            try:
                if self.service.current_model_type == "neural_network":
                    if self.ensure_nn_model_loaded():
                        # Extract landmarks and predict
                        landmarks = utils.extract_landmarks(results)
                        row = utils.flatten_landmarks(landmarks)
                        row = utils.prepare_features_for_model(row, self.service)
                        
                        # Scale features
                        X_scaled = self.feature_scaler.transform([row])
                        
                        # Predict with neural network
                        prediction = self.tf_model.predict(X_scaled)[0]
                        class_idx = np.argmax(prediction)
                        confidence = float(prediction[class_idx])
                        
                        # Get class name
                        body_language_class = self.label_encoder.inverse_transform([class_idx])[0]
                        
                        if confidence >= 0.65:
                            prediction_result = {
                                "class": body_language_class,
                                "confidence": confidence
                            }
                elif self.service.current_model_type == "rnn":
                    if self.ensure_rnn_model_loaded():
                        # Extract landmarks and predict
                        landmarks = utils.extract_landmarks(results)
                        row = utils.flatten_landmarks(landmarks)
                        row = utils.prepare_features_for_model(row, self.service)
                        
                        # Scale features
                        X_scaled = self.rnn_feature_scaler.transform([row])
                        
                        # Reshape for RNN input [samples, timesteps, features]
                        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                        
                        # Predict with RNN
                        prediction = self.rnn_model.predict(X_scaled)[0]
                        class_idx = np.argmax(prediction)
                        confidence = float(prediction[class_idx])
                        
                        # Get class name
                        body_language_class = self.rnn_label_encoder.inverse_transform([class_idx])[0]
                        
                        if confidence >= 0.65:
                            prediction_result = {
                                "class": body_language_class,
                                "confidence": confidence
                            }
                else:
                    if self.service.model is not None:
                        # Get landmarks and predict
                        landmarks = utils.extract_landmarks(results)
                        row = utils.flatten_landmarks(landmarks)
                        row = utils.prepare_features_for_model(row, self.service)
                        
                        # Predict
                        X = pd.DataFrame([row])
                        body_language_class = self.service.model.predict(X)[0]
                        body_language_prob = self.service.model.predict_proba(X)[0]
                        
                        # Get confidence
                        class_idx = list(self.service.model.classes_).index(body_language_class)
                        confidence = body_language_prob[class_idx]
                        
                        if confidence >= 0.65:
                            prediction_result = {
                                "class": body_language_class,
                                "confidence": float(confidence)
                            }
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                prediction_result = None
                
            # Add prediction text to image if available
            if prediction_result:
                # Add text with prediction
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{prediction_result['class']}: {prediction_result['confidence']:.2f}"
                cv2.putText(annotated_image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            # Convert image to base64 for JSON response
            _, buffer = cv2.imencode('.jpg', annotated_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result = {
                "annotated_image": image_base64,
                "success": True
            }
            
            if prediction_result:
                result["class"] = prediction_result["class"]
                result["confidence"] = prediction_result["confidence"]
                
            return result 