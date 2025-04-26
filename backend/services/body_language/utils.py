import mediapipe as mp
import numpy as np
from typing import Dict, List, Any

class BodyLanguageUtils:
    """Utility class for body language analysis with shared methods."""
    
    # MediaPipe landmark counts for consistent feature dimensions
    POSE_LANDMARK_COUNT = 33
    FACE_LANDMARK_COUNT = 468
    HAND_LANDMARK_COUNT = 21
    
    def __init__(self):
        # MediaPipe initialization
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
    
    def create_default_landmarks(self):
        """Create default landmark values to ensure consistent dimensions"""
        return {
            'pose': [[0, 0, 0, 0] for _ in range(self.POSE_LANDMARK_COUNT)],
            'face': [[0, 0, 0, 0] for _ in range(self.FACE_LANDMARK_COUNT)],
            'left_hand': [[0, 0, 0, 0] for _ in range(self.HAND_LANDMARK_COUNT)],
            'right_hand': [[0, 0, 0, 0] for _ in range(self.HAND_LANDMARK_COUNT)],
            'shoulder_angle': [0.0],
            'face_angle': [0.0],
            'left_hand_thumb_index': [0.0],
            'right_hand_thumb_index': [0.0]
        }
    
    def extract_landmarks(self, results) -> Dict[str, List]:
        """Extract landmarks from MediaPipe results with consistent dimensions"""
        # Start with default zero landmarks to ensure consistency
        landmarks = self.create_default_landmarks()
        
        # Extract pose landmarks if available
        if results.pose_landmarks:
            landmarks['pose'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            
            try:
                # Calculate shoulder angle
                right_shoulder = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
                left_shoulder = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
                
                if np.linalg.norm(right_shoulder - left_shoulder) > 0:
                    shoulder_angle = np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                           right_shoulder[0] - left_shoulder[0])
                    landmarks['shoulder_angle'] = [shoulder_angle]
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not calculate shoulder angle: {str(e)}")
            
        # Extract face landmarks if available
        if results.face_landmarks:
            landmarks['face'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.face_landmarks.landmark]
            
            try:
                # Calculate face orientation angle
                nose_tip = np.array([results.face_landmarks.landmark[1].x, results.face_landmarks.landmark[1].y])
                left_eye = np.array([results.face_landmarks.landmark[33].x, results.face_landmarks.landmark[33].y])
                right_eye = np.array([results.face_landmarks.landmark[263].x, results.face_landmarks.landmark[263].y])
                
                if np.linalg.norm(right_eye - left_eye) > 0:
                    eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                    landmarks['face_angle'] = [eye_angle]
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not calculate face angle: {str(e)}")
            
        # Extract left hand landmarks if available
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.left_hand_landmarks.landmark]
            
            try:
                # Calculate hand gesture features
                thumb_tip = np.array([results.left_hand_landmarks.landmark[4].x, results.left_hand_landmarks.landmark[4].y])
                index_tip = np.array([results.left_hand_landmarks.landmark[8].x, results.left_hand_landmarks.landmark[8].y])
                
                # Calculate distance between fingertips
                thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
                landmarks['left_hand_thumb_index'] = [thumb_index_dist]
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not calculate left hand features: {str(e)}")
            
        # Extract right hand landmarks if available
        if results.right_hand_landmarks:
            landmarks['right_hand'] = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.right_hand_landmarks.landmark]
            
            try:
                # Calculate hand gesture features for right hand
                thumb_tip = np.array([results.right_hand_landmarks.landmark[4].x, results.right_hand_landmarks.landmark[4].y])
                index_tip = np.array([results.right_hand_landmarks.landmark[8].x, results.right_hand_landmarks.landmark[8].y])
                
                # Calculate distance between fingertips
                thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
                landmarks['right_hand_thumb_index'] = [thumb_index_dist]
            except (ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not calculate right hand features: {str(e)}")
            
        return landmarks
    
    def flatten_landmarks(self, landmarks: Dict[str, List]) -> List[float]:
        """Flatten landmarks for model input"""
        flattened = []
        
        # Add all landmark categories in a consistent order
        for key in ['pose', 'face', 'left_hand', 'right_hand']:
            if key in landmarks:
                flattened.extend(np.array(landmarks[key]).flatten().tolist())
        
        # Add calculated features
        for key in ['shoulder_angle', 'face_angle', 'left_hand_thumb_index', 'right_hand_thumb_index']:
            if key in landmarks:
                flattened.extend(landmarks[key])
            
        return flattened
    
    def prepare_features_for_model(self, features, service, expected_count=None):
        """Ensure features match the expected count by padding or truncating"""
        if expected_count is None:
            # Try to get expected feature count from model
            if service.model:
                for step in service.model.steps:
                    if hasattr(step, 'n_features_in_'):
                        expected_count = step.n_features_in_
                        break
            
            # If still None, try to read from file
            if expected_count is None and os.path.exists(service.feature_count_path):
                try:
                    with open(service.feature_count_path, 'r') as f:
                        expected_count = int(f.read().strip())
                except (ValueError, FileNotFoundError):
                    pass
        
        # If we have an expected count, ensure our features match
        if expected_count is not None and len(features) != expected_count:
            if len(features) < expected_count:
                # Pad with zeros
                features.extend([0] * (expected_count - len(features)))
            else:
                # Truncate
                features = features[:expected_count]
                
            # Verify
            if len(features) != expected_count:
                raise ValueError(f"Feature count mismatch after adjustment: got {len(features)}, expected {expected_count}")
                
        return features


# Create a singleton instance
utils = BodyLanguageUtils()

# Missing import
import os 