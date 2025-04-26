import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import cv2
import os
import argparse
from typing import Dict, List, Any, Tuple

class BodyLanguageNNInference:
    """Neural Network-based body language detection inference."""
    
    # MediaPipe landmark counts for consistent feature dimensions
    POSE_LANDMARK_COUNT = 33
    FACE_LANDMARK_COUNT = 468
    HAND_LANDMARK_COUNT = 21
    
    def __init__(self, model_dir="body_language_nn_model"):
        """Initialize inference model with trained neural network."""
        self.model_dir = model_dir
        
        # MediaPipe initialization
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load model and associated data
        self.load_model()
    
    def load_model(self):
        """Load the trained neural network model and associated data."""
        try:
            # Load Keras model
            model_path = os.path.join(self.model_dir, 'model.h5')
            self.model = tf.keras.models.load_model(model_path)
            
            # Load label encoder
            with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            # Load feature scaler
            with open(os.path.join(self.model_dir, 'feature_scaler.pkl'), 'rb') as f:
                self.feature_scaler = pickle.load(f)
                
            print(f"Model loaded successfully from {self.model_dir}")
            print(f"Available classes: {self.label_encoder.classes_}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def create_default_landmarks(self):
        """Create default landmark values to ensure consistent dimensions."""
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
        """Extract landmarks from MediaPipe results with consistent dimensions."""
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
        """Flatten landmarks for model input."""
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
    
    def process_image(self, image) -> Dict[str, Any]:
        """Process image and detect body language."""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Process with MediaPipe
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            # Process the image
            results = holistic.process(image_rgb)
            
            # Get landmarks with consistent dimensions
            landmarks = self.extract_landmarks(results)
            
            # Flatten landmarks for prediction
            features = self.flatten_landmarks(landmarks)
            
            try:
                # Scale features
                X = np.array([features])
                X_scaled = self.feature_scaler.transform(X)
                
                # Make prediction
                prediction = self.model.predict(X_scaled)[0]
                class_idx = np.argmax(prediction)
                confidence = prediction[class_idx]
                
                # Get class name
                body_language_class = self.label_encoder.inverse_transform([class_idx])[0]
                
                # Only return predictions with confidence above threshold
                if confidence < 0.65:
                    body_language_class = "Unknown"
                    
                return {
                    "class": body_language_class,
                    "confidence": float(confidence),
                    "landmarks": landmarks
                }
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                return {
                    "class": "Error",
                    "confidence": 0.0,
                    "error": str(e)
                }
    
    def draw_landmarks(self, image, results) -> np.ndarray:
        """Draw landmarks on the image."""
        annotated_image = image.copy()
        
        # Draw face landmarks
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.face_landmarks, 
                self.mp_holistic.FACEMESH_CONTOURS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.pose_landmarks, 
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        # Draw hands landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image, 
                results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
            
        return annotated_image
    
    def process_image_with_prediction(self, image_path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process image from file, draw landmarks, and make prediction."""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            # Process the image
            results = holistic.process(image_rgb)
            
            # Draw landmarks
            annotated_image = self.draw_landmarks(image, results)
            
            # Make prediction
            prediction = self.process_image(image)
            
            # Add prediction to image
            if prediction["class"] != "Error" and prediction["confidence"] >= 0.65:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{prediction['class']}: {prediction['confidence']:.2f}"
                cv2.putText(annotated_image, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            return annotated_image, prediction
    
    def process_video(self, video_path, output_path=None):
        """Process video, make predictions, and optionally save the result."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize results list
        results = []
        
        # Process each frame
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2
        ) as holistic:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Display progress
                frame_idx += 1
                if frame_idx % 10 == 0:
                    print(f"Processing frame {frame_idx}/{frame_count} ({frame_idx/frame_count*100:.1f}%)")
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                mp_results = holistic.process(frame_rgb)
                
                # Draw landmarks
                annotated_frame = self.draw_landmarks(frame, mp_results)
                
                # Make prediction
                prediction = self.process_image(frame)
                results.append(prediction)
                
                # Add prediction to frame
                if prediction["class"] != "Error" and prediction["confidence"] >= 0.65:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"{prediction['class']}: {prediction['confidence']:.2f}"
                    cv2.putText(annotated_frame, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Write frame if output path provided
                if output_path:
                    out.write(annotated_frame)
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
            print(f"Output video saved to {output_path}")
        
        return results

def main():
    """Main function to run inference from command line."""
    parser = argparse.ArgumentParser(description='Body language detection with neural network')
    parser.add_argument('--model-dir', default='body_language_nn_model', help='Directory with trained model')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Image processing command
    image_parser = subparsers.add_parser('image', help='Process an image')
    image_parser.add_argument('image_path', help='Path to the image file')
    image_parser.add_argument('--output', help='Path to save output image', required=False)
    
    # Video processing command
    video_parser = subparsers.add_parser('video', help='Process a video')
    video_parser.add_argument('video_path', help='Path to the video file')
    video_parser.add_argument('--output', help='Path to save output video', required=False)
    
    args = parser.parse_args()
    
    # Initialize inference
    detector = BodyLanguageNNInference(model_dir=args.model_dir)
    
    if args.command == 'image':
        # Process image
        try:
            annotated_image, prediction = detector.process_image_with_prediction(args.image_path)
            
            # Print prediction
            print(f"Prediction: {prediction['class']} (Confidence: {prediction['confidence']:.4f})")
            
            # Save or display
            if args.output:
                cv2.imwrite(args.output, annotated_image)
                print(f"Output image saved to {args.output}")
            else:
                cv2.imshow('Body Language Detection', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            
    elif args.command == 'video':
        # Process video
        try:
            results = detector.process_video(args.video_path, args.output)
            
            # Print summary
            class_counts = {}
            for result in results:
                if result["class"] not in class_counts:
                    class_counts[result["class"]] = 0
                class_counts[result["class"]] += 1
            
            total_frames = len(results)
            print("\nPrediction Summary:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_frames * 100 if total_frames > 0 else 0
                print(f"{cls}: {count} frames ({percentage:.1f}%)")
        
        except Exception as e:
            print(f"Error processing video: {str(e)}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 