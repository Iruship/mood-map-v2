import unittest
import sys
import os
import cv2
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from predict import EmotionPredictor
from pre_process import VideoPreprocessor

class FunctionalTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.predictor = EmotionPredictor()
        self.preprocessor = VideoPreprocessor(openpose_model_path="path/to/openpose/models")
        
        # Create a test video
        self.test_video_path = "test_video.mp4"
        self._create_test_video()

    def _create_test_video(self):
        """Create a simple test video with a face"""
        # Create a black background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a simple face (circle)
        center = (320, 240)
        radius = 100
        cv2.circle(frame, center, radius, (255, 255, 255), -1)
        
        # Add eyes
        cv2.circle(frame, (280, 220), 20, (0, 0, 0), -1)
        cv2.circle(frame, (360, 220), 20, (0, 0, 0), -1)
        
        # Add mouth (smile)
        cv2.ellipse(frame, (320, 280), (60, 30), 0, 0, 180, (0, 0, 0), 2)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.test_video_path, fourcc, 30.0, (640, 480))
        for _ in range(75):  # 2.5 seconds at 30 fps
            out.write(frame)
        out.release()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)

    def test_home_route(self):
        """Test if the home route returns 200 status code"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_route(self):
        """Test if the predict route works with video input"""
        with open(self.test_video_path, 'rb') as f:
            test_data = {
                'video': (f, 'test_video.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 200)
            self.assertIn('categorical', response.json)
            self.assertIn('continuous', response.json)

    def test_video_preprocessing(self):
        """Test if video preprocessing works correctly"""
        processed_data = self.preprocessor.process_video(self.test_video_path)
        
        # Check if all required tensors are present
        self.assertIn('rgb', processed_data)
        self.assertIn('flow', processed_data)
        self.assertIn('joints', processed_data)
        self.assertIn('face', processed_data)
        
        # Check tensor shapes
        self.assertEqual(processed_data['rgb'].shape[0], 25)  # num_segments
        self.assertEqual(processed_data['flow'].shape[0], 24)  # num_segments - 1
        self.assertEqual(processed_data['joints'].shape[0], 25)  # num_segments
        self.assertEqual(processed_data['face'].shape[0], 25)  # num_segments

    def test_emotion_prediction(self):
        """Test if emotion prediction returns valid output"""
        processed_data = self.preprocessor.process_video(self.test_video_path)
        result = self.predictor.predict(processed_data)
        
        # Check if results contain all required emotions
        self.assertIn('categorical', result)
        self.assertIn('continuous', result)
        
        # Check if all emotion categories are present
        expected_emotions = [
            "Peace", "Affection", "Esteem", "Anticipation", "Engagement", 
            "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise", 
            "Sympathy", "Doubt/Confusion", "Disconnect", "Fatigue", 
            "Embarrassment", "Yearning", "Disapproval", "Aversion", 
            "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", 
            "Fear", "Pain", "Suffering"
        ]
        
        for emotion in expected_emotions:
            self.assertIn(emotion, result['categorical'])
            self.assertIsInstance(result['categorical'][emotion], float)
            self.assertGreaterEqual(result['categorical'][emotion], 0.0)
            self.assertLessEqual(result['categorical'][emotion], 1.0)

    def test_invalid_input(self):
        """Test handling of invalid input"""
        # Test with no file
        response = self.app.post('/predict', data={}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        
        # Test with invalid file type
        with open('test.txt', 'w') as f:
            f.write('test')
        with open('test.txt', 'rb') as f:
            test_data = {
                'video': (f, 'test.txt')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)
        os.remove('test.txt')

if __name__ == '__main__':
    unittest.main() 