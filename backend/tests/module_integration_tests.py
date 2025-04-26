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

class ModuleIntegrationTests(unittest.TestCase):
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

    def test_model_integration(self):
        """Test if the models can be loaded and used for prediction"""
        # Test TSN RGB model
        self.assertIsNotNone(self.predictor.tsn_rgb)
        self.assertEqual(self.predictor.tsn_rgb.training, False)
        
        # Test TSN Flow model
        self.assertIsNotNone(self.predictor.tsn_flow)
        self.assertEqual(self.predictor.tsn_flow.training, False)
        
        # Test STGCN model
        self.assertIsNotNone(self.predictor.stgcn)
        self.assertEqual(self.predictor.stgcn.training, False)

    def test_data_processing_pipeline(self):
        """Test the complete video processing pipeline"""
        # Process video
        processed_data = self.preprocessor.process_video(self.test_video_path)
        
        # Verify all required data is present
        self.assertIn('rgb', processed_data)
        self.assertIn('flow', processed_data)
        self.assertIn('joints', processed_data)
        self.assertIn('face', processed_data)
        
        # Make prediction
        result = self.predictor.predict(processed_data)
        
        # Verify prediction format
        self.assertIn('categorical', result)
        self.assertIn('continuous', result)
        self.assertIsInstance(result['categorical'], dict)
        self.assertIsInstance(result['continuous'], dict)

    def test_api_integration(self):
        """Test the complete API flow"""
        with open(self.test_video_path, 'rb') as f:
            test_data = {
                'video': (f, 'test_video.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            
            # Check response
            self.assertEqual(response.status_code, 200)
            result = response.json
            
            # Verify response format
            self.assertIn('categorical', result)
            self.assertIn('continuous', result)
            self.assertIsInstance(result['categorical'], dict)
            self.assertIsInstance(result['continuous'], dict)

    def test_error_handling_integration(self):
        """Test error handling across components"""
        # Test with invalid video file
        with open('invalid.mp4', 'wb') as f:
            f.write(b'invalid video data')
        with open('invalid.mp4', 'rb') as f:
            test_data = {
                'video': (f, 'invalid.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)
        os.remove('invalid.mp4')
        
        # Test with corrupted video file
        with open('corrupted.mp4', 'wb') as f:
            f.write(b'corrupted video data')
        with open('corrupted.mp4', 'rb') as f:
            test_data = {
                'video': (f, 'corrupted.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)
        os.remove('corrupted.mp4')

if __name__ == '__main__':
    unittest.main() 