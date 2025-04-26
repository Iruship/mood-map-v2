import unittest
import sys
import os
import time
import cv2
import numpy as np
import torch
import requests
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from predict import EmotionPredictor
from pre_process import VideoPreprocessor

class NonFunctionalTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.predictor = EmotionPredictor()
        self.preprocessor = VideoPreprocessor(openpose_model_path="path/to/openpose/models")
        self.base_url = "http://localhost:5000"
        
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

    def test_accuracy(self):
        """Test the accuracy of emotion predictions"""
        # Process video
        processed_data = self.preprocessor.process_video(self.test_video_path)
        
        # Make multiple predictions
        predictions = []
        for _ in range(10):
            result = self.predictor.predict(processed_data)
            predictions.append(result)
        
        # Check consistency of predictions
        first_pred = predictions[0]
        for pred in predictions[1:]:
            # Check if predictions are similar (within 0.1 difference)
            for emotion in first_pred['categorical']:
                self.assertLess(
                    abs(first_pred['categorical'][emotion] - pred['categorical'][emotion]),
                    0.1,
                    f"Predictions for {emotion} vary too much"
                )

    def test_performance(self):
        """Test the performance of the prediction endpoint"""
        num_requests = 10
        start_time = time.time()
        
        with open(self.test_video_path, 'rb') as f:
            test_data = {
                'video': (f, 'test_video.mp4')
            }
            for _ in range(num_requests):
                response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
                self.assertEqual(response.status_code, 200)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        
        # Assert that average response time is less than 5 seconds
        self.assertLess(avg_time, 5.0)

    def test_load_balancing(self):
        """Test the system's ability to handle concurrent requests"""
        num_requests = 5
        successful_requests = 0
        
        def make_request():
            try:
                with open(self.test_video_path, 'rb') as f:
                    files = {'video': (f, 'test_video.mp4')}
                    response = requests.post(f"{self.base_url}/predict", files=files)
                    return response.status_code == 200
            except:
                return False
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(lambda _: make_request(), range(num_requests)))
            successful_requests = sum(results)
        
        # Assert that at least 80% of requests were successful
        self.assertGreaterEqual(successful_requests / num_requests, 0.8)

    def test_security(self):
        """Test basic security measures"""
        # Test with large video file
        large_video = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        cv2.imwrite('large_video.mp4', large_video)
        with open('large_video.mp4', 'rb') as f:
            test_data = {
                'video': (f, 'large_video.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)  # Should reject large files
        os.remove('large_video.mp4')
        
        # Test with malicious file
        with open('malicious.mp4', 'wb') as f:
            f.write(b'<?php system($_GET["cmd"]); ?>')
        with open('malicious.mp4', 'rb') as f:
            test_data = {
                'video': (f, 'malicious.mp4')
            }
            response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 400)  # Should reject malicious files
        os.remove('malicious.mp4')
        
        # Test rate limiting (if implemented)
        for _ in range(20):  # Make many rapid requests
            with open(self.test_video_path, 'rb') as f:
                test_data = {
                    'video': (f, 'test_video.mp4')
                }
                response = self.app.post('/predict', data=test_data, content_type='multipart/form-data')
        self.assertNotEqual(response.status_code, 429)  # Should not be rate limited

if __name__ == '__main__':
    unittest.main() 