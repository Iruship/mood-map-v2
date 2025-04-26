import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from openpose import pyopenpose as op
from PIL import Image
import os
from pathlib import Path

class VideoPreprocessor:
    def __init__(self, openpose_model_path, temp_dir="temp_processing"):
        """Initialize video preprocessor with required models and parameters"""
        self.openpose = self._load_openpose(openpose_model_path)
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Define normalization parameters
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.flow_mean = [0.5]
        self.flow_std = [np.mean(self.rgb_std)]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])

    def _load_openpose(self, model_path):
        """Load OpenPose model - implement based on your OpenPose setup"""
        try:
            # Configure OpenPose parameters
            params = {
                "model_folder": model_path,
                "model_pose": "BODY_25",  # Common pose model
                "net_resolution": "-1x368"  # Default resolution
            }
            
            # Initialize OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            
            return opWrapper
        except ImportError:
            print("Error: OpenPose Python API not found. Please install OpenPose properly.")
            raise

    def extract_frames(self, video_path, num_segments=25):
        """Extract frames from video at regular intervals"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = np.linspace(0, total_frames-1, num_segments, dtype=int)
        
        frames = []
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames

    def compute_optical_flow(self, frames):
        """Compute optical flow between consecutive frames"""
        flows = []
        for i in range(len(frames)-1):
            prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        return flows

    def extract_joints(self, frames):
        """Extract body joints using OpenPose"""
        joints = []
        for frame in frames:
            # Process frame with OpenPose
            # This needs to be implemented based on your OpenPose setup
            keypoints = self.openpose.process(frame)
            joints.append(keypoints)
        return np.array(joints)

    def get_face_crops(self, frames, joints):
        """Extract face regions using joint information"""
        face_crops = []
        for frame, joint in zip(frames, joints):
            # Use head keypoints to define face region
            head_indices = [0,1,14,15,16,17]  # OpenPose head keypoints
            head_points = joint[head_indices]
            
            if not np.isnan(head_points).all():
                x_min = int(np.nanmin(head_points[:,0]))
                y_min = int(np.nanmin(head_points[:,1]))
                x_max = int(np.nanmax(head_points[:,0]))
                y_max = int(np.nanmax(head_points[:,1]))
                
                # Add margin
                margin_x = int(0.3 * (x_max - x_min))
                margin_y = int(0.3 * (y_max - y_min))
                
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(frame.shape[1], x_max + margin_x)
                y_max = min(frame.shape[0], y_max + margin_y)
                
                face_crop = frame[y_min:y_max, x_min:x_max]
                face_crop = Image.fromarray(face_crop)
            else:
                face_crop = Image.new('RGB', (224, 224))
                
            face_crops.append(face_crop)
            
        return face_crops

    def process_video(self, video_path):
        """Process video and return all required inputs for the models"""
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Extract joints
        joints = self.extract_joints(frames)
        
        # Compute optical flow
        flows = self.compute_optical_flow(frames)
        
        # Get face crops
        face_crops = self.get_face_crops(frames, joints)
        
        # Process frames for RGB stream
        rgb_frames = [Image.fromarray(f) for f in frames]
        rgb_tensors = torch.stack([self.transform(f) for f in rgb_frames])
        
        # Process faces
        face_tensors = torch.stack([self.transform(f) for f in face_crops])
        
        # Process optical flow
        flow_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.flow_mean, std=self.flow_std)
        ])
        
        flow_tensors = []
        for flow in flows:
            flow_x = Image.fromarray(flow[..., 0])
            flow_y = Image.fromarray(flow[..., 1])
            flow_x = flow_transform(flow_x)
            flow_y = flow_transform(flow_y)
            flow_tensors.append(torch.cat([flow_x, flow_y], dim=0))
        flow_tensors = torch.stack(flow_tensors)
        
        return {
            'rgb': rgb_tensors,
            'flow': flow_tensors,
            'joints': torch.from_numpy(joints).float(),
            'face': face_tensors
        }

    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir) 