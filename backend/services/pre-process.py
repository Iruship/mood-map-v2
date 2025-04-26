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
        """
        Initialize video preprocessor with required models and parameters
        
        Args:
            openpose_model_path: Path to the OpenPose model files
            temp_dir: Directory to store temporary processing files
        """
        # Initialize OpenPose for skeleton detection
        self.openpose = self._load_openpose(openpose_model_path)
        
        # Create temporary directory for processing artifacts
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Define normalization parameters for RGB images (ImageNet standards)
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        
        # Define normalization parameters for optical flow
        self.flow_mean = [0.5]  # Center flow values around 0.5
        self.flow_std = [np.mean(self.rgb_std)]  # Use average of RGB std for flow
        
        # Setup transforms for RGB image preprocessing
        # - Resize to 224x224 (standard input size for many CNNs)
        # - Convert to tensor (HWC to CHW and [0,255] to [0,1])
        # - Normalize using ImageNet statistics
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.rgb_mean, std=self.rgb_std)
        ])

    def _load_openpose(self, model_path):
        """
        Load OpenPose model for human pose estimation
        
        Args:
            model_path: Path to OpenPose model directory
            
        Returns:
            Initialized OpenPose wrapper object
        """
        try:
            # Configure OpenPose parameters
            params = {
                "model_folder": model_path,
                "model_pose": "BODY_25",  # Use BODY_25 model which provides 25 body keypoints
                "net_resolution": "-1x368"  # Default resolution for processing
            }
            
            # Initialize OpenPose wrapper
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            
            return opWrapper
        except ImportError:
            print("Error: OpenPose Python API not found. Please install OpenPose properly.")
            raise

    def extract_frames(self, video_path, num_segments=25):
        """
        Extract frames from video at regular intervals
        
        Args:
            video_path: Path to the video file
            num_segments: Number of frames to extract
            
        Returns:
            List of extracted frames as numpy arrays
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract (evenly distributed)
        frames_to_extract = np.linspace(0, total_frames-1, num_segments, dtype=int)
        
        frames = []
        for frame_idx in frames_to_extract:
            # Set position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        # Release video capture object
        cap.release()
        return frames

    def compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames: List of video frames
            
        Returns:
            List of optical flow fields (x and y components)
        """
        flows = []
        for i in range(len(frames)-1):
            # Convert frames to grayscale for optical flow calculation
            prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            
            # Calculate dense optical flow using Farneback method
            # Parameters:
            # - prev, next: input frames
            # - None: output flow (will be allocated)
            # - 0.5: pyramid scale
            # - 3: pyramid levels
            # - 15: window size
            # - 3: iterations
            # - 5: poly_n (pixel neighborhood size)
            # - 1.2: poly_sigma (standard deviation)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        return flows

    def extract_joints(self, frames):
        """
        Extract body joints using OpenPose
        
        Args:
            frames: List of video frames
            
        Returns:
            Numpy array of joint positions for each frame
        """
        joints = []
        for frame in frames:
            # Process frame with OpenPose to extract keypoints
            # The exact implementation depends on your OpenPose setup
            keypoints = self.openpose.process(frame)
            joints.append(keypoints)
        return np.array(joints)

    def get_face_crops(self, frames, joints):
        """
        Extract face regions using joint information from OpenPose
        
        Args:
            frames: List of video frames
            joints: Joint positions extracted by OpenPose
            
        Returns:
            List of face crop images as PIL Images
        """
        face_crops = []
        for frame, joint in zip(frames, joints):
            # Use head keypoints to define face region
            # These indices correspond to face/head keypoints in BODY_25 model
            head_indices = [0,1,14,15,16,17]  # OpenPose head keypoints
            head_points = joint[head_indices]
            
            if not np.isnan(head_points).all():
                # Calculate bounding box from head keypoints
                x_min = int(np.nanmin(head_points[:,0]))
                y_min = int(np.nanmin(head_points[:,1]))
                x_max = int(np.nanmax(head_points[:,0]))
                y_max = int(np.nanmax(head_points[:,1]))
                
                # Add margin around face (30% of face size)
                margin_x = int(0.3 * (x_max - x_min))
                margin_y = int(0.3 * (y_max - y_min))
                
                # Ensure coordinates are within image boundaries
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(frame.shape[1], x_max + margin_x)
                y_max = min(frame.shape[0], y_max + margin_y)
                
                # Crop the face region
                face_crop = frame[y_min:y_max, x_min:x_max]
                face_crop = Image.fromarray(face_crop)
            else:
                # If no face detected, create a blank image
                face_crop = Image.new('RGB', (224, 224))
                
            face_crops.append(face_crop)
            
        return face_crops

    def process_video(self, video_path):
        """
        Process video and return all required inputs for the emotion prediction models
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing processed tensors for RGB, flow, joints, and face
        """
        # Extract frames at regular intervals
        frames = self.extract_frames(video_path)
        
        # Extract skeletal joints using OpenPose
        joints = self.extract_joints(frames)
        
        # Compute optical flow between consecutive frames
        flows = self.compute_optical_flow(frames)
        
        # Extract face regions using joint information
        face_crops = self.get_face_crops(frames, joints)
        
        # Process frames for RGB stream (convert to PIL and apply transforms)
        rgb_frames = [Image.fromarray(f) for f in frames]
        rgb_tensors = torch.stack([self.transform(f) for f in rgb_frames])
        
        # Process face crops
        face_tensors = torch.stack([self.transform(f) for f in face_crops])
        
        # Create transform for optical flow (different normalization)
        flow_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.flow_mean, std=self.flow_std)
        ])
        
        # Process optical flow (x and y components separately)
        flow_tensors = []
        for flow in flows:
            # Extract x and y components of the flow
            flow_x = Image.fromarray(flow[..., 0])
            flow_y = Image.fromarray(flow[..., 1])
            
            # Apply transforms to each component
            flow_x = flow_transform(flow_x)
            flow_y = flow_transform(flow_y)
            
            # Concatenate x and y components along channel dimension
            flow_tensors.append(torch.cat([flow_x, flow_y], dim=0))
        
        # Stack all flow tensors
        flow_tensors = torch.stack(flow_tensors)
        
        # Return dictionary with all processed inputs
        return {
            'rgb': rgb_tensors,         # RGB frames for appearance analysis
            'flow': flow_tensors,       # Optical flow for motion analysis
            'joints': torch.from_numpy(joints).float(),  # Skeletal joints for pose analysis
            'face': face_tensors        # Face crops for facial expression analysis
        }

    def cleanup(self):
        """
        Clean up temporary files created during processing
        """
        import shutil
        # Remove the entire temporary directory and its contents
        shutil.rmtree(self.temp_dir) 