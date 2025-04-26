import os
import pickle
from typing import Dict, List, Any

from .training import BodyLanguageTraining
from .detection import BodyLanguageDetection
from .data_management import BodyLanguageDataManager

class BodyLanguageService:
    """
    Main service class that combines training, detection and data management functionality
    for body language analysis.
    """
    def __init__(self):
        # Model and data paths
        self.model_dir = "./ml/body_language"
        self.data_path = os.path.join(self.model_dir, "coords.csv")
        self.model_path = os.path.join(self.model_dir, "body_language_model.pkl")
        self.nn_model_dir = os.path.join(self.model_dir, "body_language_nn_model")
        self.rnn_model_dir = os.path.join(self.model_dir, "body_language_rnn_model")
        self.feature_count_path = os.path.join(self.model_dir, "feature_count.txt")
        
        # Initialize model
        self.model = None
        self.nn_model = None
        self.rnn_model = None
        self.current_model_type = "default"  # Options: "default", "neural_network", "rnn"
        
        # Initialize components
        self.training = BodyLanguageTraining(self)
        self.detection = BodyLanguageDetection(self)
        self.data_manager = BodyLanguageDataManager(self)
        
        # Initialize directories and model
        self.initialize()
    
    def initialize(self):
        """Initialize directories and model if available"""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.nn_model_dir, exist_ok=True)
        os.makedirs(self.rnn_model_dir, exist_ok=True)
        
        # Create empty CSV file if it doesn't exist
        if not os.path.exists(self.data_path):
            with open(self.data_path, 'w', newline='') as f:
                pass
        else:
            # Try to repair the data file if it exists
            try:
                self.repair_data_file()
            except Exception as e:
                print(f"Warning: Could not repair data file: {str(e)}")
                
        # Load default model if exists
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Body language model loaded successfully")
            except Exception as e:
                print(f"Error loading body language model: {str(e)}")
                self.model = None
        
        # Check if neural network model exists
        nn_model_path = os.path.join(self.nn_model_dir, 'model.h5')
        if os.path.exists(nn_model_path):
            try:
                # We'll load this on demand when switching to NN model
                print("Neural network model files found")
            except Exception as e:
                print(f"Error checking neural network model: {str(e)}")
        
        # Check if RNN model exists
        rnn_model_path = os.path.join(self.rnn_model_dir, 'model.h5')
        if os.path.exists(rnn_model_path):
            try:
                # We'll load this on demand when switching to RNN model
                print("RNN model files found")
            except Exception as e:
                print(f"Error checking RNN model: {str(e)}")
    
    def get_current_model_type(self) -> str:
        """Get the current model type being used"""
        return self.current_model_type
    
    def get_available_model_types(self) -> List[str]:
        """Get list of available model types that can be selected"""
        available_models = ["default"]
        
        # Check if neural network model exists
        nn_model_path = os.path.join(self.nn_model_dir, 'model.h5')
        if os.path.exists(nn_model_path):
            available_models.append("neural_network")
        
        # Check if RNN model exists
        rnn_model_path = os.path.join(self.rnn_model_dir, 'model.h5')
        if os.path.exists(rnn_model_path):
            available_models.append("rnn")
            
        return available_models
    
    def switch_model(self, model_type: str) -> bool:
        """Switch to a different model type"""
        if model_type not in self.get_available_model_types():
            return False
            
        # If already using this model type, do nothing
        if model_type == self.current_model_type:
            return True
            
        if model_type == "default":
            # Switch to default model
            if self.model is None:
                try:
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except Exception as e:
                    print(f"Error loading default model: {str(e)}")
                    return False
            self.current_model_type = "default"
            return True
            
        elif model_type == "neural_network":
            # Switch to neural network model
            # The actual loading happens in the detection module when needed
            self.current_model_type = "neural_network"
            return True
        
        elif model_type == "rnn":
            # Switch to RNN model
            # The actual loading happens in the detection module when needed
            self.current_model_type = "rnn"
            return True
            
        return False
    
    # Proxy methods to the specific service components
    
    # Training methods
    def train_model(self) -> Dict[str, Any]:
        """Train model on recorded data"""
        return self.training.train_model()
        
    def record_landmark_data(self, image_data: bytes, class_name: str) -> bool:
        """Record landmark data for training"""
        return self.training.record_landmark_data(image_data, class_name)
    
    # Detection methods
    def process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and detect body language"""
        return self.detection.process_image(image_data)
    
    def process_image_with_landmarks(self, image_data: bytes) -> Dict[str, Any]:
        """Process image and return it with landmarks drawn on it"""
        return self.detection.process_image_with_landmarks(image_data)
    
    # Data management methods
    def get_available_classes(self) -> List[str]:
        """Get list of available classes in the training data"""
        return self.data_manager.get_available_classes()
    
    def get_model_path(self) -> str:
        """Get path to the trained model file"""
        if self.current_model_type == "neural_network":
            return os.path.join(self.nn_model_dir, 'model.h5')
        elif self.current_model_type == "rnn":
            return os.path.join(self.rnn_model_dir, 'model.h5')
        return self.model_path
    
    def delete_all_data(self) -> bool:
        """Delete all training data, models, and feature count information"""
        return self.data_manager.delete_all_data()
    
    def repair_data_file(self) -> bool:
        """Repair the data file by ensuring all rows have the same number of columns"""
        return self.data_manager.repair_data_file() 