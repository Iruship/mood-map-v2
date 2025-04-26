import os
import cv2
import pickle
import numpy as np
import pandas as pd
import csv
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from .utils import utils

class BodyLanguageTraining:
    """Handles the model training and data recording functionality."""
    
    def __init__(self, service):
        self.service = service
    
    def record_landmark_data(self, image_data: bytes, class_name: str) -> bool:
        """Record landmark data for training"""
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
            
            try:
                # Extract detailed landmarks with consistent dimensions
                landmarks = utils.extract_landmarks(results)
                
                # Flatten landmarks
                flattened = utils.flatten_landmarks(landmarks)
                
                # Check for NaN values in flattened data
                if any(np.isnan(val) if isinstance(val, (int, float)) else False for val in flattened):
                    # Replace NaN values with 0
                    flattened = [0 if (isinstance(val, (int, float)) and np.isnan(val)) else val for val in flattened]
                    print("NaN values have been replaced with 0")
                
                # Check if file exists and has content
                file_exists = os.path.exists(self.service.data_path) and os.path.getsize(self.service.data_path) > 0
                
                # Handle feature count consistency
                if file_exists:
                    try:
                        # Read existing feature count
                        if os.path.exists(self.service.feature_count_path):
                            with open(self.service.feature_count_path, 'r') as f:
                                expected_count = int(f.read().strip())
                                flattened = utils.prepare_features_for_model(flattened, self.service, expected_count)
                        
                        # Fallback to checking CSV header
                        with open(self.service.data_path, 'r', newline='') as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            expected_cols = len(header)
                            
                            if len(flattened) + 1 != expected_cols:
                                flattened = utils.prepare_features_for_model(flattened, self.service, expected_cols - 1)
                    except Exception as e:
                        print(f"Warning: Error handling feature count: {str(e)}")
                else:
                    # First recording, save the feature count
                    with open(self.service.feature_count_path, 'w') as f:
                        f.write(str(len(flattened)))
                
                # Insert class at the beginning 
                flattened.insert(0, class_name)
                
                # Append to CSV
                mode = 'a' if file_exists else 'w'
                with open(self.service.data_path, mode, newline='') as f:
                    csv_writer = csv.writer(f)
                    
                    # Write header if file is new
                    if not file_exists:
                        header = ['class'] + [f'feature_{i}' for i in range(1, len(flattened))]
                        csv_writer.writerow(header)
                    
                    # Write data row
                    csv_writer.writerow(flattened)
                    
                return True
            except Exception as e:
                print(f"Error recording landmarks: {str(e)}")
                return False
    
    def train_model(self) -> Dict[str, Any]:
        """Train model on recorded data with advanced techniques"""
        # Check if data file exists
        if not os.path.exists(self.service.data_path) or os.path.getsize(self.service.data_path) == 0:
            raise ValueError("No training data available")
            
        # Load data
        df = pd.read_csv(self.service.data_path)
        
        # Ensure we have enough data
        if len(df) < 10:
            raise ValueError("Not enough training data. Need at least 10 samples.")
            
        # Handle missing values
        if df.isnull().values.any():
            print("Handling missing values in training data...")
            
            # Get class column first
            y = df.iloc[:, 0]
            # Handle NaN values in features 
            X = df.iloc[:, 1:]
            
            # Replace NaN values with column mean or 0
            for col in X.columns:
                if X[col].isnull().all():
                    X[col] = 0  # If all values are NaN, replace with 0
                else:
                    X[col] = X[col].fillna(X[col].mean())  # Fill NaN with column mean
            
            # Reconstruct dataframe
            df = pd.concat([y, X], axis=1)
            
        # Find max feature count and update feature count file
        feature_count = df.shape[1] - 1  # -1 because first column is the class label
        with open(self.service.feature_count_path, 'w') as f:
            f.write(str(feature_count))
        print(f"Training model with {feature_count} features")
        
        # Split features and target
        X = df.iloc[:, 1:]  # all columns except first (class)
        y = df.iloc[:, 0]   # first column (class)
        
        # Get unique classes
        classes = y.unique().tolist()
        num_classes = len(classes)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Select classifier based on data characteristics
        if num_classes > 5 or len(df) > 100:
            # More complex datasets - use Gradient Boosting
            classifier = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
            )
        elif num_classes <= 3 and len(df) >= 30:
            # Few classes with enough samples - use SVM
            classifier = SVC(probability=True, kernel='rbf', C=10, gamma='scale', random_state=42)
        else:
            # Default - use RandomForest 
            classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
            
        # Create pipeline with preprocessing
        pipeline = make_pipeline(StandardScaler(), classifier)
        
        # Train model
        model = pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_accuracy:.4f}, Testing accuracy: {test_accuracy:.4f}")
        
        # Save model
        with open(self.service.model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Update model instance
        self.service.model = model
        
        return {
            "accuracy": float(test_accuracy),
            "training_accuracy": float(train_accuracy),
            "classes": classes,
            "feature_count": feature_count,
            "samples_per_class": dict(y.value_counts().to_dict())
        } 