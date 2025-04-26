import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

class BodyLanguageNNTrainer:
    """Neural Network trainer for body language detection."""
    
    def __init__(self, data_path="coords.csv", feature_count_path="feature_count.txt", model_save_path="body_language_nn_model"):
        self.data_path = data_path
        self.feature_count_path = feature_count_path
        self.model_save_path = model_save_path
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
    
    def load_data(self):
        """Load and preprocess data."""
        print(f"Loading data from {self.data_path}...")
        
        # Load data
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
        
        print(f"Loaded dataframe with shape: {df.shape}")
        
        # Ensure we have enough data
        if len(df) < 10:
            print("Not enough training data. Need at least 10 samples.")
            return False
        
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
        
        # Save feature count to file
        feature_count = df.shape[1] - 1  # -1 because first column is class
        with open(self.feature_count_path, 'w') as f:
            f.write(str(feature_count))
        print(f"Feature count: {feature_count}")
        
        # Extract features and labels
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # One-hot encode
        num_classes = len(self.label_encoder.classes_)
        y_onehot = to_categorical(y_encoded, num_classes=num_classes)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.num_classes = num_classes
        self.feature_count = feature_count
        
        print(f"Processed data: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples, {num_classes} classes")
        return True
    
    def build_model(self):
        """Create neural network architecture."""
        input_shape = self.feature_count
        
        model = Sequential([
            # Input layer
            Dense(256, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        return model
    
    def train_model(self, epochs=100, batch_size=32):
        """Train the neural network model."""
        if not hasattr(self, 'model'):
            self.build_model()
        
        # Create directory for model save
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Callbacks for training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateauing
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\nTraining model with {epochs} epochs and batch size {batch_size}...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Save model info
        self.save_model()
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def save_model(self):
        """Save trained model and associated data."""
        # Save Keras model
        self.model.save(os.path.join(self.model_save_path, 'model.h5'))
        
        # Save label encoder and scaler
        with open(os.path.join(self.model_save_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(os.path.join(self.model_save_path, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save class mapping
        classes = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
        pd.DataFrame(list(classes.items()), columns=['Index', 'Class']).to_csv(
            os.path.join(self.model_save_path, 'class_mapping.csv'), index=False
        )
        
        print(f"\nModel and associated data saved to {self.model_save_path}")
    
    def plot_training_history(self, history):
        """Plot training accuracy and loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='train')
        ax1.plot(history.history['val_accuracy'], label='validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        # Loss plot
        ax2.plot(history.history['loss'], label='train')
        ax2.plot(history.history['val_loss'], label='validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()

def main():
    """Main function to train the neural network."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a neural network for body language detection')
    parser.add_argument('--data', default='coords.csv', help='Path to the feature data CSV file')
    parser.add_argument('--feature-count', default='feature_count.txt', help='Path to save feature count')
    parser.add_argument('--save-dir', default='body_language_nn_model', help='Directory to save model and data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = BodyLanguageNNTrainer(
        data_path=args.data,
        feature_count_path=args.feature_count,
        model_save_path=args.save_dir
    )
    
    # Load and preprocess data
    if trainer.load_data():
        # Train model
        trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)
    else:
        print("Failed to load and preprocess data. Exiting.")

if __name__ == "__main__":
    main() 