"""
Configuration file for body language detection thresholds.
These settings control the sensitivity and accuracy of the body language detection.
"""

# MediaPipe detection confidence thresholds
DETECTION_CONFIDENCE = 0.7  # Minimum confidence for detection (0.0-1.0)
TRACKING_CONFIDENCE = 0.7   # Minimum confidence for tracking (0.0-1.0)
MODEL_COMPLEXITY = 2        # MediaPipe model complexity (0, 1, or 2)

# Prediction confidence thresholds
PREDICTION_CONFIDENCE = 0.65  # Minimum confidence required for prediction (0.0-1.0)

# Feature processing settings
# Note: These settings are for configuration only. In critical code paths, 
# feature padding/truncation is always enabled to ensure model compatibility
PAD_MISSING_FEATURES = True    # Whether to pad missing features with zeros
HANDLE_NAN_VALUES = True       # Whether to handle NaN values in data

# Training parameters
MIN_TRAINING_SAMPLES = 10      # Minimum number of samples required for training 