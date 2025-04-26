import torch
import numpy as np
from model.models import TSN
from model.stgcn import Model as STGCN
import torch.nn.functional as F

class EmotionPredictor:
    def __init__(self, model_dir="pretrained", device="cuda"):
        """
        Initialize the EmotionPredictor with models for emotion and depression analysis.
        
        Args:
            model_dir (str): Directory containing pretrained model weights
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_dir = model_dir
        
        # Load the three complementary models for multimodal analysis
        self.tsn_rgb = self._load_tsn_rgb()  # RGB appearance model
        self.tsn_flow = self._load_tsn_flow()  # Optical flow motion model
        self.stgcn = self._load_stgcn()  # Skeleton-based motion model
        
        # Define the 26 categorical emotion labels used in our model
        # These emotions represent a comprehensive set of affective states
        self.categorical_emotions = [
            "Peace", "Affection", "Esteem", "Anticipation", "Engagement", 
            "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise", 
            "Sympathy", "Doubt/Confusion", "Disconnect", "Fatigue", 
            "Embarrassment", "Yearning", "Disapproval", "Aversion", 
            "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", 
            "Fear", "Pain", "Suffering"
        ]
        
        # Define the 3 continuous emotion dimensions
        # Valence: positive vs negative feelings
        # Arousal: calm vs excited
        # Dominance: feeling of control vs feeling controlled
        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]
        
        # Depression-related emotions and their weights (negative indicators)
        # Higher weights indicate stronger correlation with depression
        # These weights are based on psychological research on depression markers
        self.depression_indicators = {
            "Sadness": 0.25,      # Strongest depression indicator
            "Fatigue": 0.15,      # Physical and mental exhaustion
            "Disconnect": 0.15,   # Social withdrawal and isolation
            "Pain": 0.10,         # Emotional or physical pain
            "Suffering": 0.10,    # Prolonged distress
            "Fear": 0.10,         # Anxiety and worry
            "Disquietment": 0.10, # Restlessness and agitation
            "Doubt/Confusion": 0.05 # Cognitive symptoms
        }
        
        # Positive emotion weights (inverse relationship with depression)
        # These emotions typically decrease with depression
        self.positive_indicators = {
            "Happiness": 0.15,    # General positive affect
            "Peace": 0.15,        # Calmness and contentment
            "Confidence": 0.10,   # Self-assurance
            "Engagement": 0.10,   # Interest in activities
            "Pleasure": 0.10,     # Enjoyment and satisfaction
            "Excitement": 0.10,   # Enthusiasm
            "Affection": 0.10,    # Warmth toward others
            "Esteem": 0.10,       # Self-worth
            "Anticipation": 0.10  # Positive future outlook
        }

    def _load_tsn_rgb(self):
        """
        Load the RGB-based Temporal Segment Network model
        
        This model processes visual appearance features from:
        - Body movements
        - Facial expressions
        - Environmental context
        - Scene attributes
        
        Returns:
            Loaded and initialized TSN model for RGB data
        """
        model = TSN(
            logger=None,
            num_classes=26,           # 26 emotion categories
            num_dimensions=3,         # 3 continuous dimensions (VAD)
            rgb_body=True,            # Process body RGB data
            rgb_context=True,         # Process context RGB data
            rgb_face=True,            # Process face RGB data
            flow_body=False,          # No optical flow for body
            flow_context=False,       # No optical flow for context
            flow_face=False,          # No optical flow for face
            scenes=True,              # Include scene recognition
            attributes=True,          # Include attribute recognition
            depth=False,              # No depth information
            rgbdiff_body=False,       # No RGB difference for body
            rgbdiff_context=False,    # No RGB difference for context
            rgbdiff_face=False,       # No RGB difference for face
            arch='resnet50',          # ResNet50 backbone
            consensus_type='avg',     # Average pooling for consensus
            partial_bn=True,          # Partial batch normalization
            embed=True                # Use embedding features
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(f"{self.model_dir}/tsn_rgb_bcfsa_embed_pbn.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)  # Move model to GPU/CPU
        model.eval()  # Set to evaluation mode
        return model

    def _load_tsn_flow(self):
        """
        Load the Optical Flow-based Temporal Segment Network model
        
        This model processes motion features from:
        - Body movements over time
        - Facial expression changes
        - Environmental context changes
        
        Returns:
            Loaded and initialized TSN model for optical flow data
        """
        model = TSN(
            logger=None,
            num_classes=26,           # 26 emotion categories
            num_dimensions=3,         # 3 continuous dimensions (VAD)
            rgb_body=False,           # No RGB for body
            rgb_context=False,        # No RGB for context
            rgb_face=False,           # No RGB for face
            flow_body=True,           # Process body optical flow
            flow_context=True,        # Process context optical flow
            flow_face=True,           # Process face optical flow
            scenes=False,             # No scene recognition
            attributes=False,         # No attribute recognition
            depth=False,              # No depth information
            rgbdiff_body=False,       # No RGB difference for body
            rgbdiff_context=False,    # No RGB difference for context
            rgbdiff_face=False,       # No RGB difference for face
            arch='resnet50',          # ResNet50 backbone
            consensus_type='avg',     # Average pooling for consensus
            partial_bn=True,          # Partial batch normalization
            embed=True                # Use embedding features
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(f"{self.model_dir}/tsn_flow_bcf_embed_pbn.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)  # Move model to GPU/CPU
        model.eval()  # Set to evaluation mode
        return model

    def _load_stgcn(self):
        """
        Load the Spatial-Temporal Graph Convolutional Network model
        
        This model processes skeleton-based features:
        - Joint positions and movements
        - Body posture changes over time
        - Spatial relationships between body parts
        
        Returns:
            Loaded and initialized STGCN model
        """
        model = STGCN(
            in_channels=3,                    # 3D joint coordinates (x,y,z)
            num_class=26,                     # 26 emotion categories
            num_dim=3,                        # 3 continuous dimensions (VAD)
            layout='openpose',                # OpenPose skeleton format
            strategy='spatial',               # Spatial graph convolution
            max_hop=1,                        # Maximum graph hop distance
            dilation=1,                       # Dilation rate
            edge_importance_weighting=True    # Learn edge importance
        )
        
        # Load pre-trained weights
        checkpoint = torch.load(f"{self.model_dir}/stgcn_spatial_kinetics.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)  # Move model to GPU/CPU
        model.eval()  # Set to evaluation mode
        return model

    def calculate_depression_score(self, categorical_scores):
        """
        Calculate depression score based on emotional indicators
        
        This function combines:
        1. Presence of negative emotions associated with depression
        2. Absence of positive emotions (inverse relationship)
        
        Args:
            categorical_scores (dict): Dictionary of emotion scores (0-1 range)
            
        Returns:
            float: Depression score (0-1 range, higher = more depressive indicators)
        """
        # Calculate negative indicators contribution
        # Higher scores in negative emotions increase depression score
        negative_score = sum(
            categorical_scores[emotion] * weight 
            for emotion, weight in self.depression_indicators.items()
        )
        
        # Calculate positive indicators contribution (inverse relationship)
        # Lower scores in positive emotions increase depression score
        positive_score = sum(
            (1 - categorical_scores[emotion]) * weight 
            for emotion, weight in self.positive_indicators.items()
        )
        
        # Combine scores (normalized to 0-1 range)
        # Average of negative presence and positive absence
        depression_score = (negative_score + positive_score) / 2
        
        return depression_score

    def predict(self, processed_data):
        """
        Make predictions using all models
        
        This function:
        1. Prepares input data for each model
        2. Runs inference on all three models
        3. Combines predictions with weighted averaging
        4. Formats results into a structured dictionary
        5. Calculates depression score based on emotional indicators
        
        Args:
            processed_data (dict): Dictionary containing preprocessed video features
                                  (RGB frames, optical flow, skeleton joints)
                                  
        Returns:
            dict: Prediction results including categorical emotions, 
                 continuous dimensions, and depression score
        """
        with torch.no_grad():  # Disable gradient calculation for inference
            # Prepare inputs for each model
            # Move tensors to the appropriate device (GPU/CPU)
            rgb_input = {
                'body': processed_data['rgb'].to(self.device),      # Body region RGB
                'face': processed_data['face'].to(self.device),     # Face region RGB
                'context': processed_data['rgb'].to(self.device)    # Full frame as context
            }
            
            flow_input = {
                'body': processed_data['flow'].to(self.device),     # Body region flow
                'face': processed_data['flow'].to(self.device),     # Face region flow
                'context': processed_data['flow'].to(self.device)   # Full frame flow
            }
            
            skeleton_input = {
                'skeleton': processed_data['joints'].to(self.device)  # Body joint positions
            }
            
            # Get predictions from each model
            # Each model returns both categorical and continuous emotion predictions
            rgb_out = self.tsn_rgb(rgb_input, num_segments=25)        # RGB model (25 temporal segments)
            flow_out = self.tsn_flow(flow_input, num_segments=25)     # Flow model (25 temporal segments)
            skeleton_out = self.stgcn(skeleton_input)                 # Skeleton model
            
            # Combine predictions using weighted average
            # Weights reflect the relative importance of each modality
            weights = [0.4, 0.3, 0.3]  # RGB (40%), Flow (30%), Skeleton (30%)
            
            # Collect predictions from all models
            cat_preds = []   # Categorical emotion predictions
            cont_preds = []  # Continuous emotion dimension predictions
            
            # Apply sigmoid to convert logits to probabilities (0-1 range)
            for out in [rgb_out, flow_out, skeleton_out]:
                cat_preds.append(torch.sigmoid(out['categorical']))
                cont_preds.append(torch.sigmoid(out['continuous']))
            
            # Stack predictions for weighted averaging
            cat_preds = torch.stack(cat_preds)    # Shape: [3, 1, 26]
            cont_preds = torch.stack(cont_preds)  # Shape: [3, 1, 3]
            
            # Calculate weighted averages
            # Reshape weights tensor to allow broadcasting
            weighted_cat = (cat_preds * torch.tensor(weights).view(-1, 1, 1).to(self.device)).sum(0)
            weighted_cont = (cont_preds * torch.tensor(weights).view(-1, 1, 1).to(self.device)).sum(0)
            
            # Format results into a structured dictionary
            # Convert tensor values to Python floats for JSON serialization
            results = {
                'categorical': {
                    emotion: float(score)  # Map each emotion to its score
                    for emotion, score in zip(self.categorical_emotions, weighted_cat[0])
                },
                'continuous': {
                    emotion: float(score)  # Map each dimension to its score
                    for emotion, score in zip(self.continuous_emotions, weighted_cont[0])
                },
                'depression_score': float(self.calculate_depression_score({
                    emotion: float(score)  # Calculate depression score from categorical emotions
                    for emotion, score in zip(self.categorical_emotions, weighted_cat[0])
                }))
            }
            
            return results 