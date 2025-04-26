import torch
import numpy as np
from model.models import TSN
from model.stgcn import Model as STGCN
import torch.nn.functional as F

class EmotionPredictor:
    def __init__(self, model_dir="pretrained", device="cuda"):
        self.device = device
        self.model_dir = model_dir
        
        # Load models
        self.tsn_rgb = self._load_tsn_rgb()
        self.tsn_flow = self._load_tsn_flow()
        self.stgcn = self._load_stgcn()
        
        # Emotion labels
        self.categorical_emotions = [
            "Peace", "Affection", "Esteem", "Anticipation", "Engagement", 
            "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise", 
            "Sympathy", "Doubt/Confusion", "Disconnect", "Fatigue", 
            "Embarrassment", "Yearning", "Disapproval", "Aversion", 
            "Annoyance", "Anger", "Sensitivity", "Sadness", "Disquietment", 
            "Fear", "Pain", "Suffering"
        ]
        
        self.continuous_emotions = ["Valence", "Arousal", "Dominance"]

    def _load_tsn_rgb(self):
        model = TSN(
            logger=None,
            num_classes=26,
            num_dimensions=3,
            rgb_body=True,
            rgb_context=True,
            rgb_face=True,
            flow_body=False,
            flow_context=False,
            flow_face=False,
            scenes=True,
            attributes=True,
            depth=False,
            rgbdiff_body=False,
            rgbdiff_context=False,
            rgbdiff_face=False,
            arch='resnet50',
            consensus_type='avg',
            partial_bn=True,
            embed=True
        )
        
        checkpoint = torch.load(f"{self.model_dir}/tsn_rgb_bcfsa_embed_pbn.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def _load_tsn_flow(self):
        model = TSN(
            logger=None,
            num_classes=26,
            num_dimensions=3,
            rgb_body=False,
            rgb_context=False,
            rgb_face=False,
            flow_body=True,
            flow_context=True,
            flow_face=True,
            scenes=False,
            attributes=False,
            depth=False,
            rgbdiff_body=False,
            rgbdiff_context=False,
            rgbdiff_face=False,
            arch='resnet50',
            consensus_type='avg',
            partial_bn=True,
            embed=True
        )
        
        checkpoint = torch.load(f"{self.model_dir}/tsn_flow_bcf_embed_pbn.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def _load_stgcn(self):
        model = STGCN(
            in_channels=3,
            num_class=26,
            num_dim=3,
            layout='openpose',
            strategy='spatial',
            max_hop=1,
            dilation=1,
            edge_importance_weighting=True
        )
        
        checkpoint = torch.load(f"{self.model_dir}/stgcn_spatial_kinetics.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, processed_data):
        """Make predictions using all models"""
        with torch.no_grad():
            # Prepare inputs
            rgb_input = {
                'body': processed_data['rgb'].to(self.device),
                'face': processed_data['face'].to(self.device),
                'context': processed_data['rgb'].to(self.device)
            }
            
            flow_input = {
                'body': processed_data['flow'].to(self.device),
                'face': processed_data['flow'].to(self.device),
                'context': processed_data['flow'].to(self.device)
            }
            
            skeleton_input = {
                'skeleton': processed_data['joints'].to(self.device)
            }
            
            # Get predictions from each model
            rgb_out = self.tsn_rgb(rgb_input, num_segments=25)
            flow_out = self.tsn_flow(flow_input, num_segments=25)
            skeleton_out = self.stgcn(skeleton_input)
            
            # Combine predictions (weighted average)
            weights = [0.4, 0.3, 0.3]  # RGB, Flow, Skeleton weights
            
            cat_preds = []
            cont_preds = []
            
            for out in [rgb_out, flow_out, skeleton_out]:
                cat_preds.append(torch.sigmoid(out['categorical']))
                cont_preds.append(torch.sigmoid(out['continuous']))
            
            cat_preds = torch.stack(cat_preds)
            cont_preds = torch.stack(cont_preds)
            
            weighted_cat = (cat_preds * torch.tensor(weights).view(-1, 1, 1).to(self.device)).sum(0)
            weighted_cont = (cont_preds * torch.tensor(weights).view(-1, 1, 1).to(self.device)).sum(0)
            
            # Format results
            results = {
                'categorical': {
                    emotion: float(score)
                    for emotion, score in zip(self.categorical_emotions, weighted_cat[0])
                },
                'continuous': {
                    emotion: float(score)
                    for emotion, score in zip(self.continuous_emotions, weighted_cont[0])
                }
            }
            
            return results 