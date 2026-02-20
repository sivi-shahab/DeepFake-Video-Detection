# video_analyzer/utils.py
"""
ML Utilities for Video Authenticity Analysis
VideoClassificationModel: ResNeXt50 + LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE = {}


class VideoClassificationModel(nn.Module):
    """
    Video Classification Model: ResNeXt50 + LSTM
    """
    
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, 
                 hidden_dim=2048, bidirectional=False):
        super(VideoClassificationModel, self).__init__()
        
        # ResNeXt50 backbone
        resnext = models.resnext50_32x4d(pretrained=True)
        self.cnn = nn.Sequential(*list(resnext.children())[:-2])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            latent_dim, 
            hidden_dim, 
            lstm_layers, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        
        # Classifier
        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(linear_input_dim, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        
    def forward(self, x, return_features=False):
        batch_size, seq_length, c, h, w = x.shape
        
        # CNN features
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.cnn(x)
        x = self.avgpool(features)
        x = x.view(batch_size, seq_length, 2048)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x_pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        x = self.relu(x_pooled)
        if self.training:
            x = self.dropout(x)
        logits = self.classifier(x)
        
        if return_features:
            return features, logits
        return logits


class VideoPreprocessor:
    """
    Video preprocessing pipeline
    """
    
    def __init__(self, 
                 target_size=(112, 112),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        
        self.target_size = target_size
        self.mean = mean
        self.std = std

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def extract_frames(self, video_path, num_frames=60):
        """Extract frames uniformly from video"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, 
                                       min(num_frames, total_frames), dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            
            if len(frames) == 0:
                raise ValueError("No frames extracted")
            
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
        
        return frames

    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        try:
            return self.transform(frame)
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            # Return zero tensor with correct shape
            return torch.zeros(3, *self.target_size)

    def preprocess_video(self, video_path, num_frames=60, device=None):
        """Preprocess entire video"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract frames
        frames = self.extract_frames(video_path, num_frames)
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            tensor = self.preprocess_frame(frame)
            processed_frames.append(tensor)
        
        # Stack frames
        video_tensor = torch.stack(processed_frames).to(device)
        
        logger.info(f"Preprocessed video tensor shape: {video_tensor.shape}")
        
        return video_tensor


class VideoAnalyzer:
    """
    Main class for video analysis
    """
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocessor = VideoPreprocessor()
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load VideoClassificationModel from checkpoint"""
        try:
            logger.info(f"Loading model from: {model_path}")
            
            # Initialize model
            self.model = VideoClassificationModel(
                num_classes=2,
                latent_dim=2048,
                lstm_layers=1,
                hidden_dim=2048,
                bidirectional=False
            )
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract state_dict
            state_dict = None
            if isinstance(checkpoint, dict):
                for key in ['state_dict', 'model_state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                
                if state_dict is None:
                    # Direct state_dict
                    if any(k.startswith(('cnn.', 'lstm.', 'classifier.')) for k in checkpoint.keys()):
                        state_dict = checkpoint
            
            if state_dict is None:
                raise ValueError("Cannot find state_dict in checkpoint")
            
            # Clean keys (remove 'module.' prefix)
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                clean_state_dict[k] = v
            
            # Load weights
            self.model.load_state_dict(clean_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def predict_video(self, video_path, sequence_length=60):
        """
        Predict video authenticity
        
        Returns:
            {
                'prediction': 'REAL' or 'FAKE',
                'confidence': float (0-100),
                'probabilities': {'REAL': float, 'FAKE': float}
            }
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        logger.info(f"Predicting: {video_path}")
        
        # Preprocess video
        video_tensor = self.preprocessor.preprocess_video(
            video_path, 
            num_frames=sequence_length,
            device=self.device
        )
        
        # Add batch dimension
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            logits = self.model(video_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            result = {
                'prediction': 'REAL' if prediction.item() == 1 else 'FAKE',
                'confidence': round(confidence.item() * 100, 2),
                'probabilities': {
                    'REAL': round(probabilities[0, 1].item(), 4),
                    'FAKE': round(probabilities[0, 0].item(), 4),
                }
            }
        
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%)")
        
        return result

    def generate_heatmap(self, video_path, sequence_length=60):
        """
        Generate Grad-CAM heatmap
        
        Returns:
            (overlay_image, prediction_idx, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess
        video_tensor = self.preprocessor.preprocess_video(
            video_path, num_frames=sequence_length, device=self.device
        )
        
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        
        # Get features
        with torch.no_grad():
            features, logits = self.model(video_tensor, return_features=True)
            probabilities = F.softmax(logits, dim=1)
            _, prediction = torch.max(probabilities, dim=1)
        
        # Get classifier weights
        weights = self.model.classifier.weight[prediction.item()].detach().cpu().numpy()
        features_np = features[-1].detach().cpu().numpy()
        
        # Generate CAM
        cam = np.zeros(features_np.shape[1:], dtype=np.float32)
        for c in range(features_np.shape[0]):
            cam += weights[c] * features_np[c]
        
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (112, 112))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        
        # Get original frame
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
        else:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        return overlay, prediction.item(), probabilities[0, prediction.item()].item() * 100


def get_model_path(sequence_length=60):
    """Get model file path based on sequence length"""
    model_dir = getattr(settings, 'MODEL_PATH', 'models/')
    
    # Map sequence length to model file
    model_mapping = {
        40: 'model_95_acc_40_frames_FF_data.pt',
        80: 'model_97_acc_80_frames_FF_data.pt',
        100: 'model_97_acc_1000_frames_FF_data.pt'
    }
    
    model_filename = model_mapping.get(sequence_length, 'model_97_acc_1000_frames_FF_data.pt')
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return model_path


def get_or_load_model(sequence_length=60):
    """
    Get cached model or load new one
    
    Returns:
        (VideoAnalyzer, device)
    """
    cache_key = f'analyzer_{sequence_length}'
    
    # Return cached model if available
    if cache_key in _MODEL_CACHE:
        logger.info(f"Using cached model for seq={sequence_length}")
        return _MODEL_CACHE[cache_key]
    
    # Load new model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = get_model_path(sequence_length)
    
    logger.info(f"Loading VideoAnalyzer from: {model_path}")
    analyzer = VideoAnalyzer(model_path=model_path, device=device)
    
    # Cache the model
    _MODEL_CACHE[cache_key] = (analyzer, device)
    
    return analyzer, device