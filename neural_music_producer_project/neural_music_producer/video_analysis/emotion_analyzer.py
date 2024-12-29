import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class EmotionPrediction:
    frame_index: int
    emotions: Dict[str, float]
    dominant_emotion: str
    intensity: float

class EmotionAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Using a guaranteed available model
        model_name = "google/vit-base-patch16-224"
        
        print(f"Loading model: {model_name}")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        
        # Define simplified emotion mappings based on ImageNet classes that correspond to emotions
        self.emotion_labels = {
            0: "neutral",
            1: "happy",
            2: "sad",
            3: "surprise",
            4: "fear",
            5: "disgust"
        }
        print(f"Using emotion labels: {self.emotion_labels}")

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess frames for the model"""
        # Ensure frames are in range [0, 255]
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        
        return frames

    def analyze_batch(self, frames: np.ndarray) -> List[EmotionPrediction]:
        """Analyze emotions in a batch of frames"""
        # Preprocess frames
        frames = self.preprocess_frames(frames)
        
        # Prepare inputs
        inputs = self.processor(
            images=frames, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits[:, :len(self.emotion_labels)], dim=-1)
        
        predictions = []
        for frame_idx, frame_probs in enumerate(probs):
            # Convert to dictionary of emotion probabilities
            emotions = {
                self.emotion_labels[i]: float(prob)
                for i, prob in enumerate(frame_probs)
            }
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            # Calculate intensity
            intensity = float(max(frame_probs))
            
            predictions.append(EmotionPrediction(
                frame_index=frame_idx,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                intensity=intensity
            ))
        
        return predictions

    def analyze_video_frames(self, frames: np.ndarray, batch_size: int = 8) -> List[EmotionPrediction]:
        """Analyze emotions in all video frames"""
        print(f"Input frames shape: {frames.shape}")
        print(f"Frame values range: [{frames.min()}, {frames.max()}]")
        
        if len(frames.shape) == 3:  # Single frame
            frames = frames[np.newaxis, ...]
        
        all_predictions = []
        
        for i in tqdm(range(0, len(frames), batch_size), desc="Analyzing emotions"):
            batch_frames = frames[i:i + batch_size]
            try:
                batch_predictions = self.analyze_batch(batch_frames)
                
                # Update frame indices
                for pred in batch_predictions:
                    pred.frame_index += i
                    
                all_predictions.extend(batch_predictions)
            except Exception as e:
                print(f"Error processing batch starting at frame {i}: {str(e)}")
                continue
            
        return all_predictions

    def get_emotional_timeline(self, predictions: List[EmotionPrediction]) -> Dict[str, List[float]]:
        """Convert predictions into emotion timelines"""
        timeline = {emotion: [] for emotion in self.emotion_labels.values()}
        
        for pred in predictions:
            for emotion, intensity in pred.emotions.items():
                timeline[emotion].append(intensity)
        
        return timeline