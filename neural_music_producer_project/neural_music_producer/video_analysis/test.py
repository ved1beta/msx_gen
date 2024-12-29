import matplotlib.pyplot as plt
from pathlib import Path
import sys
import traceback
from frame_extractor import FrameExtractor
from emotion_analyzer import EmotionAnalyzer

def test_emotion_analyzer(video_path: str):
    try:
        print("Initializing components...")
        frame_extractor = FrameExtractor(target_fps=5, target_size=(224, 224))
        emotion_analyzer = EmotionAnalyzer()
        
        print("\n1. Extracting frames...")
        frames = frame_extractor.extract_frames(video_path)
        print(f"Frame array shape: {frames.shape}")
        print(f"Frame value range: [{frames.min():.3f}, {frames.max():.3f}]")
        
        print("\n2. Analyzing emotions...")
        predictions = emotion_analyzer.analyze_video_frames(frames)
        print(f"Generated {len(predictions)} predictions")
        
        if not predictions:
            print("No predictions generated!")
            return None, None
        
        # Print sample prediction
        print("\n3. Sample prediction for frame 0:")
        first_pred = predictions[0]
        print(f"Dominant emotion: {first_pred.dominant_emotion}")
        print(f"Confidence: {first_pred.intensity:.2f}")
        print("All emotions:")
        for emotion, prob in sorted(first_pred.emotions.items()):
            print(f"  {emotion}: {prob:.3f}")
        
        # Plot emotional timeline
        print("\n4. Generating emotional timeline plot...")
        timeline = emotion_analyzer.get_emotional_timeline(predictions)
        
        plt.figure(figsize=(15, 6))
        colors = {
            'happy': 'green',
            'sad': 'blue',
            'neutral': 'gray',
            'surprise': 'yellow',
            'fear': 'purple',
            'disgust': 'red'
        }
        
        for emotion, intensities in timeline.items():
            plt.plot(intensities, 
                    label=emotion, 
                    color=colors.get(emotion, 'black'),
                    linewidth=2, 
                    alpha=0.7)
        
        plt.title("Video Emotional Timeline")
        plt.xlabel("Frame Number")
        plt.ylabel("Emotion Intensity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = Path("emotion_timeline.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"\nPlot saved to {output_path}")
        
        return predictions, timeline
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "stock_vids/new.mp4"
    
    print(f"Processing video: {video_path}")
    predictions, timeline = test_emotion_analyzer(video_path)