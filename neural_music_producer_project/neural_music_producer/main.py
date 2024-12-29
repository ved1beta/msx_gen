import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_analysis.frame_extractor import FrameExtractor
from video_analysis.emotion_analyzer import EmotionAnalyzer
from music_generation.generator import MusicGenerator

def main(video_path: str):
    print("1. Initializing components...")
    frame_extractor = FrameExtractor(target_fps=5, target_size=(224, 224))
    emotion_analyzer = EmotionAnalyzer()
    music_generator = MusicGenerator()
    
    print("\n2. Extracting frames...")
    frames = frame_extractor.extract_frames(video_path)
    print(f"Frame array shape: {frames.shape}")
    
    print("\n3. Analyzing emotions...")
    predictions = emotion_analyzer.analyze_video_frames(frames)
    print(f"Generated {len(predictions)} predictions")
    
    print("\n4. Getting emotional timeline...")
    timeline = emotion_analyzer.get_emotional_timeline(predictions)
    
    print("\n5. Generating music...")
    output_file = "output_music.mid"
    music_generator.generate_midi(timeline, output_file)
    print(f"Generated MIDI file: {output_file}")
    
    print("\n6. Playing generated music...")
    music_generator.play_midi(output_file)

if __name__ == "__main__":
    video_path = "video_analysis/stock_vids/new.mp4"  # Update this path as needed
    main(video_path)