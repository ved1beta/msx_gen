import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_generator import MusicGenerator
from video_analysis.emotion_analyzer import EmotionAnalyzer
from video_analysis.frame_extractor import FrameExtractor

def test_music_generation():
    # Test with a sample timeline
    sample_timeline = {
        'happy': [0.1, 0.2, 0.3],
        'sad': [0.2, 0.1, 0.1],
        'neutral': [0.4, 0.4, 0.3],
        'surprise': [0.1, 0.2, 0.2],
        'fear': [0.1, 0.0, 0.0],
        'disgust': [0.1, 0.1, 0.1]
    }
    
    print("Testing music generation...")
    generator = MusicGenerator()
    
    output_file = "test_output.mid"
    generator.generate_midi(sample_timeline, output_file)
    print(f"Generated MIDI file: {output_file}")
    
    print("\nPlaying generated music...")
    generator.play_midi(output_file)

if __name__ == "__main__":
    test_music_generation()