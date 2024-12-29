import numpy as np
from typing import Dict, List
import pygame.midi
import time
from dataclasses import dataclass
from midiutil import MIDIFile
import pygame
import random

@dataclass
class MusicParams:
    tempo: int
    key: str
    scale: str
    intensity: float
    dominant_emotion: str

class MusicGenerator:
    def __init__(self):
        self.emotion_to_scale = {
            'happy': ['major', 'lydian'],
            'sad': ['minor', 'dorian'],
            'neutral': ['major', 'minor'],
            'surprise': ['lydian', 'mixolydian'],
            'fear': ['phrygian', 'locrian'],
            'disgust': ['locrian', 'phrygian']
        }
        
        self.emotion_to_tempo = {
            'happy': (120, 140),
            'sad': (60, 80),
            'neutral': (90, 110),
            'surprise': (130, 150),
            'fear': (100, 120),
            'disgust': (70, 90)
        }
        
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10]
        }

    def timeline_to_music_params(self, timeline: Dict[str, List[float]], segment_idx: int) -> MusicParams:
        # Get the dominant emotion for this segment
        emotions = {emotion: intensities[segment_idx] 
                   for emotion, intensities in timeline.items()}
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        intensity = emotions[dominant_emotion]
        
        # Select musical parameters based on emotion
        tempo_range = self.emotion_to_tempo[dominant_emotion]
        tempo = int(tempo_range[0] + (tempo_range[1] - tempo_range[0]) * intensity)
        
        # Select scale based on emotion
        possible_scales = self.emotion_to_scale[dominant_emotion]
        scale = random.choice(possible_scales)
        
        # Select key (C, C#, etc.) - could be made more sophisticated
        key = random.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B'])
        
        return MusicParams(
            tempo=tempo,
            key=key,
            scale=scale,
            intensity=intensity,
            dominant_emotion=dominant_emotion
        )

    def generate_midi(self, timeline: Dict[str, List[float]], output_file: str):
        # Create MIDI file with 1 track
        midifile = MIDIFile(1)
        track = 0
        time = 0
        midifile.addTrackName(track, time, "Generated Music")
        
        for segment_idx in range(len(list(timeline.values())[0])):
            params = self.timeline_to_music_params(timeline, segment_idx)
            
            # Add tempo change
            midifile.addTempo(track, time, params.tempo)
            
            # Generate notes based on scale and emotion
            scale_notes = self.scales[params.scale]
            base_note = 60  # Middle C
            
            # Generate a short musical phrase
            for _ in range(4):  # 4 beats per segment
                if random.random() < params.intensity:  # More notes when intensity is higher
                    note = base_note + random.choice(scale_notes)
                    duration = random.choice([0.5, 1.0])  # Note duration in beats
                    volume = int(60 + params.intensity * 40)  # Volume based on intensity
                    
                    midifile.addNote(track, 0, note, time, duration, volume)
                time += 0.5  # Advance time by half a beat
        
        # Write the MIDI file
        with open(output_file, 'wb') as f:
            midifile.writeFile(f)
            
    def play_midi(self, midi_file: str):
        from .music_player import MusicPlayer
        player = MusicPlayer()
        player.play_midi(midi_file, method='timidity')