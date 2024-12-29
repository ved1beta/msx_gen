import pygame
import subprocess
import os
from typing import Optional

class MusicPlayer:
    @staticmethod
    def play_midi(midi_file: str, method: str = 'timidity'):
        """
        Play a MIDI file using either pygame or timidity directly
        
        Args:
            midi_file (str): Path to the MIDI file
            method (str): 'pygame' or 'timidity'
        """
        if not os.path.exists(midi_file):
            raise FileNotFoundError(f"MIDI file not found: {midi_file}")

        if method == 'pygame':
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(midi_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"Pygame playback failed: {e}")
                print("Falling back to timidity...")
                MusicPlayer.play_midi(midi_file, method='timidity')
        
        elif method == 'timidity':
            try:
                # Convert MIDI to WAV first
                wav_file = midi_file.replace('.mid', '.wav')
                subprocess.run(['timidity', midi_file, '-Ow', '-o', wav_file], check=True)
                
                # Play the WAV file
                pygame.mixer.init()
                pygame.mixer.music.load(wav_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                # Clean up the WAV file
                pygame.mixer.quit()
                if os.path.exists(wav_file):
                    os.remove(wav_file)
                    
            except subprocess.CalledProcessError:
                print("Error: Could not convert MIDI file. Please make sure TiMidity is installed.")
                print("Install with: sudo apt-get install timidity")
            except Exception as e:
                print(f"Error playing music: {e}")