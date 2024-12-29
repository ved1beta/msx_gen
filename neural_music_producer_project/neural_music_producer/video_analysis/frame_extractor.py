
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

def extract_frames(video_path, fps=5):
    """Extract frames from video at specified fps"""
    video = VideoFileClip(video_path)
    frames = []
    for t in np.arange(0, video.duration, 1/fps):
        frame = video.get_frame(t)
        frames.append(frame)
    return np.array(frames)
