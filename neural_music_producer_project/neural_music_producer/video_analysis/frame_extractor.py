import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

class VideoExtractor:
    def __init__(self, fps: float, frame_count: int, duration: float, width: int, height: int):
        self.fps = fps
        self.frame_count = frame_count
        self.duration = duration
        self.width = width
        self.height = height

    def __str__(self):
        return f"""
        Video Metadata:
        FPS: {self.fps}
        Frame Count: {self.frame_count}
        Duration: {self.duration} seconds
        Width: {self.width} pixels
        Height: {self.height} pixels
        """


class FrameExtractor:
    def __init__(self, 
                 target_fps : float =5, 
                 target_size : int = 32,
                 batch_size : Tuple[int, int] =(224,224)
                 ):
        self.target_fps = target_fps
        self.batch_size = batch_size
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)

    def get_video_matadata(self,video_path : str)-> VideoExtractor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"could not find video at path {video_path}")
        
        metadata = VideoExtractor(
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        cap.release()
        print(metadata)
        return metadata
    def preprocess_frames(self, frame:np.ndarray ) -> np.ndarray :
        frame = cv2.resize(frame, self.traget_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)/255.0
        return frame 

# extractor= FrameExtractor()
# extractor.get_video_matadata("stock_vids/new.mp4")