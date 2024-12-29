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
    def preprocessframes(self, frame:np.ndarray ) -> np.ndarray :
        frame = cv2.resize(frame, self.target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)/255.0

        return frame 
    
    def extract_frames(self, 
                       video_path:str,
                       save_path:Optional[str]= None)-> np.ndarray:
        video_path= str(Path(video_path).resolve())
        metadata = self.get_video_matadata(video_path)

        sample_interval = int(metadata.fps / self.target_fps)

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        batch = []

        self.logger.info(f"Extracting frames from {video_path}")
        self.logger.info(f"Video FPS: {metadata.fps}, Target FPS: {self.target_fps}")
        
        while True:
            ret , frame= cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0 :
                batch.append(frame)

                if len(batch)==self.batch_size:
                    processed_batcj= self.process_batch(batch)
                    batch= []

            frame_count += 1
        if batch:
            processed_batch = self._process_batch(batch)
            frames.extend(processed_batch)
        cap.release()

        frames_array = np.array(frames)
        
        if save_path:
            self._save_frames(frames_array, save_path)
            
        self.logger.info(f"Extracted {len(frames)} frames")
        return frames_array


    def _process_batch(self, batch: List[np.ndarray]) -> List[np.ndarray]:
        """Process a batch of frames in parallel."""
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(self.preprocessframes, batch))
        return processed_frames
    
    def _save_frames(self, frames: np.ndarray, save_path: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame = (frame * 255).astype(np.uint8)
            cv2.imwrite(str(save_path / f"frame_{i:04d}.jpg"), frame)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = FrameExtractor(
        target_fps=5,
        target_size=(224, 224),
        batch_size=32
    )
    
    video_path = "stock_vids/new.mp4"
    frames = extractor.extract_frames(
        video_path=video_path,
        save_path="extracted_frames"
    )
    
    print(f"Extracted frames shape: {frames.shape}")        
# extractor.get_video_matadata("stock_vids/new.mp4")