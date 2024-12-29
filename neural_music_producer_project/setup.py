
from setuptools import setup, find_packages

setup(
    name="neural_music_producer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "moviepy>=1.0.3",
        "numpy>=1.24.0",
        "streamlit>=1.25.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "soundfile>=0.12.1",
        "python-dotenv>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI system that generates custom music for videos",
    keywords="ai, music generation, computer vision, deep learning",
    python_requires=">=3.8",
)
