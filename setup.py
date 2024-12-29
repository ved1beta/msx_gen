# setup.py
from setuptools import setup, find_packages

setup(
    name="msx_gen",
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
    author="ved1beta",
    author_email="ved.work2024@gmail.com",
    description="An AI system that generates custom music for videos",
    keywords="ai, music generation, computer vision, deep learning",
    python_requires=">=3.8",
)

# Project structure will be:
# neural_music_producer/
# ├── __init__.py
# ├── video_analysis/
# │   ├── __init__.py
# │   ├── emotion_analyzer.py
# │   └── frame_extractor.py
# ├── music_generation/
# │   ├── __init__.py
# │   ├── generator.py
# │   └── audio_synthesis.py
# ├── utils/
# │   ├── __init__.py
# │   └── preprocessing.py
# └── ui/
#     ├── __init__.py
#     └── streamlit_app.py