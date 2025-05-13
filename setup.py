from setuptools import setup, find_packages

setup(
    name="vidapter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "moviepy",
        "SpeechRecognition",
        "pytubefix",
        "Pillow",
        "matplotlib",
        "llama-index",
        "llama-index-vector-stores-lancedb",
        "llama-index-multi-modal-llms-openai",
        "openai",
        "lancedb",
        "opencv-python",
    ],
    author="Arnav Gupta",
    author_email="ar9avg@gmail.com",
    description="A pipeline for converting RTSP and video files into RAG ready embeddings using vision and language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/aeyei/vidapter",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 