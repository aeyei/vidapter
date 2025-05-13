"""
Vidapter - A pipeline for converting RTSP and video files into RAG ready embeddings
"""

from .video_processor import VideoProcessor
from .rag_engine import RAGEngine
from .rtsp_streamer import RTSPStreamer

__version__ = "0.1.0"
__all__ = ['VideoProcessor', 'RAGEngine', 'RTSPStreamer'] 