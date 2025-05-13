import cv2
import time
from datetime import datetime
from typing import Optional, Generator, Dict, Any, List
import threading
from queue import Queue
import logging
import os
from pathlib import Path
import numpy as np
from .rag_engine import RAGEngine, RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTSPStreamer:
    def __init__(self, rtsp_url: str, frame_interval: float = 1.0, output_dir: str = "./rtsp_frames"):
        """
        Initialize RTSP streamer.
        
        Args:
            rtsp_url (str): URL of the RTSP stream
            frame_interval (float): Interval in seconds between captured frames
            output_dir (str): Directory to save processed frames
        """
        self.rtsp_url = rtsp_url
        self.frame_interval = frame_interval
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=100)
        self._thread = None
        self.output_dir = output_dir
        self.rag_engine = None
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def setup_rag_engine(self, config: Optional[RAGConfig] = None, api_key: Optional[str] = None):
        """Set up the RAG engine for processing frames."""
        self.rag_engine = RAGEngine(config=config, api_key=api_key)
        self.rag_engine.setup_index()

    def _save_frame(self, frame: np.ndarray, timestamp: str) -> str:
        """Save frame to disk and return the file path."""
        filename = f"frame_{timestamp.replace(':', '-')}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def process_frame(self, frame_data: Dict[str, Any]) -> Optional[str]:
        """
        Process a frame by saving it and computing embeddings.
        
        Args:
            frame_data (Dict[str, Any]): Frame data containing timestamp and frame array
            
        Returns:
            Optional[str]: Path to the saved frame if successful, None otherwise
        """
        if not self.rag_engine:
            logger.warning("RAG engine not set up. Call setup_rag_engine() first.")
            return None

        try:
            # Save frame to disk
            filepath = self._save_frame(frame_data['frame'], frame_data['timestamp'])
            
            # Create a document for the frame
            from llama_index.core.schema import ImageNode
            image_node = ImageNode(
                image=frame_data['frame'],
                metadata={
                    "timestamp": frame_data['timestamp'],
                    "source": "rtsp_stream",
                    "file_path": filepath
                }
            )
            
            # Add to the index
            self.rag_engine.retriever_engine.index.insert_nodes([image_node])
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None

    def start_processing(self):
        """Start processing frames in a separate thread."""
        if not self.rag_engine:
            raise ValueError("RAG engine not set up. Call setup_rag_engine() first.")

        def process_frames():
            while self.is_running:
                frame_data = self.get_frame(timeout=1.0)
                if frame_data:
                    self.process_frame(frame_data)

        self.processing_thread = threading.Thread(target=process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def start(self):
        """Start the RTSP stream capture in a separate thread."""
        if self.is_running:
            logger.warning("Stream is already running")
            return

        self.is_running = True
        self._thread = threading.Thread(target=self._capture_frames)
        self._thread.daemon = True
        self._thread.start()
        
        if self.rag_engine:
            self.start_processing()

    def stop(self):
        """Stop the RTSP stream capture."""
        self.is_running = False
        if self._thread:
            self._thread.join()
        if self.cap:
            self.cap.release()

    def _capture_frames(self):
        """Internal method to capture frames from RTSP stream."""
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                raise Exception(f"Failed to open RTSP stream: {self.rtsp_url}")

            last_capture_time = 0
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from RTSP stream")
                    time.sleep(1)  # Wait before retrying
                    continue

                current_time = time.time()
                if current_time - last_capture_time >= self.frame_interval:
                    frame_data = {
                        'timestamp': datetime.now().isoformat(),
                        'frame': frame
                    }
                    try:
                        self.frame_queue.put(frame_data, block=False)
                    except Queue.Full:
                        # If queue is full, remove oldest frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame_data, block=False)
                        except:
                            pass
                    last_capture_time = current_time

        except Exception as e:
            logger.error(f"Error in RTSP stream capture: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

    def get_frame(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next frame from the queue.
        
        Args:
            timeout (float, optional): Timeout in seconds to wait for a frame
            
        Returns:
            Optional[Dict[str, Any]]: Frame data containing timestamp and frame array
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 