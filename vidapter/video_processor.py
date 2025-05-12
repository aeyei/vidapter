import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm

from moviepy.editor import VideoFileClip
from pytubefix import YouTube
from PIL import Image
import speech_recognition as sr

@dataclass
class VideoConfig:
    """Configuration for video processing."""
    output_video_path: str = "./video_data/"
    output_folder: str = "./mixed_data/"
    frame_rate: float = 0.5  # frames per second
    audio_format: str = "wav"
    image_format: str = "png"
    image_prefix: str = "frame"
    audio_filename: str = "output_audio"
    text_filename: str = "output_text.txt"

class VideoProcessor:
    def __init__(self, config: Optional[VideoConfig] = None):
        """Initialize the video processor with configuration."""
        self.config = config or VideoConfig()
        self.logger = logging.getLogger(__name__)
        
        # Create necessary directories
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_video_path).mkdir(parents=True, exist_ok=True)
        
        self.output_audio_path = os.path.join(
            self.config.output_folder,
            f"{self.config.audio_filename}.{self.config.audio_format}"
        )
        self.filepath = os.path.join(self.config.output_video_path, "input_vid.mp4")

    def download_video(self, url: str) -> Dict[str, Any]:
        """Download video from YouTube and return metadata."""
        try:
            self.logger.info(f"Downloading video from URL: {url}")
            yt = YouTube(url)
            
            # Get video metadata
            metadata = {
                "Author": yt.author,
                "Title": yt.title,
                "Views": yt.views,
                "Length": yt.length,
                "Description": yt.description,
                "PublishDate": str(yt.publish_date),
                "Keywords": yt.keywords
            }
            
            # Download with progress bar
            stream = yt.streams.get_highest_resolution()
            stream.download(
                output_path=self.config.output_video_path,
                filename="input_vid.mp4"
            )
            
            self.logger.info("Video download complete")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            raise

    def video_to_images(self) -> None:
        """Extract frames from video and save as images."""
        try:
            self.logger.info("Extracting frames from video...")
            clip = VideoFileClip(self.filepath)
            
            # Calculate total frames for progress bar
            total_frames = int(clip.duration * self.config.frame_rate)
            
            # Create progress bar
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                def save_frame(t):
                    frame = clip.get_frame(t)
                    frame_path = os.path.join(
                        self.config.output_folder,
                        f"{self.config.image_prefix}{int(t * self.config.frame_rate):04d}.{self.config.image_format}"
                    )
                    Image.fromarray(frame).save(frame_path)
                    pbar.update(1)
                
                # Extract frames at specified intervals
                for t in range(int(clip.duration * self.config.frame_rate)):
                    save_frame(t / self.config.frame_rate)
            
            clip.close()
            self.logger.info("Frame extraction complete")
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
            raise

    def video_to_audio(self) -> None:
        """Extract audio from video."""
        try:
            self.logger.info("Extracting audio from video...")
            clip = VideoFileClip(self.filepath)
            audio = clip.audio
            
            # Extract audio with progress bar
            audio.write_audiofile(
                self.output_audio_path,
                logger=None  # Disable moviepy's logger to use our own
            )
            
            clip.close()
            self.logger.info("Audio extraction complete")
            
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            raise

    def audio_to_text(self) -> str:
        """Convert audio to text using speech recognition."""
        try:
            self.logger.info("Converting audio to text...")
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(self.output_audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_whisper(audio_data)
                    self.logger.info("Audio transcription complete")
                    return text
                except sr.UnknownValueError:
                    self.logger.warning("Audio not recognized")
                    return "Audio not recognized"
                except sr.RequestError as e:
                    self.logger.error(f"Error with speech recognition service: {str(e)}")
                    return f"Error: {str(e)}"
                    
        except Exception as e:
            self.logger.error(f"Error converting audio to text: {str(e)}")
            raise

    def process_video(self, url: str) -> Tuple[Dict[str, Any], str]:
        """Process video through the complete pipeline."""
        try:
            self.logger.info("Starting video processing pipeline...")
            
            # Download video and get metadata
            metadata = self.download_video(url)
            
            # Extract frames and audio
            self.video_to_images()
            self.video_to_audio()
            
            # Convert audio to text
            text_data = self.audio_to_text()

            # Save extracted text
            text_path = os.path.join(self.config.output_folder, self.config.text_filename)
            with open(text_path, "w") as file:
                file.write(text_data)

            # Clean up audio file
            if os.path.exists(self.output_audio_path):
                os.remove(self.output_audio_path)
                self.logger.info("Cleaned up temporary audio file")

            self.logger.info("Video processing pipeline complete")
            return metadata, text_data
            
        except Exception as e:
            self.logger.error(f"Error in video processing pipeline: {str(e)}")
            raise 