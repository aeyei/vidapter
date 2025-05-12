import os
import sys
import logging
import argparse
from typing import Optional
from pathlib import Path

from .video_processor import VideoProcessor, VideoConfig
from .rag_engine import RAGEngine, RAGConfig, get_openai_api_key

def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video RAG: Process videos and perform RAG-based Q&A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Base directory for all output files"
    )
    
    # Video processing configuration
    video_group = parser.add_argument_group("Video Processing")
    video_group.add_argument(
        "--frame-rate",
        type=float,
        default=0.5,
        help="Frame extraction rate (frames per second)"
    )
    video_group.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Format for extracted frames"
    )
    
    # RAG configuration
    rag_group = parser.add_argument_group("RAG Configuration")
    rag_group.add_argument(
        "--similarity-top-k",
        type=int,
        default=3,
        help="Number of similar text chunks to retrieve"
    )
    rag_group.add_argument(
        "--image-similarity-top-k",
        type=int,
        default=3,
        help="Number of similar images to retrieve"
    )
    rag_group.add_argument(
        "--model",
        default="gpt-4-turbo",
        help="OpenAI model to use for generation"
    )
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a video and prepare it for RAG"
    )
    process_parser.add_argument(
        "url",
        help="YouTube URL of the video to process"
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query the processed video using RAG"
    )
    query_parser.add_argument(
        "query",
        help="Question to ask about the video"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Create base output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure paths
        video_dir = output_dir / "video_data"
        mixed_dir = output_dir / "mixed_data"
        
        # Create configurations
        video_config = VideoConfig(
            output_video_path=str(video_dir),
            output_folder=str(mixed_dir),
            frame_rate=args.frame_rate,
            image_format=args.image_format
        )
        
        rag_config = RAGConfig(
            output_folder=str(mixed_dir),
            similarity_top_k=args.similarity_top_k,
            image_similarity_top_k=args.image_similarity_top_k,
            model_name=args.model
        )
        
        if args.command == "process":
            logger.info("Starting video processing...")
            processor = VideoProcessor(config=video_config)
            metadata, text = processor.process_video(args.url)
            logger.info("Video processing complete")
            logger.info(f"Video metadata: {metadata}")
            logger.info(f"Transcribed text length: {len(text)} characters")
            
        elif args.command == "query":
            try:
                api_key = get_openai_api_key(args.api_key)
            except ValueError as e:
                logger.error(str(e))
                sys.exit(1)
                
            logger.info("Initializing RAG engine...")
            rag_engine = RAGEngine(config=rag_config, api_key=api_key)
            
            logger.info("Setting up index...")
            rag_engine.setup_index()
            
            logger.info(f"Processing query: {args.query}")
            retrieved_images, retrieved_text = rag_engine.retrieve(args.query)
            
            logger.info("Generating response...")
            response = rag_engine.generate_response(
                query_str=args.query,
                context_str="\n".join(retrieved_text),
                metadata={},  # TODO: Load metadata from file
                image_paths=retrieved_images
            )
            
            print("\nResponse:")
            print(response)
            
        else:
            logger.error("No command specified. Use 'process' or 'query'.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 