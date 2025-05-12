# Vidapter

A powerful tool for processing videos and performing RAG-based question answering using LlamaIndex.

## Features

- Download and process YouTube videos
- Extract frames and audio from videos
- Transcribe audio to text
- Perform RAG-based question answering using GPT-4 Vision
- Support for both text and image retrieval
- Progress tracking and detailed logging

## Installation

```bash
pip install vidapter
```

## Environment Variables

The following environment variables can be set:

- `OPENAI_API_KEY`: Your OpenAI API key (required for querying)
- `VIDAPTER_OUTPUT_DIR`: Default output directory (defaults to `./output`)

## Usage

### Process a Video

```bash
# Basic usage
vidapter process "https://youtube.com/watch?v=..."

# With custom output directory
vidapter process "https://youtube.com/watch?v=..." --output-dir ./my_video

# With custom frame rate and image format
vidapter process "https://youtube.com/watch?v=..." --frame-rate 1.0 --image-format jpg
```

### Query a Processed Video

```bash
# Basic usage
vidapter query "What is happening in the video?"

# With custom output directory
vidapter query "What is happening in the video?" --output-dir ./my_video

# With custom RAG parameters
vidapter query "What is happening in the video?" --similarity-top-k 5 --image-similarity-top-k 3
```

### Command Line Options

#### Common Options

- `--verbose`, `-v`: Enable verbose logging
- `--api-key`: OpenAI API key (can also be set via OPENAI_API_KEY environment variable)
- `--output-dir`: Base directory for all output files (default: `./output`)

#### Video Processing Options

- `--frame-rate`: Frame extraction rate in frames per second (default: 0.5)
- `--image-format`: Format for extracted frames (choices: png, jpg, jpeg, default: png)

#### RAG Configuration Options

- `--similarity-top-k`: Number of similar text chunks to retrieve (default: 3)
- `--image-similarity-top-k`: Number of similar images to retrieve (default: 3)
- `--model`: OpenAI model to use for generation (default: gpt-4-turbo)

## Output Structure

```
output/
├── video_data/          # Downloaded video files
└── mixed_data/          # Processed data
    ├── frame*.png       # Extracted video frames
    └── output_text.txt  # Transcribed text
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vidapter.git
cd vidapter
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
