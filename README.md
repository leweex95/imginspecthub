# imginspecthub

[![Tests](https://github.com/leweex95/imginspecthub/workflows/Tests/badge.svg)](https://github.com/leweex95/imginspecthub/actions/workflows/test.yml)
[![Nightly Regression Tests](https://github.com/leweex95/imginspecthub/workflows/Nightly%20Regression%20Tests/badge.svg)](https://github.com/leweex95/imginspecthub/actions/workflows/nightly-regression.yml)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Poetry](https://img.shields.io/badge/poetry-2.2.0-blue.svg)](https://python-poetry.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A unified hub for running and comparing image understanding models (BLIP-2, MiniGPT-4, LLaVA, CLIP, etc.) through a simple CLI and API.

## Features

- **Unified Interface**: Single API for multiple image understanding models
- **CLI Tool**: Simple command-line interface with `imginspect` command
- **Multiple Operations**: Text descriptions, embeddings, and similarity scores
- **Batch Processing**: Process multiple images efficiently
- **Auto-download**: Automatically download and cache models
- **CPU Fallback**: Graceful fallback to CPU when GPU unavailable
- **Flexible Output**: JSON and CSV export formats
- **Extensible**: Easy to add new models
- **Cross-platform**: Works on Windows, macOS, and Linux

## Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/leweex95/imginspecthub.git
cd imginspecthub
poetry install
```

### Using pip

```bash
git clone https://github.com/leweex95/imginspecthub.git
cd imginspecthub
pip install -e .
```

## Quick Start

### CLI Usage

#### Basic image description
```bash
imginspect run --model blip2 --image photo.jpg --operation description
```

#### Visual question answering (recommended)
```bash
imginspect run --model blip2 --image photo.jpg --prompt "What does this image show?"
```

#### Get image embeddings
```bash
imginspect run --model clip --image photo.jpg --operation embedding
```

#### Compare image similarity
```bash
imginspect run --model clip --image photo1.jpg --image2 photo2.jpg --operation similarity
```

#### Batch processing
```bash
imginspect batch --model blip2 --directory ./images --operations description,embedding --output results.json
```

#### Compare multiple models
```bash
imginspect compare --models clip,blip2 --image photo.jpg --operations description --output comparison.json
```

### Python API Usage

```python
from imginspecthub import ImageInspector

# Initialize inspector with working model
inspector = ImageInspector(model_name="blip2", device="auto")

# Visual question answering (primary feature)
response = inspector.get_description("photo.jpg", "What does this image show?")
print(f"AI Response: {response}")

# Get basic description without prompt
description = inspector.get_description("photo.jpg")
print(f"Description: {description}")

# Get embedding for similarity search
embedding = inspector.get_embedding("photo.jpg")
print(f"Embedding shape: {embedding.shape}")

# Calculate similarity between images
similarity = inspector.get_similarity("photo1.jpg", "photo2.jpg")
print(f"Similarity: {similarity:.3f}")

# Batch processing
results = inspector.process_directory(
    directory="./images",
    operations=["description", "embedding"],
    output_file="results.json"
)
```

## Supported Models

| Model | Description | Embedding | Visual Q&A | Status |
|-------|-------------|-----------|-----------|--------|
| CLIP | Yes | Yes | Limited | Ready |
| BLIP-2 | Yes | Yes | Yes | **✅ Working** |
| LLaVA | Yes | Yes | Yes | ⚠️ **Compatibility Issues** |
| MiniGPT-4 | Yes | Yes | Coming Soon | Coming Soon |

### Model Status Details:
- **BLIP-2**: Fully functional for visual question answering and image description
- **LLaVA**: Model loading works but has input format compatibility issues (needs fixing)
- **CLIP**: Working with mock fallback (real model requires authentication)

## CLI Commands

### `imginspect run`
Process a single image with a specific model.

**Options:**
- `--model, -m`: Model to use (required)
- `--image, -i`: Path to input image (required)
- `--operation, -o`: Operation (description/embedding/similarity)
- `--prompt, -p`: Text prompt for guided generation
- `--device, -d`: Device to use (cuda/cpu/auto)
- `--output`: Output file for results
- `--format`: Output format (json/csv)

### `imginspect batch`
Process multiple images in a directory.

**Options:**
- `--model, -m`: Model to use (required)
- `--directory, -d`: Directory with images (required)
- `--operations, -o`: Comma-separated operations
- `--output`: Output file (required)
- `--workers, -w`: Number of parallel workers
- `--extensions`: File extensions to process

### `imginspect compare`
Compare multiple models on the same image.

**Options:**
- `--models, -m`: Comma-separated model names (required)
- `--image, -i`: Path to input image (required)
- `--operations, -o`: Operations to perform
- `--output`: Output file for comparison

### `imginspect models`
List all available models.

### `imginspect info`
Show system and device information.

## Visual Question Answering

### Real-World Example with Budapest Street Scene

The tool provides sophisticated visual question answering capabilities. Here's a real example using a Budapest street image:

```bash
# Basic image description
imginspect run --model blip2 --image budapest_street.png
# Output: "two pictures of a street with a bicycle and a bus stop"

# Specific question about the image
imginspect run --model blip2 --image budapest_street.png --prompt "What does this image show?"
# Output: "a street with a bicycle and a phone booth"

# Architecture-focused analysis
imginspect run --model blip2 --image budapest_street.png --prompt "Describe the architecture in this image"
# Output: Analysis of architectural elements visible in the scene
```

### Financial Chart Analysis
The tool supports sophisticated visual question answering for financial chart analysis:

```bash
# Analyze chart patterns
imginspect run --model blip2 --image chart.jpg --prompt "does this image show a chart pattern with increasing trend and a double bottom at the end of the movement?"

# Technical analysis
imginspect run --model blip2 --image chart.jpg --prompt "what type of chart pattern is shown in this image?"

# Market trend analysis
imginspect run --model blip2 --image chart.jpg --prompt "is there a bullish or bearish trend visible in this chart?"
```

### General Visual Q&A
```bash
# Object detection and description
imginspect run --model blip2 --image photo.jpg --prompt "what objects can you see in this image?"

# Scene understanding
imginspect run --model blip2 --image scene.jpg --prompt "describe the setting and atmosphere of this image"

# Specific questions
imginspect run --model blip2 --image document.jpg --prompt "what text is visible in this image?"
```

### ⚠️ Known Issues

**LLaVA Model**: Currently has compatibility issues with input format processing. The model loads successfully but encounters errors during inference. This will be fixed in a future update. Use BLIP-2 for reliable visual question answering.

## Configuration

### HuggingFace Authentication

For BLIP-2 and other advanced models, you need to authenticate with HuggingFace:

1. Create account at [https://huggingface.co](https://huggingface.co)
2. Generate token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login via CLI:
   ```bash
   huggingface-cli login
   # Enter your token when prompted
   ```

### Model Requirements

- **BLIP-2**: Requires HuggingFace authentication (~16GB download)
- **LLaVA**: Requires HuggingFace authentication (~14GB download, compatibility issues)
- **CLIP**: Works with mock fallback, authentication needed for real model

### Environment Variables

- `IMGINSPECTHUB_CACHE_DIR`: Directory for model cache (default: `~/.cache/imginspecthub`)
- `IMGINSPECTHUB_DEVICE`: Default device (cuda/cpu/auto)
- `IMGINSPECTHUB_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Device Selection

The tool automatically selects the best available device:
1. CUDA (if available)
2. CPU (fallback)

You can override with `--device` or `IMGINSPECTHUB_DEVICE`.

## Output Formats

### JSON Format
```json
[
  {
    "image_path": "photo.jpg",
    "model_name": "clip",
    "device": "cpu",
    "operations": {
      "description": {
        "result": "a photo of a cat",
        "execution_time": 1.23,
        "success": true
      }
    },
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### CSV Format
```csv
image_path,model_name,device,operations.description.result,operations.description.execution_time,operations.description.success,timestamp
photo.jpg,clip,cpu,"a photo of a cat",1.23,true,2024-01-01T12:00:00
```

## Adding New Models

To add a new model, create a class that inherits from `BaseModel`:

```python
from imginspecthub.models.base import BaseModel
from imginspecthub.models.registry import register_model

@register_model("my_model")
class MyModel(BaseModel):
    def load_model(self):
        # Load your model here
        pass
    
    def get_description(self, image, prompt=None):
        # Generate description
        pass
    
    def get_embedding(self, image):
        # Generate embedding
        pass
```

## Development

### Setup Development Environment

```bash
git clone https://github.com/leweex95/imginspecthub.git
cd imginspecthub
poetry install --with dev
```

### Run Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black imginspecthub/
poetry run isort imginspecthub/
```

### Type Checking

```bash
poetry run mypy imginspecthub/
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `poetry run pytest`
5. Submit a pull request

## License

This project is licensed under the GPL-3.0-or-later License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use imginspecthub in your research, please cite:

```bibtex
@software{imginspecthub,
  title={imginspecthub: A Unified Hub for Image Understanding Models},
  author={imginspecthub contributors},
  year={2024},
  url={https://github.com/leweex95/imginspecthub}
}
```
