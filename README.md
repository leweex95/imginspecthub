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
imginspect run --model clip --image photo.jpg --operation description
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
imginspect batch --model clip --directory ./images --operations description,embedding --output results.json
```

#### Compare multiple models
```bash
imginspect compare --models clip,blip2 --image photo.jpg --operations description --output comparison.json
```

### Python API Usage

```python
from imginspecthub import ImageInspector

# Initialize inspector
inspector = ImageInspector(model_name="clip", device="auto")

# Get description
description = inspector.get_description("photo.jpg")
print(f"Description: {description}")

# Get embedding
embedding = inspector.get_embedding("photo.jpg")
print(f"Embedding shape: {embedding.shape}")

# Calculate similarity
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

| Model | Description | Embedding | Status |
|-------|-------------|-----------|--------|
| CLIP | Yes | Yes | Ready |
| BLIP-2 | Yes | Yes | Coming Soon |
| LLaVA | Yes | Yes | Coming Soon |
| MiniGPT-4 | Yes | Yes | Coming Soon |

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

## Configuration

### Environment Variables

- `IMGINSPECTHUB_CACHE_DIR`: Directory for model cache (default: `~/.cache/imginspecthub`)
- `IMGINSPECTHUB_DEVICE`: Default device (cuda/cpu/auto)
- `IMGINSPECTHUB_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Device Selection

The tool automatically selects the best available device:
1. CUDA (if available)
2. MPS (Apple Silicon, if available)
3. CPU (fallback)

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
