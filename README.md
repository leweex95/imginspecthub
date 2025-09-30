# imginspecthub

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Visual question answering for images using AI models. Ask questions about any image and get intelligent responses.

## Available Models

- **LLaVA**: Detailed conversational analysis, best for complex scenes
- **BLIP-2**: Fast and concise descriptions, good for simple questions

## CLI Examples

```bash
# Detailed scene analysis with LLaVA
imginspect run --model llava --image street.jpg --prompt "Describe the architecture"

# Quick object identification with BLIP-2  
imginspect run --model blip2 --image street.jpg --prompt "What vehicles are visible?"

# Process multiple images in batch
imginspect batch --model llava --directory ./images --output results.json
```

## Python API

```python
from imginspecthub import ImageInspector

# Ask questions about images
inspector = ImageInspector("llava")
response = inspector.get_description("photo.jpg", "What's happening in this image?")
print(response)
```

## Setup

### HuggingFace Authentication (Required)

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login: `huggingface-cli login`

## Quick Start

```bash
# Ask questions about images
imginspect run --model llava --image photo.jpg --prompt "What's in this image?"
imginspect run --model blip2 --image photo.jpg --prompt "Describe this scene"

# Process multiple images
imginspect batch --model llava --directory ./images --output results.json
```
