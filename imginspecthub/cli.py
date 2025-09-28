"""
Command-line interface for imginspecthub.
"""

import os
import sys
import click
from typing import Optional, List
from pathlib import Path

from . import __version__
from .core import ImageInspector
from .models import get_available_models
from .utils.device import get_device_info
from .utils.batch_processing import find_images


@click.group()
@click.version_option(version=__version__, prog_name="imginspect")
def cli():
    """imginspect: A unified hub for image understanding models."""
    pass


@cli.command()
@click.option("--model", "-m", required=True, 
              help="Model to use (e.g., clip, blip2, llava, minigpt4)")
@click.option("--image", "-i", required=True, type=click.Path(exists=True),
              help="Path to input image")
@click.option("--operation", "-o", default="description",
              type=click.Choice(["description", "embedding", "similarity"]),
              help="Operation to perform")
@click.option("--prompt", "-p", help="Text prompt for guided generation")
@click.option("--device", "-d", help="Device to use (cuda, cpu, auto)")
@click.option("--output", help="Output file for results")
@click.option("--format", "output_format", default="json",
              type=click.Choice(["json", "csv"]),
              help="Output format")
@click.option("--log/--no-log", default=False, help="Enable logging")
@click.option("--image2", type=click.Path(exists=True),
              help="Second image for similarity comparison")
def run(model: str, image: str, operation: str, prompt: Optional[str],
        device: Optional[str], output: Optional[str], output_format: str,
        log: bool, image2: Optional[str]):
    """Run image understanding model on a single image."""
    
    # Validate model
    available_models = get_available_models()
    if model not in available_models:
        click.echo(f"Error: Model '{model}' not available.", err=True)
        click.echo(f"Available models: {', '.join(available_models)}", err=True)
        sys.exit(1)
    
    # Validate similarity operation
    if operation == "similarity" and not image2:
        click.echo("Error: --image2 required for similarity operation", err=True)
        sys.exit(1)
    
    try:
        # Initialize inspector
        inspector = ImageInspector(
            model_name=model,
            device=device,
            log_results_to_file=log,
            log_file=output if log and output else None
        )
        
        click.echo(f"Using model: {model}")
        click.echo(f"Device: {inspector.device}")
        click.echo(f"Processing: {image}")
        
        # Perform operation
        if operation == "description":
            result = inspector.get_description(image, prompt)
            click.echo(f"\nDescription: {result}")
            
        elif operation == "embedding":
            embedding = inspector.get_embedding(image)
            click.echo(f"\nEmbedding shape: {embedding.shape}")
            click.echo(f"Embedding (first 10 values): {embedding.flatten()[:10].tolist()}")
            result = embedding.tolist()
            
        elif operation == "similarity":
            similarity = inspector.get_similarity(image, image2)
            click.echo(f"\nSimilarity score: {similarity:.4f}")
            result = similarity
        
        # Save output if specified
        if output and not log:
            from .utils.logging import log_results, create_result_entry
            
            result_entry = create_result_entry(
                model_name=model,
                image_path=image,
                operation=operation,
                result=result,
                execution_time=0.0,  # Not tracked in simple mode
                device=inspector.device,
                additional_info={
                    "prompt": prompt,
                    "image2": image2 if image2 else None
                }
            )
            
            log_results(result_entry, output, output_format)
            click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", required=True,
              help="Model to use")
@click.option("--directory", "-d", required=True, type=click.Path(exists=True),
              help="Directory containing images")
@click.option("--operations", "-o", default="description",
              help="Comma-separated operations (description,embedding)")
@click.option("--output", required=True, help="Output file for results")
@click.option("--format", "output_format", default="json",
              type=click.Choice(["json", "csv"]),
              help="Output format")
@click.option("--prompt", "-p", help="Text prompt for descriptions")
@click.option("--device", help="Device to use (cuda, cpu, auto)")
@click.option("--workers", "-w", default=1, type=int,
              help="Number of worker threads")
@click.option("--extensions", help="Comma-separated file extensions (default: jpg,png,etc)")
def batch(model: str, directory: str, operations: str, output: str,
          output_format: str, prompt: Optional[str], device: Optional[str],
          workers: int, extensions: Optional[str]):
    """Process multiple images in batch."""
    
    # Parse operations
    operation_list = [op.strip() for op in operations.split(',')]
    valid_operations = ["description", "embedding"]
    
    for op in operation_list:
        if op not in valid_operations:
            click.echo(f"Error: Invalid operation '{op}'. Valid: {valid_operations}", err=True)
            sys.exit(1)
    
    # Parse extensions
    ext_list = None
    if extensions:
        ext_list = [f".{ext.strip().lstrip('.')}" for ext in extensions.split(',')]
    
    try:
        # Initialize inspector
        inspector = ImageInspector(model_name=model, device=device)
        
        click.echo(f"Using model: {model}")
        click.echo(f"Device: {inspector.device}")
        click.echo(f"Processing directory: {directory}")
        click.echo(f"Operations: {operation_list}")
        click.echo(f"Workers: {workers}")
        
        # Process directory
        results = inspector.process_directory(
            directory=directory,
            operations=operation_list,
            output_file=output,
            format_type=output_format,
            prompt=prompt,
            max_workers=workers,
            extensions=ext_list,
            show_progress=True
        )
        
        click.echo(f"\nProcessed {len(results)} images")
        click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--models", "-m", required=True,
              help="Comma-separated list of models to compare")
@click.option("--image", "-i", required=True, type=click.Path(exists=True),
              help="Path to input image")
@click.option("--operations", "-o", default="description",
              help="Comma-separated operations to perform")
@click.option("--output", help="Output file for comparison results")
@click.option("--format", "output_format", default="json",
              type=click.Choice(["json", "csv"]),
              help="Output format")
@click.option("--prompt", "-p", help="Text prompt for descriptions")
@click.option("--device", help="Device to use")
def compare(models: str, image: str, operations: str, output: Optional[str],
            output_format: str, prompt: Optional[str], device: Optional[str]):
    """Compare multiple models on a single image."""
    
    # Parse models and operations
    model_list = [m.strip() for m in models.split(',')]
    operation_list = [op.strip() for op in operations.split(',')]
    
    # Validate models
    available_models = get_available_models()
    for model in model_list:
        if model not in available_models:
            click.echo(f"Error: Model '{model}' not available.", err=True)
            click.echo(f"Available models: {', '.join(available_models)}", err=True)
            sys.exit(1)
    
    try:
        # Initialize models
        inspectors = []
        for model_name in model_list:
            inspector = ImageInspector(model_name=model_name, device=device)
            inspectors.append(inspector)
        
        click.echo(f"Comparing models: {model_list}")
        click.echo(f"Image: {image}")
        click.echo(f"Operations: {operation_list}")
        
        # Compare models
        from .utils.batch_processing import compare_models_on_image
        
        models_instances = [insp.model for insp in inspectors]
        comparison = compare_models_on_image(models_instances, image, operation_list, prompt)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("COMPARISON RESULTS")
        click.echo("="*60)
        
        for model_name, model_results in comparison["models"].items():
            click.echo(f"\nModel: {model_name}")
            click.echo("-" * 40)
            
            for operation, op_result in model_results.get("operations", {}).items():
                if op_result.get("success", False):
                    result = op_result["result"]
                    time_taken = op_result["execution_time"]
                    
                    if operation == "description":
                        click.echo(f"Description: {result}")
                    elif operation == "embedding":
                        click.echo(f"Embedding shape: {len(result) if isinstance(result, list) else 'N/A'}")
                    
                    click.echo(f"Time: {time_taken:.3f}s")
                else:
                    click.echo(f"{operation}: FAILED - {op_result.get('error', 'Unknown error')}")
        
        # Save comparison results
        if output:
            from .utils.logging import log_results
            log_results(comparison, output, output_format)
            click.echo(f"\nComparison results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def models():
    """List available models."""
    available_models = get_available_models()
    
    click.echo("Available models:")
    for model in available_models:
        click.echo(f"  - {model}")
    
    if not available_models:
        click.echo("  No models available. Please check your installation.")


@cli.command()
def info():
    """Show system and device information."""
    device_info = get_device_info()
    
    click.echo("System Information:")
    click.echo(f"  Python version: {sys.version}")
    click.echo(f"  imginspecthub version: {__version__}")
    
    click.echo("\nDevice Information:")
    click.echo(f"  Recommended device: {device_info['recommended_device']}")
    click.echo(f"  CUDA available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        click.echo(f"  CUDA devices: {device_info['cuda_device_count']}")
        for i, device_name in enumerate(device_info.get('cuda_devices', [])):
            click.echo(f"    {i}: {device_name}")
    
    click.echo(f"  MPS (Apple Silicon) available: {device_info['mps_available']}")
    
    click.echo(f"\nAvailable models: {', '.join(get_available_models())}")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()