#!/usr/bin/env python3
"""
Test image generation utilities for CI workflows.
"""

from PIL import Image, ImageDraw
import os


def create_comprehensive_test_images(output_dir='regression_test_data'):
    """Create comprehensive test images for regression testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    configs = [
        {'name': 'simple_scene', 'color': 'lightblue', 'text': 'Simple Scene'},
        {'name': 'complex_scene', 'color': 'lightgreen', 'text': 'Complex Scene with Objects'},
        {'name': 'portrait', 'color': 'lightcoral', 'text': 'Portrait Style'},
        {'name': 'landscape', 'color': 'lightyellow', 'text': 'Landscape View'},
        {'name': 'abstract', 'color': 'lightpink', 'text': 'Abstract Pattern'},
        {'name': 'document', 'color': 'white', 'text': 'Document Text Sample'},
    ]
    
    for config in configs:
        # Create different sized images
        for size_name, (width, height) in [('small', (200, 150)), ('medium', (400, 300)), ('large', (800, 600))]:
            img = Image.new('RGB', (width, height), color=config['color'])
            draw = ImageDraw.Draw(img)
            
            # Add text
            text_size = max(12, width // 20)
            draw.text((10, 10), config['text'], fill='black')
            
            # Add geometric shapes
            if 'complex' in config['name']:
                draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], outline='red', width=2)
                draw.ellipse([width//6, height//6, width//3, height//3], outline='blue', width=2)
            
            if 'landscape' in config['name']:
                # Add horizon line
                draw.line([0, height//2, width, height//2], fill='darkgreen', width=3)
            
            filename = f'{output_dir}/{config["name"]}_{size_name}.jpg'
            img.save(filename)
            print(f'Created {filename}')


if __name__ == '__main__':
    create_comprehensive_test_images()