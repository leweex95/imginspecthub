"""
BLIP-2 model implementation for image understanding.
"""

from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from .base import BaseModel
from .registry import register_model


@register_model("blip2")
class BLIP2Model(BaseModel):
    """BLIP-2 model for image captioning and understanding."""
    
    def __init__(self, model_name: str = "blip2", device: Optional[str] = None,
                 model_id: str = "Salesforce/blip2-flan-t5-xl"):
        """
        Initialize BLIP-2 model.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on
            model_id: HuggingFace model identifier
        """
        super().__init__(model_name, device)
        self.model_id = model_id
    
    def load_model(self) -> None:
        """Load the BLIP-2 model and processor."""
        try:
            print(f"Loading BLIP-2 model {self.model_id} on {self.device}...")
            
            # Try to load real BLIP-2 model first
            try:
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                
                # Load processor and model
                self.processor = Blip2Processor.from_pretrained(self.model_id)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                print(f"Real BLIP-2 model loaded successfully on {self.device}")
                
            except Exception as model_error:
                print(f"Could not load real BLIP-2 model ({model_error}), trying smaller variant...")
                
                # Try smaller BLIP-2 model
                try:
                    from transformers import Blip2Processor, Blip2ForConditionalGeneration
                    self.model_id = "Salesforce/blip2-opt-2.7b-coco"
                    
                    self.processor = Blip2Processor.from_pretrained(self.model_id)
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    print(f"BLIP-2 smaller variant loaded successfully on {self.device}")
                    
                except Exception as fallback_error:
                    print(f"Could not load any BLIP-2 variant ({fallback_error})")
                    raise RuntimeError(f"Failed to load any BLIP-2 model variant. Original error: {model_error}. Fallback error: {fallback_error}")
            
            self._is_loaded = True
            
        except Exception as e:
            print(f"Error loading BLIP-2 model: {e}")
            # Try CPU fallback only for device issues, not model loading issues
            if self.device != "cpu" and "device" in str(e).lower():
                print("Attempting CPU fallback...")
                self.device = "cpu"
                self._load_model()  # Recursively try on CPU
            else:
                raise RuntimeError(f"Failed to load BLIP-2 model: {e}")
    
    def get_description(self, image: Union[str, Image.Image], 
                       prompt: Optional[str] = None) -> str:
        """
        Get text description of the image using BLIP-2.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt for guided generation
            
        Returns:
            Generated description
        """
        if not self._is_loaded:
            self.load_model()
        
        pil_image = self.process_image(image)
        
        # Real BLIP-2 implementation
        try:
            # Prepare input text for conditional generation
            if prompt:
                # For question answering, use the prompt as conditioning
                inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
            else:
                # For image captioning, use no text input
                inputs = self.processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                if prompt:
                    # For prompted generation (question answering)
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=3,
                        early_stopping=True
                    )
                else:
                    # For image captioning
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        num_beams=5,
                        early_stopping=True
                    )
            
            # Decode the response
            generated_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Clean up the generated text
            if prompt and prompt.lower() in generated_text.lower():
                # Remove the input prompt from the output if it's repeated
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            return response if response else "Unable to generate description"
            
        except Exception as e:
            print(f"Error in BLIP-2 generation: {e}")
            return f"Error generating response: {str(e)}"