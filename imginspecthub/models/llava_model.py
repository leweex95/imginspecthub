"""
LLaVA model implementation for image understanding.
"""

from typing import Optional, Union
import torch
import numpy as np
from PIL import Image

from .base import BaseModel
from .registry import register_model


@register_model("llava")
class LLaVAModel(BaseModel):
    """LLaVA (Large Language and Vision Assistant) model."""
    
    def __init__(self, model_name: str = "llava", device: Optional[str] = None,
                 model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA model.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on
            model_id: HuggingFace model identifier
        """
        super().__init__(model_name, device)
        self.model_id = model_id
    
    def load_model(self) -> None:
        """Load the LLaVA model and processor."""
        try:
            print(f"Loading LLaVA model {self.model_id} on {self.device}...")
            
            # Try to load real LLaVA model first
            try:
                from transformers import AutoProcessor, LlavaForConditionalGeneration
                
                # Load processor and model using AutoProcessor (recommended)
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                
                self.model.eval()
                print(f"Real LLaVA model loaded successfully on {self.device}")
                
            except Exception as model_error:
                print(f"Could not load LLaVA processor ({model_error}), trying fallback...")
                
                # Try alternative model
                try:
                    from transformers import LlavaProcessor, LlavaForConditionalGeneration
                    self.model_id = "unsloth/llava-1.5-7b-hf"  # Fallback to unsloth if official fails
                    
                    self.processor = LlavaProcessor.from_pretrained(self.model_id)
                    self.model = LlavaForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    
                    if self.device != "cuda":
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    print(f"LLaVA fallback model loaded successfully on {self.device}")
                    
                except Exception as fallback_error:
                    print(f"Could not load LLaVA fallback either ({fallback_error})")
                    raise RuntimeError(f"Failed to load any LLaVA model variant. Original error: {model_error}. Fallback error: {fallback_error}")
            
            self._is_loaded = True
            
        except Exception as e:
            print(f"Error loading LLaVA model: {e}")
            # Try CPU fallback only for device issues, not model loading issues
            if self.device != "cpu" and "device" in str(e).lower():
                print("Attempting CPU fallback...")
                self.device = "cpu"
                self._load_model()  # Recursively try on CPU
            else:
                raise RuntimeError(f"Failed to load LLaVA model: {e}")
    
    def get_description(self, image: Union[str, Image.Image], 
                       prompt: Optional[str] = None) -> str:
        """
        Get text description of the image using LLaVA.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt for guided generation
            
        Returns:
            Generated description
        """
        if not self._is_loaded:
            self.load_model()
        
        pil_image = self.process_image(image)
        
        # Real LLaVA implementation using official best practices
        try:
            # Use the official conversation format from HuggingFace docs
            if prompt:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ]
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {"type": "image"},
                        ]
                    }
                ]
            
            # Apply the chat template (official method)
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process inputs with the formatted prompt
            inputs = self.processor(images=pil_image, text=prompt_text, return_tensors="pt")
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response with optimal parameters
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased for more detailed responses
                    do_sample=False,     # Greedy decoding for consistent results
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode only the generated tokens (skip input)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error in LLaVA generation: {e}")
            return f"Error generating response: {str(e)}"