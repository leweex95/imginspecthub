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
                 model_id: str = "unsloth/llava-1.5-7b-hf"):
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
                from transformers import LlavaProcessor, LlavaForConditionalGeneration
                
                # Load processor and model
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
                print(f"Real LLaVA model loaded successfully on {self.device}")
                
            except Exception as model_error:
                print(f"Could not load LLaVA processor ({model_error}), trying LlavaNext...")
                
                # Try LlavaNext which might be required for this model
                try:
                    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
                    
                    self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
                    self.model = LlavaNextForConditionalGeneration.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    
                    if self.device != "cuda":
                        self.model = self.model.to(self.device)
                    
                    self.model.eval()
                    print(f"LLaVA-Next model loaded successfully on {self.device}")
                    
                except Exception as fallback_error:
                    print(f"Could not load LLaVA-Next either ({fallback_error})")
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
        
        # Real LLaVA implementation
        try:
            # Simple direct approach that should work with LLaVA 1.5
            if prompt:
                text_prompt = prompt
            else:
                text_prompt = "Describe this image in detail."
            
            # Process inputs - try the simplest approach first
            inputs = self.processor(images=pil_image, text=text_prompt, return_tensors="pt")
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,  # Use greedy decoding for more stable results
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response (skip the input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error in LLaVA generation: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Get image embedding using LLaVA.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Image embedding as numpy array
        """
        if not self._is_loaded:
            self.load_model()
        
        pil_image = self.process_image(image)
        
        # Real LLaVA implementation - extract vision features
        try:
            # Prepare inputs for vision encoder
            inputs = self.processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to the correct device
            inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Extract image features from vision tower
            with torch.no_grad():
                if hasattr(self.model, 'vision_tower'):
                    # LLaVA 1.5 style
                    image_features = self.model.vision_tower(inputs['pixel_values'])
                elif hasattr(self.model, 'multi_modal_projector'):
                    # LLaVA-NeXT style
                    vision_outputs = self.model.vision_model(**inputs)
                    image_features = vision_outputs.last_hidden_state
                else:
                    # Generic approach
                    outputs = self.model.encode_images(inputs['pixel_values'])
                    image_features = outputs
                
                # Pool the features (mean pooling over spatial dimensions)
                if len(image_features.shape) > 2:
                    # If features have spatial dimensions, pool them
                    embedding = image_features.mean(dim=1)  # Average over spatial dimensions
                else:
                    embedding = image_features
                
                # Convert to numpy and normalize
                embedding = embedding.cpu().numpy().squeeze()
                if len(embedding.shape) == 0:
                    embedding = embedding.reshape(1)
                
                # Normalize the embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
        except Exception as e:
            print(f"Error extracting LLaVA embeddings: {e}")
            raise RuntimeError(f"Failed to extract embeddings from LLaVA model: {e}")