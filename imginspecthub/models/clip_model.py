"""
CLIP model implementation for image understanding.
"""

from typing import Optional, Union
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

from .base import BaseModel
from .registry import register_model


@register_model("clip")
class CLIPModel(BaseModel):
    """CLIP model for image-text understanding."""
    
    def __init__(self, model_name: str = "clip", device: Optional[str] = None,
                 model_id: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model.
        
        Args:
            model_name: Name of the model
            device: Device to run the model on
            model_id: HuggingFace model identifier
        """
        super().__init__(model_name, device)
        self.model_id = model_id
    
    def load_model(self) -> None:
        """Load the CLIP model and processor."""
        try:
            print(f"Loading CLIP model {self.model_id} on {self.device}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = HFCLIPModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            print(f"CLIP model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            # Try CPU fallback
            if self.device != "cpu":
                print("Attempting CPU fallback...")
                self.device = "cpu"
                self.model = HFCLIPModel.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()
                self._is_loaded = True
                print("CLIP model loaded successfully on CPU (fallback)")
            else:
                raise
    
    def get_description(self, image: Union[str, Image.Image], 
                       prompt: Optional[str] = None) -> str:
        """
        Get text description of the image using CLIP.
        
        Note: CLIP doesn't generate descriptions directly, so we use 
        predefined prompts and select the best match.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional custom prompts (comma-separated)
            
        Returns:
            Best matching description
        """
        if not self._is_loaded:
            self.load_model()
        
        pil_image = self.process_image(image)
        
        # Default prompts if none provided
        if prompt is None:
            default_prompts = [
                "a photo of a person",
                "a photo of an animal", 
                "a photo of a building",
                "a photo of a vehicle",
                "a photo of food",
                "a photo of nature",
                "a photo of an object",
                "a landscape photo",
                "an indoor photo",
                "an outdoor photo",
                "a close-up photo",
                "a wide-angle photo"
            ]
            texts = default_prompts
        else:
            texts = [p.strip() for p in prompt.split(',')]
        
        # Process inputs
        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get best match
        best_idx = probs.argmax().item()
        confidence = probs[0][best_idx].item()
        
        return f"{texts[best_idx]} (confidence: {confidence:.3f})"
    
    def get_embedding(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Get image embedding using CLIP.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            Image embedding as numpy array
        """
        if not self._is_loaded:
            self.load_model()
        
        pil_image = self.process_image(image)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().squeeze()
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get text embedding using CLIP.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding as numpy array
        """
        if not self._is_loaded:
            self.load_model()
        
        # Process text
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get text embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize the embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().squeeze()
    
    def get_image_text_similarity(self, image: Union[str, Image.Image], text: str) -> float:
        """
        Get similarity between image and text.
        
        Args:
            image: Image path or PIL Image
            text: Text to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        image_emb = self.get_embedding(image)
        text_emb = self.get_text_embedding(text)
        
        # Cosine similarity (already normalized)
        similarity = np.dot(image_emb, text_emb)
        
        return float(similarity)