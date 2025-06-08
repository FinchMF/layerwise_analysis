"""
Control vector generation and application for transformer models.

This module provides functionality for generating and applying control vectors
to transformer model hidden states, enabling controlled text generation
and representation manipulation.
"""

import torch
from typing import Dict, List, Tuple

class ControlVectorGenerator:
    """Generates and manages control vectors for transformer models."""
    
    def __init__(self, model_manager):
        """Initialize with model manager for tokenization and inference."""
        self.model_manager = model_manager
        self.cached_controls = {}
        self.default_model = "bert-base-uncased"  # Default model for control generation
        self.current_hidden_size = None

    def update_model_config(self, hidden_size):
        """Update control vector configuration for new model."""
        if self.current_hidden_size != hidden_size:
            self.current_hidden_size = hidden_size
            self.cached_controls = {}  # Clear cache when model size changes

    def generate_sentiment_controls(self) -> Dict[str, torch.Tensor]:
        """Generate extreme sentiment control vectors using contrastive examples."""
        if self.model_manager.model is None:
            self.model_manager.load_model(self.default_model)

        # More extreme and diverse examples
        positive_examples = [
            "This is absolutely incredible! A masterpiece!", 
            "I am overjoyed and thrilled beyond words!",
            "This is the most amazing thing ever created!",
            "Absolutely perfect in every possible way!",
            "Pure brilliance and extraordinary excellence!",
            "This brings tears of joy to my eyes!",
            "A revolutionary breakthrough that changes everything!",
            "The pinnacle of human achievement!"
        ]
        
        negative_examples = [
            "This is horrifically awful and completely useless!",
            "I absolutely despise everything about this!",
            "The worst possible outcome imaginable!",
            "A complete disaster and utter failure!",
            "Absolutely devastating and terrible!",
            "This makes me feel physically ill!",
            "A catastrophic mistake that ruins everything!",
            "The epitome of complete worthlessness!"
        ]
        
        # Get base control vectors
        pos_states = self._get_averaged_states(positive_examples)
        neg_states = self._get_averaged_states(negative_examples)
        
        # Amplify the difference
        diff_vector = pos_states - neg_states
        amplification = 2.5  # Increase control strength
        
        # Normalize and scale the control vectors
        controls = {
            "positive": self._normalize_and_scale(diff_vector, amplification),
            "negative": self._normalize_and_scale(-diff_vector, amplification)
        }
        
        self.cached_controls["sentiment"] = controls
        return controls

    def _normalize_and_scale(self, vector: torch.Tensor, scale: float = 2.5) -> torch.Tensor:
        """Normalize and scale control vectors for stronger effect."""
        # Normalize per layer
        normalized = [
            layer / (layer.norm(dim=-1, keepdim=True) + 1e-5)
            for layer in vector
        ]
        # Stack and scale
        return torch.stack(normalized) * scale

    def _get_averaged_states(self, examples: List[str]) -> torch.Tensor:
        """Get averaged hidden states across examples."""
        states = []
        for text in examples:
            # Ensure model is loaded
            if self.model_manager.model is None:
                self.model_manager.load_model(self.model_manager.default_model)
                
            hidden_states, _, _, _ = self.model_manager.analyze_text(text)
            # Ensure states are on correct device and average across tokens
            layer_states = [h[0].to(self.model_manager.device).mean(0) for h in hidden_states]
            states.append(torch.stack(layer_states))
        
        return torch.stack(states).mean(0)  # Average across examples

    def apply_control(self, hidden_states: List[torch.Tensor], control_type: str,
                     direction: str, strength: float = 1.0) -> List[torch.Tensor]:
        """Apply control vector with progressive strengthening."""
        if control_type not in self.cached_controls:
            raise ValueError(f"Control type {control_type} not generated")
            
        control = self.cached_controls[control_type][direction]
        modified_states = []
        
        # Progressive strengthening through layers
        for i, (layer_state, control_state) in enumerate(zip(hidden_states, control)):
            layer_strength = strength * (1.0 + i * 0.1)  # Increase strength in deeper layers
            modified = layer_state + (control_state * layer_strength).unsqueeze(0)
            modified_states.append(modified)
            
        return modified_states
