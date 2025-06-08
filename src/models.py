"""
Model management module for transformer analysis.

This module handles loading and managing different transformer models,
including BERT, GPT-2, and TinyLlama. It provides unified interfaces for
model initialization, text analysis, and prediction generation.

Classes:
    ModelManager: Main class for handling transformer model operations.
"""

import os
import torch
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig
from .controls import ControlVectorGenerator

class ModelManager:
    """Model management and inference handler.
    
    Handles model loading, tokenization, and inference for different transformer
    architectures. Supports masked language models (BERT) and causal language
    models (GPT-2, TinyLlama).
    
    Attributes:
        model: The loaded transformer model.
        tokenizer: Associated tokenizer for the model.
        model_type (str): Type of loaded model ('bert', 'gpt2', or 'tinyllama').
    """
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        # Initialize core attributes to ensure they always exist
        instance.model = None
        instance.tokenizer = None
        instance.model_type = None
        instance.model_name = None
        instance._needs_cleanup = False
        return instance

    def __init__(self):
        """Initialize model manager state."""
        # Set environment variable for tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Control setup - only initialize if not already set
        if not hasattr(self, 'control_generator'):
            self.control_generator = ControlVectorGenerator(self)
        if not hasattr(self, 'current_controls'):
            self.current_controls = None

    def load_model(self, model_name):
        """Load a transformer model and its tokenizer."""
        try:
            # Cleanup old model if it exists
            if hasattr(self, 'model'):
                if self.model is not None:
                    self.model.cpu()
                    self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.model_type = 'bert' if 'bert' in model_name else 'gpt2' if 'gpt2' in model_name else 'tinyllama'
            
            # Initialize tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.model_type == 'gpt2':
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Common kwargs for model loading
            model_kwargs = {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "device_map": None,  # Force CPU initialization
                "local_files_only": False
            }

            # Force CPU initialization for all models
            torch.cuda.empty_cache()  # Clear GPU memory
            with torch.device('cpu'):
                if self.model_type == 'tinyllama':
                    config = AutoConfig.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        use_cache=False
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=config,
                        trust_remote_code=True,
                        attn_implementation="eager",
                        **model_kwargs
                    )
                elif self.model_type == 'bert':
                    self.model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
                else:  # gpt2
                    self.model = GPT2LMHeadModel.from_pretrained(model_name, **model_kwargs)

            # Initialize model parameters before moving to device
            with torch.no_grad():
                dummy_input = self.tokenizer("test", return_tensors="pt")
                _ = self.model(**dummy_input)

            # Move to target device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store model's hidden size for control vectors
            self.hidden_size = self.model.config.hidden_size
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None  # Ensure model is None on error
            raise

    def _verify_model_state(self):
        """Verify model is properly initialized and on correct device."""
        if self.model is None:
            raise RuntimeError("Model failed to load")
            
        # Check for meta tensors
        for name, param in self.model.named_parameters():
            if param.is_meta:
                raise RuntimeError(f"Parameter {name} is still a meta tensor")
            if param.device != torch.device(self.device):
                param.data = param.data.to(self.device)

    def set_control(self, control_type: str, direction: str, strength: float = 1.0):
        """Set active control vector configuration."""
        self.current_controls = (control_type, direction, strength)

    def clear_control(self):
        """Clear active control vector."""
        self.current_controls = None

    def analyze_text(self, text):
        """Analyze text with optional control vector application.

        Args:
            text (str): The input text to be analyzed.

        Returns:
            tuple: Contains hidden states, attentions, tokens, and predictions.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Move inputs to same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        with torch.no_grad():
            # Ensure inputs are on correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            hidden_states = list(outputs.hidden_states)
            
            # Apply control vector if active
            if self.current_controls is not None:
                control_type, direction, strength = self.current_controls
                hidden_states = self.control_generator.apply_control(
                    hidden_states, control_type, direction, strength
                )
                
            predictions = self.get_model_prediction(text, hidden_states=hidden_states)
            
        return hidden_states, outputs.attentions, tokens, predictions

    def get_model_prediction(self, text, hidden_states=None):
        """Generate model predictions for the given text."""
        if self.model_type == 'bert':
            if '[MASK]' not in text:
                text += ' [MASK]'
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
                
                if len(mask_token_index) == 0:
                    return {
                        "input": text,
                        "predictions": ["No mask token found"],
                        "type": "mask_prediction"
                    }
                
                mask_token_logits = outputs.logits[0, mask_token_index[0], :]
                probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)
                top_5 = torch.topk(probs, 5)
                
                predictions = [
                    f"{self.tokenizer.decode([token_id.item()]).strip()} ({prob.item()*100:.1f}%)"
                    for token_id, prob in zip(top_5.indices, top_5.values)
                ]
                
            return {
                "input": text,
                "predictions": predictions,
                "type": "mask_prediction"
            }
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Get multiple completions with temperature
                generated_sequences = []
                for temp in [0.7, 1.0]:  # Try different temperatures
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=3,  # Generate a few tokens for context
                        do_sample=True,
                        temperature=temp,
                        top_k=5,
                        num_return_sequences=3,
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )
                    generated_sequences.extend(outputs)

                # Format all completions
                completions = []
                for seq in generated_sequences:
                    # Get the new tokens only
                    new_tokens = seq[inputs['input_ids'].shape[1]:]
                    completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    if completion.strip():  # Only add non-empty completions
                        completions.append(completion.strip())

                # Remove duplicates and get unique completions
                unique_completions = list(dict.fromkeys(completions))[:5]
                
                # Format the display
                return {
                    "input": text,
                    "next_token": f"Possible continuations:",
                    "top_k_completions": [f"'{c}'" for c in unique_completions],
                    "full_text": f"{text} {unique_completions[0]}",
                    "type": "generation"
                }
