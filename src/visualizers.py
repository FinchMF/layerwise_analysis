"""
Visualization module for transformer model analysis.

This module provides visualization tools for analyzing transformer model internals,
including attention patterns, layer metrics, and neural activations. It supports
various visualization types including heatmaps, clustering plots, and animations.

Classes:
    AttentionVisualizer: Main class for creating visualizations of transformer internals.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm, LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy
from matplotlib.backends.backend_pdf import PdfPages
import os
import torch

class AttentionVisualizer:
    """Visualization toolkit for transformer model attention analysis.
    
    This class provides methods for visualizing various aspects of transformer models,
    including attention patterns, layer metrics, and head behaviors. It supports
    both static and animated visualizations with configurable color schemes.
    
    Attributes:
        ASSETS_DIR (str): Directory path for saving visualization artifacts.
    """

    ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    
    def __init__(self):
        """Initialize the visualizer and create assets directory."""
        os.makedirs(self.ASSETS_DIR, exist_ok=True)

    @staticmethod
    def cluster_attention_heads(attn_tensor, layer):
        """Cluster attention heads using PCA and K-means.

        Args:
            attn_tensor (torch.Tensor): Attention tensors from model.
            layer (int): Layer index to analyze.

        Returns:
            matplotlib.figure.Figure: Scatter plot of clustered attention heads.
        """
        try:
            # Handle tensor conversion safely
            if isinstance(attn_tensor[layer], (list, tuple)):
                att = attn_tensor[layer][0]
            else:
                att = attn_tensor[layer]
            
            if isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy().astype(np.float32)
            else:
                att = np.array(att, dtype=np.float32)
            
            heads = att.shape[0]
            flat_heads = att.reshape(heads, -1)
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(flat_heads)
            kmeans = KMeans(n_clusters=3, n_init="auto", random_state=0)
            labels = kmeans.fit_predict(reduced)
            
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")
            ax.set_title(f"Attention Head Clustering (Layer {layer})")
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in cluster_attention_heads: {e}")
            return None

    @staticmethod
    def enhance_contrast(values, power=2):
        """Enhance contrast of values using power normalization.

        Args:
            values (np.ndarray): Input values to enhance.
            power (float, optional): Power factor for normalization. Defaults to 2.

        Returns:
            np.ndarray: Contrast-enhanced values.
        """
        # Normalize to [0, 1] range
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        # Apply power transformation for contrast enhancement
        enhanced = np.power(norm_values, 1/power)
        return enhanced

    @staticmethod
    def create_tonal_colormap(base_color='purple'):
        """Create perceptually uniform tonal colormap.

        Args:
            base_color (str, optional): Base color for the colormap. 
                Options: 'purple', 'blue', 'warm'. Defaults to 'purple'.

        Returns:
            matplotlib.colors.LinearSegmentedColormap: Generated colormap.
        """
        if base_color == 'purple':
            # Dark to bright
            return LinearSegmentedColormap.from_list('tonal_purple', 
                ['#2B0066', '#6600CC', '#9933FF', '#BF80FF', '#D9B3FF', '#F2E6FF'])
        elif base_color == 'blue':
            return LinearSegmentedColormap.from_list('tonal_blue',
                ['#003366', '#0066CC', '#3399FF', '#80C1FF', '#B3D9FF', '#E6F3FF'])
        else:  # default warm
            return LinearSegmentedColormap.from_list('tonal_warm',
                ['#CC7A00', '#FF9900', '#FFB84D', '#FFCC80', '#FFE0B3', '#FFF5E6'])

    @staticmethod
    def plot_attention_entropy_heatmap(attn_tensor, layer):
        """Create heatmap of attention entropy values.

        Args:
            attn_tensor (torch.Tensor): Attention tensors from model.
            layer (int): Layer index to analyze.

        Returns:
            matplotlib.figure.Figure: Heatmap of attention entropies.
        """
        try:
            # Handle tensor conversion safely
            if isinstance(attn_tensor[layer], (list, tuple)):
                att = attn_tensor[layer][0]
            else:
                att = attn_tensor[layer]
            
            if isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy().astype(np.float32)
            else:
                att = np.array(att, dtype=np.float32)
            
            entropies = np.array([[scipy_entropy(row + 1e-10) for row in head] for head in att])
            # Normalize without enhancement to preserve true relationships
            normalized_entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())
            
            fig, ax = plt.subplots()
            im = ax.imshow(normalized_entropies, aspect='auto', 
                          cmap=AttentionVisualizer.create_tonal_colormap('purple'))
            ax.set_title(f"Attention Entropy Heatmap (Layer {layer})")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_attention_entropy_heatmap: {e}")
            return None

    @staticmethod
    def visualize_token_attribution(attn_tensor, tokens, layer):
        """Visualize token attribution based on attention weights.

        Args:
            attn_tensor (torch.Tensor): Attention tensors from model.
            tokens (list): List of token strings.
            layer (int): Layer index to analyze.

        Returns:
            matplotlib.figure.Figure: Heatmap of token attribution.
        """
        try:
            # Handle tensor conversion safely
            if isinstance(attn_tensor[layer], (list, tuple)):
                att = attn_tensor[layer][0]
            else:
                att = attn_tensor[layer]
            
            if isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy().astype(np.float32)
            else:
                att = np.array(att, dtype=np.float32)
            
            avg_attn = att.mean(axis=0)
            # Simple min-max normalization
            normalized_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min())
            
            fig, ax = plt.subplots()
            im = ax.imshow(normalized_attn, 
                          cmap=AttentionVisualizer.create_tonal_colormap('blue'))
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens)
            ax.set_title(f"Token Attribution (Layer {layer})")
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in visualize_token_attribution: {e}")
            return None

    @staticmethod
    def plot_layer_metrics(df):
        """Create subplots of layer-wise metrics.

        Args:
            df (pd.DataFrame): DataFrame containing layer metrics.

        Returns:
            matplotlib.figure.Figure: Grid of metric plots.
        """
        metrics = ['effective_rank', 'participation_ratio', 'intrinsic_dim', 
                  'activation_entropy', 'cosine_sim_to_input']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.plot(df['layer'], df[metric], marker='o')
            ax.set_title(f'Layer-wise {metric.replace("_", " ").title()}')
            ax.set_xlabel('Layer')
            ax.set_ylabel(metric)
            ax.grid(True)
        
        # Remove extra subplot
        if len(axes) > len(metrics):
            fig.delaxes(axes[-1])
            
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_metric_heatmap(df):
        """Create heatmap of normalized metrics across layers.

        Args:
            df (pd.DataFrame): DataFrame containing layer metrics.

        Returns:
            matplotlib.figure.Figure: Heatmap visualization.
        """
        metrics = ['effective_rank', 'participation_ratio', 'intrinsic_dim', 
                  'activation_entropy', 'cosine_sim_to_input']
        
        normalized_data = df[metrics].copy()
        for metric in metrics:
            values = df[metric].values
            normalized_data[metric] = (values - values.min()) / (values.max() - values.min())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(normalized_data.T, aspect='auto',
                      cmap=AttentionVisualizer.create_tonal_colormap('warm'))
        
        # Set labels
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['layer'])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Normalized Metrics Across Layers')
        ax.set_xlabel('Layer')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_metric_correlations(df):
        """Create correlation matrix visualization for metrics.

        Args:
            df (pd.DataFrame): DataFrame containing layer metrics.

        Returns:
            matplotlib.figure.Figure: Correlation matrix plot.
        """
        metrics = ['effective_rank', 'participation_ratio', 'intrinsic_dim', 
                  'activation_entropy', 'cosine_sim_to_input']
        
        correlations = df[metrics].corr()
        # Enhance correlation values
        enhanced_corr = np.sign(correlations) * np.power(np.abs(correlations), 0.7)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(enhanced_corr, 
                      cmap=plt.cm.PuOr,  # Purple-Orange diverging colormap
                      norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        
        # Add correlation values
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                text = f'{correlations.iloc[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center')
        
        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Metric Correlations')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def analyze_attention_head_attribution(attn_tensor, tokens, layer):
        """Visualize attention patterns for individual heads."""
        try:
            # Handle tensor conversion safely
            if isinstance(attn_tensor[layer], (list, tuple)):
                att = attn_tensor[layer][0]
            else:
                att = attn_tensor[layer]
            
            if isinstance(att, torch.Tensor):
                att = att.detach().cpu().numpy().astype(np.float32)
            else:
                att = np.array(att, dtype=np.float32)
            
            fig, axes = plt.subplots(nrows=1, ncols=min(4, att.shape[0]), figsize=(16, 4))
            if att.shape[0] == 1:
                axes = [axes]  # Ensure axes is always iterable
            
            for i, ax in enumerate(axes):
                if i >= att.shape[0]:
                    break
                    
                head_attn = np.ascontiguousarray(att[i])
                # Ensure proper normalization
                head_attn = (head_attn - head_attn.min()) / (head_attn.max() - head_attn.min() + 1e-8)
                
                im = ax.imshow(head_attn, 
                             cmap='coolwarm',
                             aspect='auto')
                ax.set_title(f"Head {i}")
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticklabels(tokens)
                
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error in attention visualization: {e}")
            return None

    def create_attention_animation(self, attentions, tokens):
        """Create animated visualization of attention patterns across layers.

        Args:
            attentions: Attention tensors from model.
            tokens (list): List of token strings.

        Returns:
            str: Path to saved animation file.
        """
        try:
            if not attentions or not tokens:
                print("No attention data or tokens available")
                return None
                
            # Convert attentions to numpy and ensure proper dtype
            attention_data = []
            for layer_attn in attentions:
                if isinstance(layer_attn, torch.Tensor):
                    # Convert to numpy and ensure float dtype
                    layer_np = layer_attn.detach().cpu().numpy().astype(np.float32)
                elif isinstance(layer_attn, (list, tuple)) and len(layer_attn) > 0:
                    # Handle list/tuple of tensors (batch dimension)
                    if isinstance(layer_attn[0], torch.Tensor):
                        layer_np = layer_attn[0].detach().cpu().numpy().astype(np.float32)
                    else:
                        layer_np = np.array(layer_attn[0], dtype=np.float32)
                elif isinstance(layer_attn, np.ndarray):
                    # Ensure float dtype
                    layer_np = layer_attn.astype(np.float32)
                else:
                    # Handle other types by converting to float array
                    layer_np = np.array(layer_attn, dtype=np.float32)
                
                attention_data.append(layer_np)
            
            if not attention_data:
                print("No valid attention data found")
                return None
                
            # Get dimensions from first layer
            first_layer = attention_data[0]
            max_layers = len(attention_data)
            
            # Handle different attention tensor shapes
            if first_layer.ndim == 4:  # [batch, heads, seq, seq]
                seq_len = first_layer.shape[-1]
                # Average over batch and heads for animation
                processed_attentions = [np.mean(layer_attn[0], axis=0) for layer_attn in attention_data]
            elif first_layer.ndim == 3:  # [heads, seq, seq]
                seq_len = first_layer.shape[-1]
                # Average over heads
                processed_attentions = [np.mean(layer_attn, axis=0) for layer_attn in attention_data]
            elif first_layer.ndim == 2:  # [seq, seq]
                seq_len = first_layer.shape[-1]
                processed_attentions = attention_data
            else:
                print(f"Unexpected attention tensor shape: {first_layer.shape}")
                return None
            
            # Ensure all matrices are 2D and float32
            processed_attentions = [np.array(matrix, dtype=np.float32) for matrix in processed_attentions]
            
            # Truncate tokens if necessary
            display_tokens = tokens[:seq_len] if len(tokens) > seq_len else tokens
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            # Initialize with first layer
            avg_attention = processed_attentions[0]
            enhanced_attn = AttentionVisualizer._enhance_attention_contrast(avg_attention)
            
            cax = ax.imshow(enhanced_attn, 
                           cmap=AttentionVisualizer.create_tonal_colormap('blue'),
                           aspect='auto')
            
            ax.set_xticks(range(len(display_tokens)))
            ax.set_xticklabels(display_tokens, rotation=90)
            ax.set_yticks(range(len(display_tokens)))
            ax.set_yticklabels(display_tokens)
            title = ax.set_title("Layer 0")
            plt.colorbar(cax)

            def update(frame):
                layer = frame % max_layers
                avg_attention = processed_attentions[layer]
                enhanced_attn = AttentionVisualizer._enhance_attention_contrast(avg_attention)
                
                # Ensure enhanced_attn is float32
                enhanced_attn = np.array(enhanced_attn, dtype=np.float32)
                
                cax.set_data(enhanced_attn)
                title.set_text(f"Layer {layer}")
                return cax,

            anim = FuncAnimation(fig, update, frames=max_layers, interval=1000, blit=False)
            plt.tight_layout()
            
            # Save animation as GIF in assets directory
            animation_path = os.path.join(AttentionVisualizer.ASSETS_DIR, "attention_animation.gif")
            anim.save(animation_path, writer='pillow')
            plt.close()
            
            return animation_path
            
        except Exception as e:
            print(f"Error creating attention animation: {str(e)}")
            plt.close('all')  # Cleanup on error
            return None

    @staticmethod
    def _enhance_attention_contrast(attention_matrix):
        """Enhance attention contrast for better visualization"""
        try:
            # Ensure input is numpy array with float dtype
            attn = np.array(attention_matrix, dtype=np.float32)
            
            # Handle edge cases
            if attn.size == 0:
                return attn
                
            # Simple normalization to avoid complex operations
            attn_min = np.min(attn)
            attn_max = np.max(attn)
            
            if attn_max > attn_min:
                enhanced = (attn - attn_min) / (attn_max - attn_min)
            else:
                enhanced = np.zeros_like(attn)
                
            return enhanced.astype(np.float32)
            
        except Exception as e:
            print(f"Error enhancing attention contrast: {str(e)}")
            # Return original matrix as fallback
            return np.array(attention_matrix, dtype=np.float32)
