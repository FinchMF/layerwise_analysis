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
        att = attn_tensor[layer][0].numpy()
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

    @staticmethod
    def enhance_contrast(values, power=2):
        """Enhance contrast of values using power normalization.

        Args:
            values (np.ndarray): Input values to enhance.
            power (float, optional): Power factor for normalization. Defaults to 2.

        Returns:
            np.ndarray: Contrast-enhanced values.
        """
        # ...existing code...

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
        att = attn_tensor[layer][0].numpy()
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
        att = attn_tensor[layer][0].numpy()
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
            # Convert attention tensor to numpy and ensure float32
            att = attn_tensor[layer][0].cpu().numpy().astype(np.float32)
            
            # Close any existing figures to prevent memory leaks
            plt.close('all')
            
            fig, axes = plt.subplots(nrows=1, ncols=min(4, att.shape[0]), figsize=(16, 4))
            axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes
            
            for i, ax in enumerate(axes.flat):
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
            plt.close('all')  # Cleanup on error
            return None

    @staticmethod
    def create_attention_animation(attn_tensor, tokens, max_layers=12):
        """Create animated visualization of attention patterns across layers.

        Args:
            attn_tensor (torch.Tensor): Attention tensors from model.
            tokens (list): List of token strings.
            max_layers (int, optional): Maximum number of layers. Defaults to 12.

        Returns:
            str: Path to saved animation file.
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        avg_attention = attn_tensor[0][0].numpy().mean(axis=0)
        # Enhance initial frame
        enhanced_attn = AttentionVisualizer.enhance_contrast(
            (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())
        )
        cax = ax.imshow(enhanced_attn, 
                       cmap=AttentionVisualizer.create_tonal_colormap('blue'),
                       norm=PowerNorm(gamma=0.7))
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        title = ax.set_title("Layer 0")
        plt.colorbar(cax)

        def update(frame):
            layer = frame % max_layers
            avg_attention = attn_tensor[layer][0].numpy().mean(axis=0)
            enhanced_attn = AttentionVisualizer.enhance_contrast(
                (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())
            )
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

    def create_complete_visualization(self, df, attn_tensor, tokens, layer):
        """Generate comprehensive set of visualizations.

        Args:
            df (pd.DataFrame): DataFrame containing layer metrics.
            attn_tensor (torch.Tensor): Attention tensors from model.
            tokens (list): List of token strings.
            layer (int): Layer index to analyze.

        Returns:
            list: List of matplotlib figures containing all visualizations.
        """
        figs = [
            self.plot_layer_metrics(df),
            self.plot_metric_heatmap(df),
            self.plot_metric_correlations(df),
            self.cluster_attention_heads(attn_tensor, layer),
            self.plot_attention_entropy_heatmap(attn_tensor, layer),
            self.visualize_token_attribution(attn_tensor, tokens, layer)
        ]
        return figs
