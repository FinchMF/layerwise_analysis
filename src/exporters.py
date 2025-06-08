"""
Analysis results export module.

This module handles the export and bundling of analysis results,
including visualizations, data files, and session logs.

Classes:
    ResultExporter: Handles export and bundling of analysis artifacts.
"""

import os
import zipfile
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class ResultExporter:
    """Export handler for analysis results.
    
    Manages the export of analysis results, including CSV data,
    visualizations, and session logs. Provides bundling functionality
    for creating comprehensive analysis packages.
    
    Attributes:
        session_log (list): List of analysis session records.
    """
    ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    
    def __init__(self):
        self.session_log = []
        os.makedirs(self.ASSETS_DIR, exist_ok=True)

    def generate_pdf_report(self, df, layer, tokens, attn_tensor, visualizer):
        pdf_path = "analysis_report.pdf"
        with PdfPages(pdf_path) as pdf:
            # Generate and save all visualizations
            figures = [
                visualizer.cluster_attention_heads(attn_tensor, layer),
                visualizer.plot_attention_entropy_heatmap(attn_tensor, layer),
                visualizer.visualize_token_attribution(attn_tensor, tokens, layer)
            ]
            
            for fig in figures:
                pdf.savefig(fig)
        return pdf_path

    def save_bundle(self, df, attn_tensor, layer, tokens, text, model_choice, visualizer):
        paths = []
        
        # Update file paths to use assets directory
        csv_path = os.path.join(self.ASSETS_DIR, "analysis_output.csv")
        df.to_csv(csv_path, index=False)
        paths.append(csv_path)

        npy_path = os.path.join(self.ASSETS_DIR, f"attention_weights_layer_{layer}.npy")
        np.save(npy_path, attn_tensor[layer][0].numpy())
        paths.append(npy_path)

        # Save plots
        plot_paths = {
            "cluster_plot.png": visualizer.cluster_attention_heads(attn_tensor, layer),
            "entropy_heatmap.png": visualizer.plot_attention_entropy_heatmap(attn_tensor, layer),
            "token_attribution.png": visualizer.visualize_token_attribution(attn_tensor, tokens, layer)
        }
        
        for name, fig in plot_paths.items():
            path = os.path.join(self.ASSETS_DIR, name)
            fig.savefig(path)
            paths.append(path)

        # Save session log
        html_log_path = os.path.join(self.ASSETS_DIR, "session_log.html")
        with open(html_log_path, "w") as f:
            f.write("<html><body><h2>Session Log</h2><ul>")
            for entry in self.session_log:
                f.write(f"<li><b>Model:</b> {entry['model']}<br><b>Input:</b> {entry['text']}<br><b>Layer:</b> {entry['layer']}</li>")
            f.write("</ul></body></html>")
        paths.append(html_log_path)

        # Create ZIP in assets directory
        zip_path = os.path.join(self.ASSETS_DIR, "analysis_bundle.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in paths:
                zipf.write(file, os.path.basename(file))

        return zip_path

    def log_session(self, text, layer, model):
        self.session_log.append({
            "text": text,
            "layer": layer,
            "model": model
        })
