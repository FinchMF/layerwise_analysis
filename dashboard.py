"""
Main dashboard interface for transformer model analysis.

This module provides the main interface for the transformer analysis dashboard,
integrating visualization, analysis, and interaction components using Gradio.

Classes:
    Dashboard: Main dashboard application class.
"""

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before other imports

import gradio as gr
import pandas as pd
from src.models import ModelManager
from src.metrics import LayerMetrics
from src.visualizers import AttentionVisualizer
from src.exporters import ResultExporter
from src.state import AnalysisState
from src.docs import METRIC_DESCRIPTIONS, ATTENTION_DESCRIPTIONS
import os
import sys
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import signal

class HotReloadHandler(FileSystemEventHandler):
    """Handler for file system events to trigger hot reload"""
    
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_reload = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only reload for Python files
        if not event.src_path.endswith('.py'):
            return
            
        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_reload < 1:
            return
            
        self.last_reload = current_time
        print(f"File changed: {event.src_path}")
        print("Reloading application...")
        
        # Trigger restart in a separate thread to avoid blocking
        threading.Thread(target=self.restart_callback, daemon=True).start()

class Dashboard:
    """Main dashboard application class.
    
    Integrates all components of the analysis system and provides
    the main interface for user interaction through Gradio.
    
    Attributes:
        model_manager (ModelManager): Handles model operations.
        metrics (LayerMetrics): Provides metric calculations.
        visualizer (AttentionVisualizer): Creates visualizations.
        exporter (ResultExporter): Handles result export.
        state (AnalysisState): Maintains analysis state.
    """
    def __init__(self):
        self.model_manager = ModelManager()
        self.metrics = LayerMetrics()
        self.visualizer = AttentionVisualizer()
        self.exporter = ResultExporter()
        self.state = AnalysisState()
        self.observer = None
        self.demo = None

    def run_analysis(self, text, model_choice):
        """Initial analysis that populates the shared state"""
        try:
            # Ensure model is loaded successfully
            if not self.model_manager.load_model(model_choice):
                return (
                    "Error: Failed to load model. Please try again.",
                    None, None, None
                )
            
            # Check if model is actually loaded
            if self.model_manager.model is None:
                return (
                    "Error: Model is not properly loaded. Please try again.",
                    None, None, None
                )
            
            hidden_states, attentions, tokens, predictions = self.model_manager.analyze_text(text)
            
            if hidden_states is None or attentions is None:
                return (
                    "Error: Failed to analyze text. Please check your input.",
                    None, None, None
                )
            
            results = []
            input_vec = hidden_states[0][0].cpu().numpy()[:10, :128]
            
            for i, layer_tensor in enumerate(hidden_states):
                rep = layer_tensor[0].cpu().numpy()[:10, :128]
                results.append({
                    "layer": i,
                    "effective_rank": self.metrics.effective_rank(rep),
                    "participation_ratio": self.metrics.participation_ratio(rep),
                    "intrinsic_dim": self.metrics.intrinsic_dimensionality(rep),
                    "activation_entropy": self.metrics.mean_activation_entropy(rep),
                    "cosine_sim_to_input": self.metrics.cosine_sim_to_input(input_vec.mean(axis=0), rep)
                })

            df = pd.DataFrame(results)
            self.state.update(hidden_states, attentions, tokens, df, model_choice, text, predictions)
            
            # Format prediction output
            if predictions["type"] == "mask_prediction":
                prediction_text = f"Predictions for [MASK]:\n" + "\n".join(predictions['predictions'])
            else:
                prediction_text = f"Next token: {predictions['next_token']}\nFull text: {predictions['full_text']}"
            
            return (
                prediction_text,  # Text output
                self.visualizer.plot_layer_metrics(df),
                self.visualizer.plot_metric_heatmap(df),
                self.visualizer.plot_metric_correlations(df),
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(f"Error in run_analysis: {e}")
            return (error_msg, None, None, None)

    def analyze_layer(self, layer):
        """Layer-specific analysis using shared state"""
        try:
            if not self.state.is_initialized():
                return [None] * 4
            
            return (
                self.visualizer.cluster_attention_heads(self.state.attentions, layer),
                self.visualizer.plot_attention_entropy_heatmap(self.state.attentions, layer),
                self.visualizer.visualize_token_attribution(self.state.attentions, self.state.tokens, layer),
                self.visualizer.analyze_attention_head_attribution(self.state.attentions, self.state.tokens, layer)
            )
        except Exception as e:
            print(f"Error in analyze_layer: {e}")
            return [None] * 4

    def create_animation(self):
        """Create layer transition animation"""
        try:
            if not self.state.is_initialized():
                return None
            return self.visualizer.create_attention_animation(self.state.attentions, self.state.tokens)
        except Exception as e:
            print(f"Error creating animation: {e}")
            return None

    def setup_hot_reload(self):
        """Setup file watching for hot reload"""
        try:
            # Watch the src directory and main dashboard file
            watch_paths = [
                os.path.join(os.path.dirname(__file__), 'src'),
                __file__  # Watch the main dashboard file
            ]
            
            self.observer = Observer()
            handler = HotReloadHandler(self.restart_app)
            
            for path in watch_paths:
                if os.path.exists(path):
                    self.observer.schedule(handler, path, recursive=True)
                    print(f"Watching for changes in: {path}")
            
            self.observer.start()
            
        except Exception as e:
            print(f"Could not setup hot reload: {e}")
            self.observer = None
    
    def restart_app(self):
        """Restart the application"""
        try:
            # Close current demo if it exists
            if self.demo is not None:
                self.demo.close()
                time.sleep(0.5)  # Give it time to close
            
            # Reload modules
            self.reload_modules()
            
            # Recreate the interface
            self.demo = self.create_interface()
            
            # Restart the server
            self.demo.launch(
                server_name="127.0.0.1",
                server_port=7862,
                share=False,
                inbrowser=False,
                quiet=True
            )
            
        except Exception as e:
            print(f"Error restarting app: {e}")
    
    def reload_modules(self):
        """Reload all modules for hot reload"""
        modules_to_reload = []
        
        # Find all modules that start with our package path
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        for module_name in list(sys.modules.keys()):
            module = sys.modules[module_name]
            if hasattr(module, '__file__') and module.__file__:
                if src_path in module.__file__:
                    modules_to_reload.append(module_name)
        
        # Reload modules
        for module_name in modules_to_reload:
            try:
                import importlib
                importlib.reload(sys.modules[module_name])
                print(f"Reloaded: {module_name}")
            except Exception as e:
                print(f"Could not reload {module_name}: {e}")
    
    def create_interface(self):
        """Create the Gradio interface"""
        dashboard = self
        
        with gr.Blocks(title="Layer Analysis Dashboard") as demo:
            with gr.Tab("Overall Analysis"):
                gr.Markdown("### Layer-wise Metrics Analysis")
                
                # Add model-specific instructions
                model_instructions = gr.Markdown("""
                **Input Instructions:**
                - For BERT: Include [MASK] token in your text for prediction (e.g., "The cat [MASK] on the mat")
                - For GPT-2/TinyLlama: Enter text for single-token continuation
                
                Note: This analysis shows a single forward pass/generation step and is focused on measuring the change over layers.
                """)
                
                text_input = gr.Textbox(label="Input Sentence")
                model_choice = gr.Dropdown(
                    choices=["bert-base-uncased", "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
                    label="Model Choice"
                )

                # Move control vector options here
                with gr.Row():
                    control_type = gr.Dropdown(
                        choices=["none", "sentiment"],
                        value="none",
                        label="Control Vector Type"
                    )
                    control_direction = gr.Dropdown(
                        choices=["positive", "negative"],
                        label="Control Direction",
                        interactive=False
                    )
                    control_strength = gr.Slider(
                        minimum=0, maximum=2, value=1,
                        label="Control Strength",
                        interactive=False
                    )

                analyze_btn = gr.Button("Analyze")
                prediction_output = gr.Textbox(label="Model Predictions", interactive=False)
                
                with gr.Row():
                    metrics_plot = gr.Plot(label="Layer-wise Metrics")
                    heatmap_plot = gr.Plot(label="Metric Heatmap")
                with gr.Row():
                    correlation_plot = gr.Plot(label="Metric Correlations")
            
                def update_controls(control_type, model_choice):
                    """Update control vector settings based on selected type."""
                    if control_type == "none":
                        dashboard.model_manager.clear_control()
                        return gr.update(interactive=False), gr.update(interactive=False)
                    else:
                        # Ensure model is loaded before generating controls
                        if not dashboard.model_manager.model:
                            dashboard.model_manager.load_model(model_choice)
                        dashboard.model_manager.control_generator.generate_sentiment_controls()
                        return gr.update(interactive=True), gr.update(interactive=True)

                # Update control type change handler
                control_type.change(
                    fn=update_controls,
                    inputs=[control_type, model_choice],
                    outputs=[control_direction, control_strength]
                )
                
                def run_with_controls(text, model_choice, c_type, c_direction, c_strength):
                    if c_type != "none":
                        dashboard.model_manager.set_control(c_type, c_direction, c_strength)
                    return dashboard.run_analysis(text, model_choice)
                
                # Update analyze button to include control parameters
                analyze_btn.click(
                    fn=run_with_controls,
                    inputs=[
                        text_input, model_choice,
                        control_type, control_direction, control_strength
                    ],
                    outputs=[
                        prediction_output,
                        metrics_plot,
                        heatmap_plot,
                        correlation_plot
                    ]
                )
            
            with gr.Tab("Layer Analysis"):
                gr.Markdown("### Attention Analysis for Specific Layer")
                layer_slider = gr.Slider(0, 11, step=1, value=0, label="Select Layer")
                
                with gr.Row():
                    cluster_plot = gr.Plot(label="Attention Head Clustering")
                    entropy_plot = gr.Plot(label="Attention Entropy Heatmap")
                with gr.Row():
                    attribution_plot = gr.Plot(label="Token Attribution")
                    head_attribution_plot = gr.Plot(label="Individual Head Attribution")
                
                animate_btn = gr.Button("Show Layer Animation")
                animation_output = gr.Video(label="Layer Transition Animation") # Changed from Plot to Video
            
            with gr.Tab("Details"):
                with gr.Tab("Layer Metrics"):
                    for metric, info in METRIC_DESCRIPTIONS.items():
                        with gr.Accordion(info["title"], open=False):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown(f"""
                                    ### Description
                                    {info['description']}
                                    
                                    ### Formula
                                    ```
                                    {info['formula']}
                                    ```
                                    
                                    ### Interpretation
                                    {info['interpretation']}
                                    """)
                
                with gr.Tab("Attention Analysis"):
                    for analysis, info in ATTENTION_DESCRIPTIONS.items():
                        with gr.Accordion(info["title"], open=False):
                            gr.Markdown(f"""
                            ### Description
                            {info['description']}
                            
                            ### Interpretation
                            {info['interpretation']}
                            """)

            # Connect components
            analyze_btn.click(
                fn=dashboard.run_analysis,
                inputs=[text_input, model_choice],
                outputs=[
                    prediction_output,  # Text output for predictions
                    metrics_plot,
                    heatmap_plot,
                    correlation_plot
                ]
            )
            
            layer_slider.change(
                fn=dashboard.analyze_layer,
                inputs=[layer_slider],
                outputs=[cluster_plot, entropy_plot, attribution_plot, head_attribution_plot]
            )
            
            animate_btn.click(
                fn=dashboard.create_animation,
                inputs=[],
                outputs=[animation_output]
            )
        
        return demo

    def run(self, enable_hot_reload=True):
        """Run the dashboard with optional hot reload"""
        try:
            # Setup hot reload if enabled
            if enable_hot_reload:
                self.setup_hot_reload()
                print("Hot reload enabled. The server will automatically restart when files change.")
            
            # Create and launch the interface
            self.demo = self.create_interface()
            self.demo.launch(
                server_name="127.0.0.1",
                server_port=7862,
                share=False,
                inbrowser=True
            )
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            if self.demo:
                self.demo.close()

if __name__ == "__main__":
    app = Dashboard()
    
    # Check if hot reload should be disabled
    disable_hot_reload = "--no-reload" in sys.argv
    
    app.run(enable_hot_reload=not disable_hot_reload)
