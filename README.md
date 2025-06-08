# Transformer Model Analysis Dashboard

An interactive dashboard for analyzing and visualizing the internal representations and attention mechanisms of transformer-based language models (BERT, GPT-2, and TinyLlama).

## Features

- **Layer-wise Analysis**:
  - Effective rank of representations
  - Participation ratio
  - Intrinsic dimensionality
  - Activation entropy
  - Cosine similarity to input layer

- **Attention Visualizations**:
  - Attention head clustering
  - Attention entropy heatmaps
  - Token attribution visualization
  - Per-head attention attribution
  - Layer transition animations
  - Individual head analysis

- **Model Predictions**:
  - BERT mask token completion
  - GPT-2 next token prediction
  - TinyLlama text continuation

- **Detailed Documentation**:
  - Comprehensive metric explanations
  - Mathematical formulas
  - Interpretation guides
  - Visual analysis documentation

- **Export Capabilities**:
  - Comprehensive PDF reports
  - CSV data export
  - Attention weights in NumPy format
  - Session logging
  - All outputs bundled in ZIP format

## Supported Models

- BERT (bert-base-uncased)
- GPT-2 (gpt2)
- TinyLlama (TinyLlama-1.1B-Chat)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the dashboard:
```bash
python dashboard.py
```

The interface will be available at `http://localhost:7860`

## Analysis Metrics Explained

- **Effective Rank**: Measures the functional dimensionality of layer representations
- **Participation Ratio**: Quantifies how evenly distributed the singular values are
- **Intrinsic Dimensionality**: Minimum dimensions needed to capture 95% of variance
- **Activation Entropy**: Measures the information content of neuron activations
- **Cosine Similarity**: Tracks how representations evolve through layers

## Output Files

The dashboard generates a comprehensive analysis bundle including:
- Layer analysis results (CSV)
- Attention visualizations (PNG)
- Summary report (PDF)
- Session log (HTML)
- Raw attention weights (NPY)

## Interface Tabs

1. **Overall Analysis**:
   - Layer-wise metric plots
   - Model predictions display
   - Metric correlations
   - Normalized metric heatmaps

2. **Layer Analysis**:
   - Interactive layer selection
   - Attention head clustering
   - Token attribution maps
   - Attention entropy analysis
   - Layer transition animations

3. **Details**:
   - In-depth metric documentation
   - Formula explanations
   - Interpretation guidelines
   - Analysis methodology

## Analysis Assets

All analysis artifacts are stored in the `assets` directory:
- Attention visualizations
- Layer animations
- Metric plots
- Analysis reports
- Session logs

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See requirements.txt for full dependencies
