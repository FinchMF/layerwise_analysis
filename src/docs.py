"""
Documentation content for dashboard metrics.

This module contains detailed descriptions, formulas, and interpretations
for all metrics and analyses provided by the dashboard.

Constants:
    METRIC_DESCRIPTIONS (dict): Detailed descriptions of analysis metrics.
    ATTENTION_DESCRIPTIONS (dict): Descriptions of attention analyses.
"""

METRIC_DESCRIPTIONS = {
    "effective_rank": {
        "title": "Effective Rank",
        "description": "Measures the functional dimensionality of layer representations. A higher value indicates the layer uses more of its available dimensions effectively.",
        "formula": "exp(-Σ(p_i * log(p_i))) where p_i are normalized singular values",
        "interpretation": "Values closer to the maximum dimension indicate more distributed representations, while lower values suggest more concentrated information."
    },
    "participation_ratio": {
        "title": "Participation Ratio",
        "description": "Quantifies how evenly distributed the singular values are across dimensions. Indicates the number of 'effective' dimensions being used.",
        "formula": "(Σs_i²)² / Σ(s_i⁴) where s_i are singular values",
        "interpretation": "Higher values indicate more distributed representations across dimensions."
    },
    "intrinsic_dim": {
        "title": "Intrinsic Dimensionality",
        "description": "The minimum number of dimensions needed to capture 95% of the variance in the representations.",
        "formula": "min(k) where Σ(λ_i)[:k] / Σ(λ_i) >= 0.95",
        "interpretation": "Lower values indicate more efficient compression of information."
    },
    "activation_entropy": {
        "title": "Activation Entropy",
        "description": "Measures the information content and uncertainty in neuron activations.",
        "formula": "-Σ(p_i * log(p_i)) for normalized activations",
        "interpretation": "Higher values indicate more diverse and distributed activation patterns."
    },
    "cosine_sim_to_input": {
        "title": "Cosine Similarity to Input",
        "description": "Measures how much the layer's representations have diverged from the input embedding.",
        "formula": "cos(θ) = (a·b)/(||a||·||b||)",
        "interpretation": "Values closer to 1 indicate preservation of input features, while lower values show transformation."
    }
}

ATTENTION_DESCRIPTIONS = {
    "clustering": {
        "title": "Attention Head Clustering",
        "description": "Groups attention heads based on their attention patterns using K-means clustering.",
        "interpretation": "Clusters indicate groups of heads that share similar attention behaviors."
    },
    "entropy": {
        "title": "Attention Entropy",
        "description": "Measures the uncertainty or spread of attention weights for each head.",
        "interpretation": "Higher entropy indicates more distributed attention, lower entropy suggests focused attention."
    },
    "attribution": {
        "title": "Token Attribution",
        "description": "Shows how attention is distributed across input tokens.",
        "interpretation": "Darker colors indicate stronger attention weights between tokens."
    }
}
