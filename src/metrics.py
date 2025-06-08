"""
Neural network layer analysis metrics.

This module provides various metrics for analyzing transformer model layers,
including measures of dimensionality, entropy, and representation similarity.

Classes:
    LayerMetrics: Collection of static methods for computing layer analysis metrics.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy

class LayerMetrics:
    """Computation methods for layer analysis metrics.
    
    Provides static methods for computing various metrics that characterize
    neural network layer representations, including effective rank,
    participation ratio, and intrinsic dimensionality.
    """

    @staticmethod
    def effective_rank(matrix):
        """Calculate effective rank using singular value entropy.
        
        Args:
            matrix (np.ndarray): Input representation matrix.
            
        Returns:
            float: Effective rank value.
        """
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        p = s / s.sum()
        return np.exp(-np.sum(p * np.log(p + 1e-10)))

    @staticmethod
    def participation_ratio(matrix):
        """Compute the participation ratio of singular values.
        
        Args:
            matrix (np.ndarray): Input representation matrix.
            
        Returns:
            float: Participation ratio value.
        """
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
        s2 = s ** 2
        return (s2.sum()) ** 2 / (np.sum(s2 ** 2) + 1e-10)

    @staticmethod
    def intrinsic_dimensionality(matrix, threshold=0.95):
        """Estimate intrinsic dimensionality using PCA.
        
        Args:
            matrix (np.ndarray): Input representation matrix.
            threshold (float): Cumulative explained variance threshold.
            
        Returns:
            int: Estimated intrinsic dimensionality.
        """
        pca = PCA(n_components=min(matrix.shape))
        pca.fit(matrix)
        return np.searchsorted(np.cumsum(pca.explained_variance_ratio_), threshold) + 1

    @staticmethod
    def mean_activation_entropy(matrix):
        """Calculate mean activation entropy across rows.
        
        Args:
            matrix (np.ndarray): Input representation matrix.
            
        Returns:
            float: Mean activation entropy value.
        """
        matrix = matrix - matrix.min()
        matrix /= (matrix.sum(axis=1, keepdims=True) + 1e-10)
        return np.mean([scipy_entropy(row + 1e-10) for row in matrix])

    @staticmethod
    def cosine_sim_to_input(reference, target):
        """Compute mean cosine similarity between target and reference.
        
        Args:
            reference (np.ndarray): Reference input vector.
            target (np.ndarray): Target representation matrix.
            
        Returns:
            float: Mean cosine similarity value.
        """
        return cosine_similarity(target, reference.reshape(1, -1)).mean()
