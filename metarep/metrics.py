__all__ = ["r2_score", "monosemanticity_score"]

import numpy as np
import torch
from tqdm import tqdm


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    computes $R^2$ (coefficient of determination) regression score.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def monosemanticity_score(
    embeddings: np.ndarray,  # raw activation embeddings from the encoder. has shape (n_observations, n_embedding_features)
    sae_activations: np.ndarray,  # the activations from the SAE. has shape (n_observations, n_sae_neurons)
    batch_size: int = 512,  # batch size for processing the similarity matrix to reduce memory usage
) -> np.ndarray:  # monosemanticity score for each SAE neuron, has shape (n_sae_neurons)
    """
    Computes the monosemanticity score for each SAE neuron, as described by [Pach et al. (2025)](https://arxiv.org/abs/2504.02821v1)
    
    Direct adaptation of the authors' implementation [here](https://github.com/ExplainableML/sae-for-vlm/blob/main/metric.py)
    """
    
    # Scale activations to 0-1 per neuron (min-max normalization)
    min_values = np.min(sae_activations, axis=0, keepdims=True)
    max_values = np.max(sae_activations, axis=0, keepdims=True)
    activations = (sae_activations - min_values) / (max_values - min_values)
    
    num_images, _ = embeddings.shape
    num_neurons = activations.shape[1]
    
    # initialize accumulators
    weighted_cosine_similarity_sum = np.zeros(num_neurons)
    weight_sum = np.zeros(num_neurons)
    
    for i in tqdm(range(num_images)):
        for j_start in tqdm(range(i + 1, num_images, batch_size), desc="batching", leave=False):
            j_end = min(j_start + batch_size, num_images)
            
            embeddings_i = embeddings[i]  # (embedding_dim,)
            embeddings_j = embeddings[j_start:j_end]  # (batch_size, embedding_dim)
            activations_i = activations[i]  # (num_neurons,)
            activations_j = activations[j_start:j_end]  # (batch_size, num_neurons)
            
            # compute cosine similarity
            embeddings_i_norm = embeddings_i / (np.linalg.norm(embeddings_i) + 1e-8)
            embeddings_j_norm = embeddings_j / (np.linalg.norm(embeddings_j, axis=1, keepdims=True) + 1e-8)            
            cosine_similarities = np.dot(embeddings_j_norm, embeddings_i_norm)  # (batch_size,)
            
            # compute weights and weighted similarities
            # expanding activations_i to (1, num_neurons)
            weights = activations_i[np.newaxis, :] * activations_j  # (batch_size, num_neurons)
            weighted_cosine_similarities = weights * cosine_similarities[:, np.newaxis]  # (batch_size, num_neurons)
            
            weighted_cosine_similarities = np.sum(weighted_cosine_similarities, axis=0)  # (num_neurons,)
            weighted_cosine_similarity_sum += weighted_cosine_similarities
            
            weights = np.sum(weights, axis=0)  # (num_neurons,)
            weight_sum += weights
    
    # compute final monosemanticity scores
    monosemanticity = np.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, np.nan)
    
    return monosemanticity