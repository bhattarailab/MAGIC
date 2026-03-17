import pickle
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from ripser import ripser


def transform_attention_scores_to_distances(attention: np.ndarray) -> np.ndarray:
    """
    Convert attention matrix to symmetric distance matrix.
    """
    attention = attention.astype(np.float32)
    n = attention.shape[-1]
    dist = 1.0 - np.clip(attention, a_min=0.0, a_max=None)
    zero_diag = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
    dist *= zero_diag
    # enforce symmetry
    dist = np.minimum(dist, dist.T)
    return dist


def compute_mtopdiv(distance_mx: np.ndarray) -> float:
    """
    Compute MTopDiv as sum of finite H0 persistence lifetimes.
    """
    dgms = ripser(distance_mx, distance_matrix=True, maxdim=0)["dgms"]

    if len(dgms[0]) <= 1:
        return 0.0

    # exclude the infinite bar
    return float(dgms[0][:-1, 1].sum())


def compute_attention_mtopdiv(
    attn: np.ndarray,
		**params
) -> dict:
    """
    Compute MTopDiv for a single attention matrix.
    """
    prompt_len= params["prompt_len"]
    response_length = attn.shape[-1]-prompt_len

    distance_matrix = transform_attention_scores_to_distances(attn)
    distance_matrix[:-response_length, :-response_length] = 0.0
    mtopdiv = compute_mtopdiv(distance_matrix) / max(1, response_length)
		
    return {
        "mtopdiv": float(mtopdiv),
    }
