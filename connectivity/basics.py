import numpy as np
import time

def compute_basic_measures(matrix):
	"""
	Computes all the basic measures that can be extracted from the attention maps including;
	1. mean attention
	2. max attention
	3. entropy
	4. gini coefficient
	5. effective rank
	"""

	features = {}
	features["mean_attention"] = matrix.mean()
	start_time = time.time()
	features["max_attention"] = matrix.max()
	end_time = time.time()
	print(f"Time spent in max_attention = {end_time-start_time}")

	probs = matrix.flatten()
	probs = probs/(probs.sum()+1e-10)
	features["entropy"] = -np.sum(probs * np.log(probs+1e-10))

	sorted_weights = np.sort(matrix.flatten())
	n = len(sorted_weights)
	index = np.arange(1, n+1)
	features["gini"] = (2*np.sum(index*sorted_weights))/(n* np.sum(sorted_weights)) - (n+1)/n

	features["effective_rank"] = np.exp(features["entropy"])

	return features
