import numpy as np

def compute_gini(values):
	sorted_values = np.sort(values.flatten())
	n = len(sorted_values)

	# Handle edge cases
	if n == 0 or np.sum(sorted_values) == 0:
			breakpoint()
			return 0.0
	
	breakpoint()
	index = np.arange(1, n+1)
	return (2*np.sum(index*sorted_values))/(n* np.sum(sorted_values)) - (n+1)/n

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
	features["max_attention"] = matrix.max()

	probs = matrix.flatten()
	probs = probs/(probs.sum()+1e-10)
	features["entropy"] = -np.sum(probs * np.log(probs+1e-10))
	features["gini"] = compute_gini(matrix)
	features["per_token_gini"] = np.mean([compute_gini(m) for m in matrix])
	features["effective_rank"] = np.exp(features["entropy"])

	return features
