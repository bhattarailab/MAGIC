import numpy as np

def compute_gini(values):
	sorted_values = np.sort(values.flatten())
	n = len(sorted_values)

	# Handle edge cases
	if n == 0 or np.sum(sorted_values) == 0:
			return 0.0
	
	index = np.arange(1, n+1)
	return (2*np.sum(index*sorted_values))/(n* np.sum(sorted_values)) - (n+1)/n

def compute_gini_traditional(values):
	values = values.flatten()

	if np.amin(values) < 0:
		print(f"Warning non-negative values is not defined for GINI")
		return 0.0
	if np.sum(values) == 0:
		print(f"Warning, all values received 0")
		return 0.0
	
	n = len(values)
	diff_sum = np.abs(values[:, None] - values[None, :]).sum()
	
	return diff_sum/(2*n**2*np.mean(values))

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
	features["gini_traditional"] = compute_gini_traditional(matrix)
	features["per_token_gini"] = np.mean([compute_gini(m) for m in matrix])
	features["effective_rank"] = np.exp(features["entropy"])

	return features
