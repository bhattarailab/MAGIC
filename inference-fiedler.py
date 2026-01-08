import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

metadata_fidler_path = "/work/FAC/HEC/DESI/yshresth/aim/noli1/graph-fiedler/outputs/fiedler_values.pkl"

with open(metadata_fidler_path, "rb") as fp:
	# each sample contains "fiedlers" with dict of keys "layer_0_head_0" with "text" and "visual"
	metadata_with_fiedler = pickle.load(fp)


# Task: for each layer in each head compute the divegence score
# 1. Divide into vision-aware and text-aware samples.
# 2. For, each samples for a particular head compute the divergence score
# 	a. divergence = visual fiedler
# 	b. divergence = visual - text 
# 3. Compute the difference of divergence score in hallucinated and non-hallucianted samples
# 4. Plot the visualization in a scatter plot
# 5. Compute the acuracy
def compute_fiedler_difference(fiedlers):
	return fiedlers["text"]-fiedlers["visual"]
	# return fiedlers["visual"]

def compute_normalized_fiedler_difference(samples):
	cumulative_fiedler_differences = dict()

	# for each layer's head compute the difference of divergence score
	for i_layer in range(32):
		for i_head in range(32):
			key = f"layer_{i_layer}_head_{i_head}"
			total_head_fiedler_difference = 0
			
			# for all the given set of samples compute the divergence score for that head
			for sample in samples:
				fiedler_difference = compute_fiedler_difference(sample["fiedlers"][key])
				total_head_fiedler_difference += fiedler_difference

			cumulative_fiedler_differences[key] = total_head_fiedler_difference/len(samples)

	return dict_to_matrix(cumulative_fiedler_differences)

def dict_to_matrix(dictionary, n_layers=32, n_heads=32):
	score_array = []
	for i_layer in range(n_layers):
		ith_layer_array = []
		for i_head in range(n_heads):
			key= f"layer_{i_layer}_head_{i_head}"
			ith_layer_array.append(dictionary[key])

		score_array.append(ith_layer_array)

	return np.array(score_array)

def get_descending_matrix_indices(matrix):
	flattened = np.argsort(matrix, axis=None)[::-1]
	row_col_indices = np.column_stack(np.unravel_index(flattened, matrix.shape))

	return row_col_indices

def select_heads(divergece_matrix, validation_samples, max_heads=10):
	n_heads_max = max_heads
	descending_divergence_heads = get_descending_matrix_indices(divergece_matrix)

	N, N_optimal = 1, 1
	heads = list()

	p_s = {s: 0 for s in range(len(validation_samples))}
	auc_max = 0.0

	while N<=n_heads_max:
		heads.append(descending_divergence_heads[N-1])

		for i, s in enumerate(validation_samples):
			l, h = heads[-1]
			l, h = int(l), int(h)

			fiedlers = s["fiedlers"][f"layer_{l}_head_{h}"]
			p_s[i] = (N-1)/N * p_s[i] + 1/N * compute_fiedler_difference(fiedlers)

		y_s = [s["is_hallucinating"] for s in validation_samples]
		auc = roc_auc_score(y_s, list(p_s.values()))

		if auc > auc_max:
			auc_max = auc
			N_optimal = N

		N = N+1

	print(f"Total {N_optimal} heads selected with maximum AUC of {auc_max}")
	
	return descending_divergence_heads[:N_optimal]

hallucinating_samples = [m for m in metadata_with_fiedler if m["is_hallucinating"]]
non_hallucinating_samples = [m for m in metadata_with_fiedler if not m["is_hallucinating"]]

hallucinating_fiedlers_diff = compute_normalized_fiedler_difference(hallucinating_samples)
non_hallucinating_fiedlers_diff = compute_normalized_fiedler_difference(non_hallucinating_samples)

divergence_scores = hallucinating_fiedlers_diff-non_hallucinating_fiedlers_diff

ax = sns.heatmap(divergence_scores)
plt.xlabel("heads")
plt.ylabel("layers")
plt.savefig("./outputs/divergence_scores.png")

selected_heads = select_heads(divergence_scores, metadata_with_fiedler)