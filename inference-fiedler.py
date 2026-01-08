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
def compute_divergence_of_head(fiedlers):
	return fiedlers["text"]-fiedlers["visual"]
	# return fiedlers["visual"]

def compute_divergence_across_samples(samples):
	cumulative_divergence_scores = dict()

	# for each layer's head compute the difference of divergence score
	for i_layer in range(32):
		for i_head in range(32):
			key = f"layer_{i_layer}_head_{i_head}"
			total_head_divergence = 0
			
			# for all the given set of samples compute the divergence score for that head
			for sample in samples:
				divergence = compute_divergence_of_head(sample["fiedlers"][key])
				total_head_divergence += divergence

			cumulative_divergence_scores[key] = total_head_divergence

	return dict_to_matrix(cumulative_divergence_scores)

def dict_to_matrix(dictionary):
	score_array = []
	for i_layer in range(32):
		ith_layer_array = []
		for i_head in range(32):
			key= f"layer_{i_layer}_head_{i_head}"
			ith_layer_array.append(dictionary[key])

		score_array.append(ith_layer_array)

	return np.array(score_array)

hallucinating_samples = [m for m in metadata_with_fiedler if m["is_hallucinating"]]
non_hallucinating_samples = [m for m in metadata_with_fiedler if not m["is_hallucinating"]]

hallucinating_divergence, non_hallucinating_divergence = compute_divergence_across_samples(hallucinating_samples), compute_divergence_across_samples(non_hallucinating_samples)

hallucinating_divergence = hallucinating_divergence/len(hallucinating_samples)
non_hallucinating_divergence = non_hallucinating_divergence/len(non_hallucinating_samples)

aggregated_divergence = hallucinating_divergence-non_hallucinating_divergence

ax = sns.heatmap(aggregated_divergence)
plt.xlabel("heads")
plt.ylabel("layers")
plt.savefig("./outputs/aggregated_divergence.png")


def select_heads(aggregated_divergence):
	n_heads_max = 10

	# flattening the aggregated_divergence matrix to sort in descending order
	flattened_aggregare_divergence = np.argsort(aggregated_divergence, axis=None)[::-1]
	row_col_indices = np.column_stack(np.unravel_index(flattened_aggregare_divergence, aggregated_divergence.shape))

	N, N_optimal = 1, 1
	heads = list()

	p_s = {s: 0 for s in range(len(metadata_with_fiedler))}
	auc_max = 0.0

	while N<=n_heads_max:
		heads.append(row_col_indices[N-1])

		for i, s in enumerate(metadata_with_fiedler):
			l, h = heads[-1]
			l, h = int(l), int(h)

			fiedlers = s["fiedlers"][f"layer_{l}_head_{h}"]
			p_s[i] = (N-1)/N * p_s[i] + 1/N * compute_divergence_of_head(fiedlers)

		y_s = [s["is_hallucinating"] for s in metadata_with_fiedler]
		auc = roc_auc_score(y_s, list(p_s.values()))

		if auc > auc_max:
			auc_max = auc
			N_optimal = N

		N = N+1
	
	return row_col_indices[:N_optimal]

selected_heads = select_heads(aggregated_divergence)