from sklearn.metrics import roc_auc_score
import numpy as np

# Task: for each layer in each head compute the divegence score
# 1. Divide into vision-aware and text-aware samples.
# 2. For, each samples for a particular head compute the divergence score
# 3. Compute the difference of divergence score in hallucinated and non-hallucianted samples
# 4. Plot the visualization in a scatter plot
# 5. Compute the acuracy
def compute_connectivity_diff(connectivity, t_weight=1, v_weight=-1):
	return t_weight*connectivity["text"]+v_weight*connectivity["visual"]

def compute_normalized_connectivity_diff(samples, t_weight=1, v_weight=-1):
	cumulative_connectivity_differences = dict()

	# for each layer's head compute the difference of divergence score
	for i_layer in range(32):
		for i_head in range(32):
			key = f"layer_{i_layer}_head_{i_head}"
			total_head_connectivity_difference = 0
			
			# for all the given set of samples compute the divergence score for that head
			for sample in samples:
				connectivity_difference = compute_connectivity_diff(sample["connectivity"][key], t_weight=t_weight, v_weight=v_weight)
				total_head_connectivity_difference += connectivity_difference

			cumulative_connectivity_differences[key] = total_head_connectivity_difference/len(samples)

	return dict_to_matrix(cumulative_connectivity_differences)

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

def select_heads(divergece_matrix, validation_samples, max_heads=10, t_weight=1, v_weight=-1):
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

			connectivity = s["connectivity"][f"layer_{l}_head_{h}"]
			p_s[i] = (N-1)/N * p_s[i] + 1/N * compute_connectivity_diff(connectivity, t_weight=t_weight, v_weight=v_weight)

		y_s = [s["is_hallucinating"] for s in validation_samples]
		auc = roc_auc_score(y_s, list(p_s.values()))

		if auc > auc_max:
			auc_max = auc
			N_optimal = N

		N = N+1

	print(f"Total {N_optimal} heads selected with maximum AUC of {auc_max}")
	
	return descending_divergence_heads[:N_optimal]

def prediction(heads, samples, t_weight=1, v_weight=-1):
	y_s = [s["is_hallucinating"] for s in samples]
	n_heads = len(heads)

	p_s = []
	for s in samples:
		connectivity_diff = 0
		for (l, h) in heads:
			key = f"layer_{l}_head_{h}"
			connectivity_diff += compute_connectivity_diff(s["connectivity"][key], t_weight=t_weight, v_weight=v_weight)

		p_s.append(connectivity_diff/n_heads)

	auc_score = roc_auc_score(y_s, p_s)
	print(f"AUC score = {auc_score}")