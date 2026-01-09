import pickle
import numpy as np
import torch

from graph_utils import create_graph, draw_graph

import json
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def get_mst_total_weight(attention_matrix):
    """
    Calculates the total weight of the Minimum Spanning Tree (MST) 
    treating the Attention Matrix as a Bipartite Graph.
    """
    # 1. Convert Similarity (Attention) to Cost
    # High Attention (1.0) -> Low Cost (0.0)
    # Low Attention (0.0)  -> High Cost (1.0)
    cost_matrix = 1.0 - attention_matrix
    
    # Ensure no negative zeros cause issues, though unlikely with 1 - A
    cost_matrix = np.maximum(cost_matrix, 0.0)

    n_answer, n_visual = cost_matrix.shape
    n_total = n_answer + n_visual

    # 2. Efficiently build the Graph in Sparse Format (COO)
    # We do NOT create a dense NxN or MxM matrix. We just list the edges.
    
    # Create indices for a full bipartite connection
    # Rows: 0 to n_answer-1
    # Cols: 0 to n_visual-1
    rows, cols = np.indices((n_answer, n_visual))
    
    # Flatten everything
    row_indices = rows.flatten()
    col_indices = cols.flatten() + n_answer  # Shift visual indices by N
    data = cost_matrix.flatten()

    # CRITICAL: MST algorithms on 'undirected' graphs need the symmetric edges 
    # (i->j and j->i) to traverse the graph correctly.
    # We simply duplicate the data arrays to create the back-links efficiently.
    all_rows = np.concatenate([row_indices, col_indices])
    all_cols = np.concatenate([col_indices, row_indices])
    all_data = np.concatenate([data, data])

    # Create the sparse adjacency matrix (Shape: Total_Nodes x Total_Nodes)
    graph_sparse = coo_matrix(
        (all_data, (all_rows, all_cols)), 
        shape=(n_total, n_total)
    )

    # 3. Compute MST
    # Scipy is highly optimized for this
    mst_csr = minimum_spanning_tree(graph_sparse)
    
    # 4. Return Total Weight
    return mst_csr.data.sum()

class FiedlerValue:
	seed = 0

	@classmethod
	def using_networkx(cls, G):
		"""
			Takes too long more than 30 minutes but no results for single graph
		"""
		return nx.algebraic_connectivity(G, weight="weight",	 normalized=True, seed=G.seed)

	@classmethod
	def using_torch(cls, matrix, device="cuda"):
		# Better to use isinstance for type checking
		assert isinstance(matrix, np.ndarray), "Input must be a numpy array"
		
		n, m = matrix.shape
		
		# Convert input to tensor
		A = torch.tensor(matrix, device=device, dtype=torch.float32)

		# Make the matrix symmetric (bipartite adjacency matrix)
		# Shape becomes (n+m) * (n+m)
		# Structure:
		# [ 0  A ]
		# [ A.T 0 ]
		A_symmetric = torch.cat([
				torch.cat([torch.zeros((n, n), device=A.device), A], dim=1),
				torch.cat([A.T, torch.zeros((m, m), device=A.device)], dim=1),
		], dim=0)

		# 1. Calculate Degrees
		degrees = A_symmetric.sum(dim=1)

		# 2. Calculate D^(-1/2) (Inverse Square Root)
		# We use a mask to handle division by zero for isolated nodes
		d_inv_sqrt = torch.pow(degrees, -0.5)
		d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
		
		D_inv_sqrt = torch.diag(d_inv_sqrt)

		# 3. Calculate Normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
		normalized_adj = D_inv_sqrt @ A_symmetric @ D_inv_sqrt
		identity = torch.eye(A_symmetric.shape[0], device=A.device)
		
		normalized_laplacian = identity - normalized_adj

		# 4. Eigen decomposition
		# eigvalsh is correct here because the Normalized Laplacian is symmetric
		eigenvalues = torch.linalg.eigvalsh(normalized_laplacian)
		
		# The Fiedler value is the second smallest eigenvalue (index 1)
		# Note: Ensure graph is connected, otherwise this might be 0.
		fiedler_value = eigenvalues[1].item()
		
		return fiedler_value

attention_matrices_path = "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_train"
metadata_path = "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_train/metadata.pkl"

with open(metadata_path, "rb") as fp:
	metadata = pickle.load(fp)

# computing the median statistics
n_answer_tokens = np.array([np.load(m["visual_subgraphs_file"])["layer_0_head_0"].shape[0] for m in metadata]).mean()
n_query_tokens = np.array([np.load(m["text_subgraphs_file"])["layer_0_head_0"].shape[1] for m in metadata]).mean()
n_visual_tokens = np.array([np.load(m["visual_subgraphs_file"])["layer_0_head_0"].shape[1] for m in metadata]).mean()
print(f"Number of answer tokens = {n_answer_tokens}")
print(f"Number of query tokens = {n_query_tokens}")
print(f"Number of visual tokens = {n_visual_tokens}")

n_answer_tokens = []

for i, meta in enumerate(metadata):
	print(f"\tExtracting fiedler value for {i+1}th sample")
	visual_path, text_path = meta["visual_subgraphs_file"], meta["text_subgraphs_file"]
	
	# contains all the text and visual attention submatrices for all layers and heads
	text_matrices, visual_matrices = np.load(text_path), np.load(visual_path)

	# stores the fiedler value with key `layer_<layer>_head_<head>`
	fiedler_value_dict = dict()

	n_layers, n_heads = 32, 32
	for layer in range(n_layers):
		for head in range(n_heads):
			key = f"layer_{layer}_head_{head}"
			text_matrix, visual_matrix = text_matrices[key], visual_matrices[key]

			# float 32 to prevent the scipy errors
			text_matrix, visual_matrix = text_matrix.astype(np.float32), visual_matrix.astype(np.float32)
			
			# extracting the fiedler value for text and visual
			# text_fiedler, visual_fiedler = FiedlerValue.using_torch(text_matrix), FiedlerValue.using_torch(visual_matrix)
			text_fiedler, visual_fiedler = get_mst_total_weight(text_matrix), get_mst_total_weight(visual_matrix)

			fiedler_value_dict[key] = {
				"text": text_fiedler,
				"visual": visual_fiedler,
			}
		
	metadata[i]["fiedlers"] = fiedler_value_dict

with open("./outputs/fiedler_values_train_mst.pkl", "wb") as fp:
	pickle.dump(metadata, fp)