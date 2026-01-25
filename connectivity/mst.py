import numpy as np
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Dict, Any
import torch

def compute_bipartite_mst(mat: np.ndarray) -> Dict[str, Any]:
		A = np.asarray(mat, dtype=np.float32)
		num_answer_tokens, num_visual_tokens = A.shape
		G = nx.Graph()
		G.add_nodes_from(range(num_answer_tokens), bipartite=0)
		G.add_nodes_from(
				range(num_answer_tokens, num_answer_tokens + num_visual_tokens),
				bipartite=1,
		)
		for i in range(num_answer_tokens):
				for j in range(num_visual_tokens):
						G.add_edge(
								i,
								num_answer_tokens + j,
								weight=float(1.0-A[i, j]),
						)
		mst = nx.minimum_spanning_tree(G, weight="weight")
		total_weight = sum(d["weight"] for _, _, d in mst.edges(data=True))

		return {
				"num_answer_tokens": num_answer_tokens,
				"num_visual_tokens": num_visual_tokens,
				"num_nodes": mst.number_of_nodes(),
				"num_edges": mst.number_of_edges(),
				"total_weight": total_weight,
				"normalized_weight": (
						total_weight / num_answer_tokens if num_answer_tokens > 0 else 0.0
				),
				"edges": [
						{"u": u, "v": v, "weight": d["weight"]}
						for u, v, d in mst.edges(data=True)
				],
		}

def get_mst_normalized_wt(mat):
	return {
		"normalized_mst": compute_bipartite_mst(mat)["normalized_weight"],
	}