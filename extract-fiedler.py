import pickle
import numpy as np

from graph_utils import create_graph

attention_matrices_path = "/scratch/noli1/graph_hallucination_detection/results/llava-med-v1.5-mistral-7b_test"

def load_attention_matrices(sample: int, layer: int, head: int) -> (np.array, np.array):
	"""
		Loads and returns the attention matrices from the .npz files;
			- text subgraph = text_subgraphs_<sample index>.npz
			- visual subgraph = visual_subgraphs_<sample index>.npz

		Returns, (np.array, np.array) => (visual, text)
	"""
	text_path = f"{attention_matrices_path}/text_subgraphs_{sample}.npz"
	visual_path = f"{attention_matrices_path}/visual_subgraphs_{sample}.npz"

	text_matrices, visual_matrices = np.load(text_path), np.load(visual_path)

	key = f"layer_{layer}_head_{head}"
	text_matrix, visual_matrix = text_matrices[key], visual_matrices[key]
	
	return text_matrix, visual_matrix	

sample_index = 0
layer, head = 0, 0

text_matrix, visual_matrix = load_attention_matrices(sample_index, layer, head)
text_graph, visual_graph = create_graph(text_matrix, "text"), create_graph(visual_matrix, "visual")
