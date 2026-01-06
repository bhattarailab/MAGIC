import pickle
import numpy as np

from graph_utils import create_graph, draw_graph
import networkx as nx

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
	text_matrix, visual_matrix = text_matrix.astype(np.float32), visual_matrix.astype(np.float32)
	
	return text_matrix, visual_matrix	

sample_index = 0
layer, head = 0, 0

text_matrix, visual_matrix = load_attention_matrices(sample_index, layer, head)
text_graph, visual_graph = create_graph(text_matrix, "text"), create_graph(visual_matrix, "visual")

# text_feidler_vector, visual_fiedler_vector = nx.fiedler_vector(text_graph, normalized=True, seed=0), nx.fielder_vector(visual_graph, normalized=True, seed=0)

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
		assert(type(matrix) == np.ndarray)
		return 0.0

print(FiedlerValue.using_torch(text_matrix))