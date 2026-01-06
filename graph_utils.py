import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_graph(G):
	nx.draw(G, with_labels=True, font_weight="bold")
	plt.savefig("./graph.png")

def create_graph(matrix, type="vision", draw=False):
	"""
		Creates graph from the given attention matrix. The type is used to change color for visual representation of the graph.
	"""
	# TODO: create a graph between answer and answer tokens too
	n_answer_tokens, n_type_tokens = matrix.shape
	answer_color, type_color = "blue", "green" if type == "visual" else "red"

	# initializing the graph
	G = nx.Graph()

	# creating the nodes for answer tokens
	G.add_nodes_from([(i, {"color": answer_color}) for i in range(n_answer_tokens)])
	
	# creating the nodes for type tokens
	G.add_nodes_from([(i, {"color": type_color}) for i in range(n_answer_tokens, n_answer_tokens+n_type_tokens)])

	# assigning the edges between answer tokens and type tokens
	for a in range(n_answer_tokens):
		a_lists = np.repeat(a, n_type_tokens)
		edges = zip(a_lists, range(n_type_tokens), matrix[a])
		G.add_weighted_edges_from(edges)

	if draw:
		draw_graph(G)

	breakpoint()

	return G
