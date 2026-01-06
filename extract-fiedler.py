import pickle
import numpy as np

text_attentions_path = "./attentions-pkl/llava-med-v1.5-mistral-7b_text_features_cache_valid.pkl"
visual_attentions_path = "./attentions-pkl/llava-med-v1.5-mistral-7b_visual_features_cache_valid.pkl"

with open(text_attentions_path, "rb") as fp:
	text_attentions = pickle.load(fp)

with open(visual_attentions_path, "rb") as fp:
	visual_attentions = pickle.load(fp)

# visual_attentions is a tuple with hallucinating and non_hallucinating features
hallucinating_visual_attentions, non_hallucinating_visual_attentions = visual_attentions
hallucinating_text_attentions, non_hallucinating_text_attentions = text_attentions

assert(len(hallucinating_text_attentions) == len(hallucinating_visual_attentions))
assert(len(non_hallucinating_text_attentions) == len(non_hallucinating_visual_attentions))

print(f"Loaded {len(hallucinating_visual_attentions)} hallucinating and {len(non_hallucinating_visual_attentions)} samples respectively")

index = 0
hallucinating_text_attention, hallucinating_visual_attention = hallucinating_text_attentions[index], hallucinating_visual_attentions[index]

breakpoint()
def process_head(mat, k=10):
    """Compute top k eigenvalues of Laplacian matrix."""
    A = np.array(mat).astype(np.float32)
    num_answer_tokens, num_visual_tokens = A.shape
    #k=2
    G = nx.Graph()
    G.add_nodes_from(range(num_answer_tokens), bipartite=0)
    G.add_nodes_from(range(num_answer_tokens, num_answer_tokens + num_visual_tokens), bipartite=1)
    for i in range(num_answer_tokens):
        for j in range(num_visual_tokens):
            weight = float(A[i, j])  
            G.add_edge(i, num_answer_tokens + j, weight=weight)
    
    L = nx.laplacian_matrix(G)
    
    try:
        eigenvalues = eigsh(L, k=min(k, L.shape[0]-1), which='LA', return_eigenvectors=False)
        return eigenvalues[::-1]
    except Exception as e:
        print(f"Error: {e}")
        print(f"Matrix: {mat.shape}")
        return 0
