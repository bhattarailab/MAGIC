import pickle
import numpy as np
import os
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, laplacian
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from itertools import repeat

def compute_bipartite_mst_fast(mat: np.ndarray) -> dict:
    """Uses SciPy for high-performance MST and PyTorch for Fiedler value."""
    n_a, n_v = mat.shape
    n_total = n_a + n_v
    weights = 1.0 - mat.astype(np.float64)
    full_adj = np.zeros((n_total, n_total))
    full_adj[:n_a, n_a:] = weights
    full_adj[n_a:, :n_a] = weights.T
    mst_sparse = minimum_spanning_tree(csr_matrix(full_adj))
    total_weight = mst_sparse.sum()
    mst_sym = mst_sparse + mst_sparse.T
    L_sparse = laplacian(mst_sym, normed=True)
    if n_total > 1:
        L_torch = torch.from_numpy(L_sparse.toarray())
        try:
            eigvals = torch.linalg.eigh(L_torch).eigenvalues
            fiedler_value = eigvals[1].item()
        except Exception:
            fiedler_value = 0.0
    else:
        fiedler_value = 0.0

    return {
        "num_answer_tokens": n_a,
        "num_visual_tokens": n_v,
        "total_weight": float(total_weight),
        "fiedler_value": float(max(0, fiedler_value)), 
    }

def process_single_entry(entry: dict, root_path: str) -> dict:
    """Worker function to process image-text attention subgraphs."""
    try:
        v_path=entry["visual_subgraphs_file"]
        t_path=entry["text_subgraphs_file"]
        
        visual_data = np.load(v_path, mmap_mode='r')
        text_data = np.load(t_path, mmap_mode='r')

        entry["connectivity"] = {}
        all_keys = sorted(set(visual_data.files) | set(text_data.files))

        for key in all_keys:
            entry["connectivity"][key] = {"visual": {}, "text": {}}
            if key in visual_data.files:
                entry["connectivity"][key]["visual"] = compute_bipartite_mst_fast(visual_data[key])
            if key in text_data.files:
                entry["connectivity"][key]["text"] = compute_bipartite_mst_fast(text_data[key])
                
        return entry
    except Exception as e:
        print(f"Error processing entry: {e}")
        return entry

def run_processing_pipeline(input_path: str, output_path: str, max_workers: int):
    """Loads a split, processes it in parallel, and saves it."""
    print(f"\n--- Processing: {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} not found.")
        return
    root_path = os.path.dirname(input_path)
    with open(input_path, "rb") as fp:
        data = pickle.load(fp)
    chunksize = max(1, 4)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        processed_data = list(tqdm(
            executor.map(process_single_entry, data, repeat(root_path), chunksize=chunksize), 
            total=len(data), 
            desc="Computing MSTs"
        ))

    with open(output_path, "wb") as fp:
        pickle.dump(processed_data, fp)
    print(f"Saved results to: {output_path}")

if __name__ == "__main__":
    TRAIN_IN = "/work/FAC/HEC/DESI/yshresth/aim/noli1/green/outputs/train_all_radvqa.pkl"
    TEST_IN = "/work/FAC/HEC/DESI/yshresth/aim/noli1/green/outputs/test_all_radvqa.pkl"
    TRAIN_OUT = "./outputs/feidlermst_all_train.pkl"
    TEST_OUT = "./outputs/feidlermst_all_test.pkl"

    N_CORES = 4
    torch.set_num_threads(1)
    print(f"Starting pipeline using {N_CORES} cores...")
    run_processing_pipeline(TRAIN_IN, TRAIN_OUT, max_workers=N_CORES)
    run_processing_pipeline(TEST_IN, TEST_OUT, max_workers=N_CORES)
    print("\nAll processing complete.")