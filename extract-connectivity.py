import pickle
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# multiprocessing boost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from connectivity import get_connectivity_function

parser = argparse.ArgumentParser(description="Script to extract the connectivity value across all the heads of a VLM")
parser.add_argument("--metadata-path", "-m", help="Path to the metadata pickle file")
parser.add_argument("--connectivity", "-c", help="Type of algebraic connectivity measure to be used. See the readme.id to know the id for each connectivity measure")
parser.add_argument("--output", "-o", help="output path for pickle file, including the filename and extension")

# parsing the user input
args = parser.parse_args()

metadata_path = args.metadata_path
output_path = args.output
connectivity_id = args.connectivity
connectivity = get_connectivity_function(connectivity_id)

with open(metadata_path, "rb") as fp:
	metadata = pickle.load(fp)

print(f"Total samples number of samples = {len(metadata)}\n\n")

# computing the median statistics
# n_answer_tokens = np.array([np.load(m["visual_subgraphs_file"])["layer_0_head_0"].shape[0] for m in metadata]).mean()
# n_query_tokens = np.array([np.load(m["text_subgraphs_file"])["layer_0_head_0"].shape[1] for m in metadata]).mean()
# n_visual_tokens = np.array([np.load(m["visual_subgraphs_file"])["layer_0_head_0"].shape[1] for m in metadata]).mean()
# print(f"\tNumber of answer tokens = {n_answer_tokens}")
# print(f"\tNumber of query tokens = {n_query_tokens}")
# print(f"\tNumber of visual tokens = {n_visual_tokens}\n")

def process_sample(args):
	i, meta = args

	visual_path, text_path = meta["visual_subgraphs_file"], meta["text_subgraphs_file"]
	
	# contains all the text and visual attention submatrices for all layers and heads
	text_matrices, visual_matrices = np.load(text_path), np.load(visual_path)

	# stores the connectivity value with key `layer_<layer>_head_<head>`
	connectivity_value_dict = dict()

	n_layers, n_heads = 32, 32
	for layer in range(n_layers):
		for head in range(n_heads):
			key = f"layer_{layer}_head_{head}"
			text_matrix, visual_matrix = text_matrices[key], visual_matrices[key]

			# float 32 to prevent the scipy errors
			text_matrix, visual_matrix = text_matrix.astype(np.float32), visual_matrix.astype(np.float32)
			
			# extracting the connectivity value for text and visual
			text_connectivity, visual_connectivity = connectivity(text_matrix), connectivity(visual_matrix)

			connectivity_value_dict[key] = {
				"text": text_connectivity,
				"visual": visual_connectivity,
			}
		
	return i, connectivity_value_dict

num_workers = os.cpu_count()-1
print(f"Number of workers = {num_workers}")
tasks = [(i, meta) for i, meta in enumerate(metadata)]
with ProcessPoolExecutor(max_workers=num_workers) as executor:
	futures = [executor.submit(process_sample, t) for t in tasks]

	for fut in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
		i, connectivity_dict = fut.result()
		metadata[i]["connectivity"] = connectivity_dict

with open(output_path, "wb") as fp:
	pickle.dump(metadata, fp)