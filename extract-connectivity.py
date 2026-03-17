import pickle
import json
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
parser.add_argument("--mode", type=str, choices=[
	"decomposed",
	"full"
], default="decomposed")
parser.add_argument("--output", "-o", help="output path for pickle file, including the filename and extension")
parser.add_argument("--n-workers", type=int, help="Number of workers")

# parsing the user input
args = parser.parse_args()

metadata_path = args.metadata_path
output_path = args.output
mode = args.mode
num_workers = args.n_workers or os.cpu_count()-1
connectivity_id = args.connectivity
connectivity = get_connectivity_function(connectivity_id)

with open(metadata_path, "rb") as fp:
	if metadata_path.endswith(".pkl"):
		metadata = pickle.load(fp)
	elif metadata_path.endswith(".json"):
		metadata = json.load(fp)
	else:
		raise Exception("Unsupported file format encountered")

print(f"Total samples number of samples = {len(metadata)}\n\n")

def process_sample(args):
	i, meta, mode = args

	text_matrices, visual_matrices = None, None
	attention_matrix = None

	if mode == "decomposed":
		# contains all the text and visual attention submatrices for all layers and heads
		visual_path, text_path = meta["visual_subgraphs_file"], meta["text_subgraphs_file"]
		text_matrices, visual_matrices = np.load(text_path), np.load(visual_path)
	elif mode == "full":
		if "attnetion_file" in meta:
			attention_path = meta["attention_file"]
		elif "attn_save_path" in meta:
			attention_path = meta["attn_save_path"]
		else:
			raise Exception(f"No attention file key exists in meta, got {meta.keys()}")
		
		attention_npx = np.load(attention_path)
		if "arr_0" in attention_npx:
			attention_matrix = attention_npx["arr_0"]
		elif "full_attn" in attention_npx:
			attention_matrix = attention_npx["full_attn"]
		
		attention_matrix = attention_matrix.astype(np.float32)
	
	# stores the connectivity value with key `layer_<layer>_head_<head>`
	connectivity_value_dict = dict()

	# parameter for the different computations
	params = {
		"prompt_len": meta.get("prompt_len", None),
	}

	n_layers, n_heads = 34, 34
	for layer in range(n_layers):
		for head in range(n_heads):
			try:
				key = f"layer_{layer}_head_{head}"

				if text_matrices or visual_matrices:
					text_matrix, visual_matrix = text_matrices[key], visual_matrices[key]
					# float 32 to prevent the scipy errors
					text_matrix, visual_matrix = text_matrix.astype(np.float32), visual_matrix.astype(np.float32)
					
					# extracting the connectivity value for text and visual
					text_connectivity, visual_connectivity = connectivity(text_matrix, **params), connectivity(visual_matrix, **params)

					connectivity_value_dict[key] = {
						"text": text_connectivity,
						"visual": visual_connectivity,
					}

				if attention_matrix is not None:
					connectivity_value_dict[key] = {
						"atttention": connectivity(attention_matrix[layer][head].astype(np.float32), **params)
					}	
			except:
				continue
		
	return i, connectivity_value_dict

print(f"Number of workers = {num_workers}")
tasks = [(i, meta, mode) for i, meta in enumerate(metadata)]

with ProcessPoolExecutor(max_workers=num_workers) as executor:
	futures = [executor.submit(process_sample, t) for t in tasks]

	for fut in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
		i, connectivity_dict = fut.result()
		metadata[i]["connectivity"] = connectivity_dict

with open(output_path, "wb") as fp:
	pickle.dump(metadata, fp)
