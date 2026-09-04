import os
import pickle as pkl
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# multiprocessing boost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_visual_embeddings(embeddings, visual_indices):
    """
    embeddings = (L+1, s, 4096) where L = number of layers, L+1 because the first one contains the initial embedding

    output = (L, 576, 4096) where 576 is number of visual tokens
    """
    # initial_embeddings = embeddings[0]
    embeddings = embeddings[1:]

    # isolating the visual embeddings only
    visual_embeddings = embeddings[:, visual_indices, :]
    return visual_embeddings.astype(np.float32)


def process_item(args):
    i, item = args

    embedding_file, visual_indices = item["embedding_file"], item["visual_indices"]
    try:
      embedding = np.load(embedding_file)["arr_0"]
    except Exception as e:
      raise RuntimeError(f"Fialed to load embedding file: {embedding_file}") from e

    visual_embeddings = get_visual_embeddings(embedding, visual_indices)

    # compute the L2 norm for each token embeddings (SHIFT REGSITERS PAPER)
    norms_across_layers = np.linalg.norm(visual_embeddings, 2, axis=2)  # (32, 576)

    # thresholding the norms for sink token masks
    sink_token_masks = norms_across_layers > 150
    sink_tokens_indices = [list(np.where(row)[0]) for row in sink_token_masks]

    return i, sink_tokens_indices


def main(metadata, output_path, n_workers):
    tasks = [(i, item) for i, item in enumerate(metadata)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_item, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Items"):
            i, sink_token_indices = fut.result()
            metadata[i]["sink_tokens_indices"] = sink_token_indices

    with open(output_path, "wb") as file:
        pkl.dump(metadata, file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Computes attention sink for visual tokens and appends it into metadata.pkl"
    )
    parser.add_argument("--input", "-i", required=True, help="Input metadata.pkl output from green")
    parser.add_argument(
        "--output",
        help="output pickle path if not then input pickle will be overwritten",
        default=None,
    )
    parser.add_argument("--n-workers", type=int, default=4)

    args = parser.parse_args()

    # reading the metadata file
    with open(args.input, "rb") as fp:
        metadata = pkl.load(fp)

    main(
        metadata=metadata,
        output_path=args.output if args.output is not None else args.input,
        n_workers=args.n_workers,
    )
