import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def get_visual_embeddings(embeddings, visual_indices):
    """
    embeddinggs = (L, s, 4096) where L = number of layers
    """
    print(embeddings["arr_0"])
    breakpoint()


def main(metadata):
    for item in metadata:
        embedding_file, visual_indices = item["embedding_file"], item["visual_indices"]
        embedding = np.load(embedding_file)

        get_visual_embeddings(embedding, visual_indices)


if __name__ == "__main__":
    parser = ArgumentParser(description="Scripts to perform experiments on shift registers")
    parser.add_argument("--input", "-i", required=True, help="Input path")

    args = parser.parse_args()
    input_path = Path(args.input)

    with open(input_path/"metadata.pkl", "rb") as file:
        metadata = pkl.load(file)

    main(metadata)
