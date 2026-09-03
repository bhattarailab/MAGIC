import pickle as pkl
from argparse import ArgumentParser

import numpy as np


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


def main(metadata, output_path):
    for item in metadata:
        embedding_file, visual_indices = item["embedding_file"], item["visual_indices"]
        embedding = np.load(embedding_file)["arr_0"]

        visual_embeddings = get_visual_embeddings(embedding, visual_indices)

        # compute the L2 norm for each token embeddings (SHIFT REGSITERS PAPER)
        norms_across_layers = np.linalg.norm(visual_embeddings, 2, axis=2)  # (32, 576)

        # thresholding the norms for sink token masks
        sink_token_masks = norms_across_layers > 150
        sink_tokens_indices = [list(np.where(row)[0]) for row in sink_token_masks]
        item["sink_tokens_indices"] = sink_tokens_indices,

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

    args = parser.parse_args()

    # reading the metadata file
    with open(args.input, "rb") as fp:
        metadata = pkl.load(fp)

    main(metadata=metadata, output_path=args.output if args.output is not None else args.input)
