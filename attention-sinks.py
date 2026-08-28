import os
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns


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


def main(metadata):
    for item in metadata:
        embedding_file, visual_indices = item["embedding_file"], item["visual_indices"]
        embedding = np.load(embedding_file)["arr_0"]

        visual_embeddings = get_visual_embeddings(embedding, visual_indices)

        # compute the L2 norm for each token embeddings (SHIFT REGSITERS PAPER)
        norms_across_layers = np.linalg.norm(visual_embeddings, 2, axis=2)  # (32, 576)
        n_layers, n_visual_tokens = norms_across_layers.shape[0], norms_across_layers.shape[1]

        # scatter plot between the layer and norm of visual tokens
        sns.scatterplot(
            x=np.repeat(np.arange(n_layers), n_visual_tokens),
            y=norms_across_layers.flatten(),
        )
        plt.xlabel("Layers")
        plt.ylabel("Norms")
        plt.title("Visualization of visual tokens' norms across layers")
        plt.savefig("./outputs/shift-registers/scatter.png")
        plt.close()

        # heatmap to identify the position of visual sink tokens
        for i_layer in range(n_layers):
            # reshape the norms into square matrix
            norms_2d = norms_across_layers[i_layer].reshape(24, 24)
            sns.heatmap(norms_2d, square=True, cbar=True, xticklabels=False, yticklabels=False)
            plt.title(f"Visual token norms' heatmap for L={i_layer}")
            plt.savefig(f"./outputs/shift-registers/norm-heatmap-{i_layer}.png")
            plt.close()

        # visual attention map and similarity with sink tokens
        visual_attention_submatrix = np.load(item["visual_subgraphs_file"])
        middle_visual_submtrix = visual_attention_submatrix["layer_10_head_10"]    # (text tokens, visual tokens)
        mean_attention_matrix = middle_visual_submtrix.mean(axis=0)
        sink_token_masks = norms_across_layers[10] > 150

        # findng the average attention in sink and non-sink tokens
        sink_tokens_attention = mean_attention_matrix[sink_token_masks].mean()
        non_sink_tokens_attention = mean_attention_matrix[~sink_token_masks].mean()
        print(f"Sink tokens = {sink_tokens_attention}, non-sink = {non_sink_tokens_attention}")

        # plot the heatmap of attention matrix
        _, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(mean_attention_matrix.reshape(24, 24), square=True, cbar=True, xticklabels=False, yticklabels=False, ax=ax)

        for row in range(24):
            for col in range(24):
                if sink_token_masks.reshape(24, 24)[row, col]:
                    ax.add_patch(Rectangle(
                        (col, row), 1, 1,
                        fill=False, edgecolor='blue', linewidth=2
                    ))
        plt.title("Mean attention heatmap for visual tokens for output tokens")
        plt.savefig("./outputs/shift-registers/attention-heatmap.png")
        plt.close()
        breakpoint()


if __name__ == "__main__":
    # create the output directories if not exists
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./outputs/shift-registers", exist_ok=True)

    parser = ArgumentParser(description="Scripts to perform experiments on shift registers")
    parser.add_argument("--input", "-i", required=True, help="Input path")

    args = parser.parse_args()
    input_path = Path(args.input)

    with open(input_path / "metadata.pkl", "rb") as file:
        metadata = pkl.load(file)

    main(metadata)
