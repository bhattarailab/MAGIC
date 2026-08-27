import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _save_single_result(result, idx, output_path):
    """Helper to save a single result to disk"""

    return {
        "question": result["question"],
        "ground_truth": result["ground_truth"],
        "prediction": result["prediction"],
        "is_hallucinating": result["is_hallucinating"],
        "attention_file": result["attention_file"],
        "visual_subgraphs_file": result["visual_subgraphs"],
        "text_subgraphs_file": result["text_subgraphs"],
        "embedding_file": result["embedding_file"],
        "visual_indices": result["visual_indices"],
        "text_indices": result["text_indices"],
        "answer_indices": result["answer_indices"],
        "sequence": result["sequence"],
    }


def load_metadata_only(metadata_file):
    """
    Load only metadata without loading large tensor files
    Useful for quick analysis of predictions and accuracy

    Args:
        output_dir: Directory containing saved results

    Returns:
        List of metadata dictionaries (without tensors)
    """

    with open(os.path.join(metadata_file), "rb") as f:
        metadata = pickle.load(f)
    breakpoint()
    # breakpoint()
    return metadata


def load_results(metadata_file):
    metadata = load_metadata_only(metadata_file)
    # breakpoint()
    information_text = {
        "hallucinations": [],
        "not_hallucinations": [],
    }
    information_imgae = {
        "hallucinations": [],
        "not_hallucinations": [],
    }
    for entry in metadata:
        visual_file = Path(entry["visual_subgraphs_file"])
        text_file = Path(entry["text_subgraphs_file"])
        visual_data = np.load(visual_file)
        text_data = np.load(text_file)
        print("Entry:", entry["is_hallucinating"])
        visual_subgraphs = {}
        text_subgraphs = {}
        is_hallucinating = entry["is_hallucinating"]
        for key in visual_data.files:
            layer_idx, head_idx = map(int, key.replace("layer_", "").split("_head_"))
            if layer_idx not in visual_subgraphs:
                visual_subgraphs[layer_idx] = []
            visual_subgraphs[layer_idx].append(visual_data[key])

        for key in text_data.files:
            layer_idx, head_idx = map(int, key.replace("layer_", "").split("_head_"))
            if layer_idx not in text_subgraphs:
                text_subgraphs[layer_idx] = []
            text_subgraphs[layer_idx].append(text_data[key])
        # breakpoint()
        if is_hallucinating:
            # breakpoint()
            information_text["hallucinations"].append(text_subgraphs)
            information_imgae["hallucinations"].append(visual_subgraphs)
        else:
            information_text["not_hallucinations"].append(text_subgraphs)
            information_imgae["not_hallucinations"].append(visual_subgraphs)
    return information_text, information_imgae


def save_results(
    results, output_dir="/scratch/sacharya/graph_hallucination_detection/results", num_workers=4
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_save_single_result, result, idx, output_path)
            for idx, result in enumerate(results)
        ]
        for future in as_completed(futures):
            metadata.append(future.result())

    with open(output_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Results saved to {output_path}")
    print(f"Total samples: {len(metadata)}")
