import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# multiprocessing boost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def compute_gini(values):
    sorted_values = np.sort(values.flatten())
    n = len(sorted_values)

    # Handle edge cases
    if n == 0 or np.sum(sorted_values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def process_sample(args):
    i, meta, mode = args

    text_matrices, visual_matrices = None, None
    attention_matrix = None

    if mode == "decomposed":
        # contains all the text and visual attention submatrices for all layers and heads
        visual_path, text_path = meta["visual_subgraphs_file"], meta["text_subgraphs_file"]
        text_matrices, visual_matrices = np.load(text_path), np.load(visual_path)
    elif mode == "full":
        attention_path = meta["attention_file"]
        attention_matrix = np.load(attention_path)["arr_0"].astype(np.float32)

    # stores the connectivity value with key `layer_<layer>_head_<head>`
    connectivity_value_dict = {}

    n_layers, n_heads = 34, 34
    # compute GINI for each attention head across layers
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                key = f"layer_{layer}_head_{head}"

                if text_matrices or visual_matrices:
                    text_matrix, visual_matrix = text_matrices[key], visual_matrices[key]

                    # float 32 to prevent the scipy errors
                    text_matrix, visual_matrix = (
                        text_matrix.astype(np.float32),
                        visual_matrix.astype(np.float32),
                    )

                    # Compute connectivity using all text and visual tokens.
                    text_gini, visual_gini = compute_gini(text_matrix), compute_gini(visual_matrix)

                    connectivity_value_dict[key] = {
                        "text": text_gini,
                        "visual": visual_gini,
                    }

                if attention_matrix is not None:
                    connectivity_value_dict[key] = {
                        "features": compute_gini(attention_matrix[layer][head].astype(np.float32))
                    }
            except Exception:
                continue

    return i, connectivity_value_dict


def process_metadata_pickle(metadata, mode, n_workers):
    tasks = [(i, meta, mode) for i, meta in enumerate(metadata)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_sample, t) for t in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
            i, connectivity_dict = fut.result()
            metadata[i]["connectivity"] = connectivity_dict

    return metadata


def metadata_to_dataset(metadata):
    X = np.array(
        [
            [value for v in meta["connectivity"].values() for value in v.values()]
            for meta in metadata
        ]
    )
    y = np.array([meta["is_hallucinating"] for meta in metadata])

    return X, y


def main(train_metadata, test_metadata, mode, n_workers):
    # compute the Gini value for train and test metadata
    train_metadata_with_gini = process_metadata_pickle(
        train_metadata, mode=mode, n_workers=n_workers
    )
    test_metadata_with_gini = process_metadata_pickle(
        test_metadata, mode=mode, n_workers=n_workers
    )

    # create the dataset
    train_X, train_y = metadata_to_dataset(train_metadata_with_gini)
    test_X, test_y = metadata_to_dataset(test_metadata_with_gini)
    print(f"Shape of X = {train_X.shape} and y = {train_y.shape} for train")
    print(f"Shape of X = {test_X.shape} and y = {test_y.shape} for test")

    # logistic regression using scikit-learn
    model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=0)
    print(model)

    model.fit(train_X, train_y)

    # evaluate on test set
    test_prob = model.predict_proba(test_X)
    auroc_score = roc_auc_score(test_y, test_prob[:, 1])
    aupr_score = average_precision_score(test_y, test_prob[:, 1])

    print(f"AUROC = {auroc_score} and AUPR = {aupr_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts Gini value, trains model, and return evaluation metrics"
    )
    parser.add_argument(
        "--train", help="train metadata.pkl path obtained from GREEN", required=True
    )
    parser.add_argument("--test", help="test metadata.pkl path obtained from GREEN", required=True)
    parser.add_argument("--mode", type=str, choices=["decomposed", "full"], default="decomposed")
    parser.add_argument(
        "--n-workers", type=int, help="Number of workers", default=os.cpu_count() - 1
    )

    # parsing the user input
    args = parser.parse_args()

    # loading the metadata file
    with open(args.train, "rb") as file:
        train_metadata = pickle.load(file)

    with open(args.test, "rb") as file:
        test_metadata = pickle.load(file)

    print(
        f"Loaded {len(train_metadata)} and {len(test_metadata)} samples for train and test respectively."
    )

    main(
        train_metadata=train_metadata,
        test_metadata=test_metadata,
        mode=args.mode,
        n_workers=args.n_workers,
    )
