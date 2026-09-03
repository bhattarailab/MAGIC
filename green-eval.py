import argparse
import pickle

import pandas as pd
from green_score import GREEN

GREEN_MODEL_ID = "StanfordAIMI/GREEN-radllama2-7b"


def main(input_path, output_path):
    with open(input_path, "rb") as fp:
        eval_datas = pickle.load(fp)

    questions = [d["question"] for d in eval_datas]
    refs = [d["ground_truth"] for d in eval_datas]
    candidates = [d["prediction"] for d in eval_datas]

    refs = [f"{questions[i]} {refs[i]}" for i in range(len(refs))]
    candidates = [f"{questions[i]} {candidates[i]}" for i in range(len(refs))]

    green_scorer = GREEN(GREEN_MODEL_ID, output_dir="./outputs", compute_summary_stats=False)

    N = len(questions)
    green_evaluations = []
    # for each batch
    for i in range(N):
        refs_sample, candidates_sample = [refs[i]], [candidates[i]]
        output = green_scorer(refs_sample, candidates_sample)

        rows = output[4].to_dict("records")
        green_evaluations.extend(rows)
        print(f"Completed = {i}")

    df = pd.DataFrame(green_evaluations)

    df["question"] = questions
    df["green_hallucination"] = df["green_score"] < 1

    # change the eval data key in is_hallucinating and create a new JSON
    updated_eval_data = [
        {**d, "is_hallucinating": str(df.iloc[i]["green_hallucination"]) == "True"}
        for i, d in enumerate(eval_datas)
    ]

    with open(output_path, "wb") as fp:
        pickle.dump(updated_eval_data, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Green Evaluation")
    parser.add_argument(
        "-i", "--input", help="Input file with question, prediction and ground truth", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output filepath including the filename", required=True
    )

    args = parser.parse_args()

    main(input_path=args.input, output_path=args.output)
