import pandas as pd
import pickle
import json
import argparse
import os

ALl_HEADS_KEYS = [f"layer_{l}_head_{h}" for l in range(32) for h in range(32)]

parser = argparse.ArgumentParser(description="Script to create the dataset for logistic regression task")

# if heads log file is provided it does the following things;
# 1. ignores the samples that were used during the training
# 2. creates metrics for features for selected heads only, the metrics will be same mentioned in heads log
parser.add_argument("--heads-log", help="path to the log file used in head selection")
parser.add_argument("--all-heads", action="store_true", help="take all the heads for features")
parser.add_argument("--fuse", help="one of addition, division", default=None)
parser.add_argument("--breakpoint", "-b", help="breakpoint at end for debugging", action="store_true")
parser.add_argument("--pickle", nargs=2, help="creates dataset using the connectivity pickle i.e. output of extract connectivity, two arguments should be provided first one is train second is test")
parser.add_argument("--name", default="dataset", help="name for the dataset optional")
parser.add_argument("--connectivity", default=["gini"], nargs="+", help="connectivity measures")
parser.add_argument("--mode", type=str, choices=[
	"decomposed",
	"full"
], default="decomposed")

args = parser.parse_args()
heads_log_path = args.heads_log
all_heads, fusion = args.all_heads, args.fuse
has_breakpoint = args.breakpoint
connectivity_pickle = args.pickle
connectivities = args.connectivity
name = args.name
mode = args.mode

if connectivity_pickle:
	train_pickle, test_pickle = connectivity_pickle

	# loading from the pickle
	with open(train_pickle, "rb") as fp:
		train_data = pickle.load(fp)
	with open(test_pickle, "rb") as fp:
		test_data = pickle.load(fp)

	# extracting the connectivity for both the files
	connectivity_list = []
	for split, split_sample in zip(["train", "test"], [train_data, test_data]):
		for sample in split_sample:
			connectivity_sample = {}
			for l in range(32):
				for h in range(32):
					for t in ["text", "visual"] if mode == "decomposed" else ["atttention"]:
						for c in connectivities:
							key = f"layer_{l}_head_{h}_{c}_{t}"
							connectivity_sample[key] = sample["connectivity"][f"layer_{l}_head_{h}"][t][c]

			connectivity_sample["label"] = sample["is_hallucinating"]
			connectivity_sample["split"] = split
			connectivity_list.append(connectivity_sample)

	df = pd.DataFrame(connectivity_list)
	df.to_csv(f"outputs/{name}.csv", index=False)

if heads_log_path:
	# opening the logs of head selection to know the parameter of new dataset
	with open(heads_log_path, "r") as fp:
		head_log = json.load(fp)

	# represents the validation set that was used for head selection
	used_train_samples = 0 if all_heads else head_log["n_train_samples"]["total"]
	connectivity_measure = head_log["connectivity_measure"]
	heads = ALl_HEADS_KEYS if all_heads else [f"layer_{l}_head_{h}" for l, h in head_log["selected_heads"]]
	train_path, test_path = head_log["files"]["train"], head_log["files"]["test"]
	visual_wt, text_wt = head_log["visual_weight"], head_log["text_weight"]
	name = head_log["name"]

	output_path = f"./outputs/datasets/{name}.csv"
	if os.path.exists(output_path):
		raise FileExistsError(f"Output file already exists at {output_path}")

	# extracting the features from train and test paths
	with open(train_path, "rb") as fp:
		train_samples = pickle.load(fp)

	with open(test_path, "rb") as fp:
		test_samples = pickle.load(fp)

	dataset_rows = []

	# for the train samples
	for i, s in enumerate(train_samples):
		row_dict = dict()

		for k in heads:
			if mode == "decomposed":
				text_feature = s["connectivity"][k]["text"][connectivity_measure]
				visual_feature = s["connectivity"][k]["visual"][connectivity_measure]

				if fusion:
					if fusion == "addition":
						row_dict[f"{k}_feature"] = visual_wt*visual_feature+text_wt*text_feature
					elif fusion == "division":
						row_dict[f"{k}_feature"] = text_feature/(text_feature+visual_feature)
					else:
						raise Exception(f"fuse can be one of addition or division, got {fusion}")
				else:
					row_dict[f"{k}_text"] = text_feature
					row_dict[f"{k}_visual"] = visual_feature
			else:
				row_dict[f"{k}_attention"] = s["connectivity"][k]["atttention"][connectivity_measure]

		row_dict["split"] = "validation" if i < used_train_samples else "train"
		row_dict["label"] = s["is_hallucinating"]

		dataset_rows.append(row_dict)

	# for the test samples
	for i, s in enumerate(test_samples):
		row_dict = dict()
		if mode == "decomposed":
			text_feature = s["connectivity"][k]["text"][connectivity_measure]
			visual_feature = s["connectivity"][k]["visual"][connectivity_measure]

			for k in heads:
				text_feature = s["connectivity"][k]["text"][connectivity_measure]
				visual_feature = s["connectivity"][k]["visual"][connectivity_measure]

				if fusion:
					if fusion == "addition":
						row_dict[f"{k}_feature"] = visual_wt*visual_feature+text_wt*text_feature
					elif fusion == "division":
						row_dict[f"{k}_feature"] = text_feature/(text_feature+visual_feature)
					else:
						raise Exception(f"fuse can be one of addition or division, got {fusion}")
				else:
					row_dict[f"{k}_text"] = text_feature
					row_dict[f"{k}_visual"] = visual_feature
			else:
				row_dict[f"{k}_attention"] = s["connectivity"][k]["atttention"][connectivity_measure]

		row_dict["split"] = "test"
		row_dict["label"] = s["is_hallucinating"]

		dataset_rows.append(row_dict)

	df = pd.DataFrame(dataset_rows)
	df.to_csv(output_path, index=False)

	if has_breakpoint:
		breakpoint()
