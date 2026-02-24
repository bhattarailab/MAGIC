import torch
import os
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description="performs logistic regression on the given csv file")
parser.add_argument("--csv", help="CSV file created using create-dataset.py")
parser.add_argument("--breakpoint", "-b", help="have breakpoint after execution better for testing")

args = parser.parse_args()
csv_path = args.csv
has_breakpoint = args.breakpoint

SEEDS = [42, 21, 23, 12, 123]

def set_seed(seed: int, deterministic: bool = True) -> None:
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

class ConnectivityVectorDataset(torch.utils.data.Dataset):
	def __init__(self, csv_path, split="train"):
		super().__init__()
		df = pd.read_csv(csv_path)
		split_df = df[df["split"] == split]

		self.feature_columns = [s for s in split_df.columns if s.startswith("layer_")]
		self.X = split_df[self.feature_columns].to_numpy()
		self.y = split_df["label"].to_numpy()

	def __len__(self):
		return len(self.y)

	def __getitem__(self, index):
		return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(1.0 if self.y[index] else 0.0, dtype=torch.float32).reshape((1,))


def dataset_to_numpy(dataset):
	ret_x, ret_y = [], []
	for X, y in dataset:
		ret_x.append(X)
		ret_y.append(y.item())

	return np.array(ret_x), np.array(ret_y)

all_auroc = []
all_aupr = []

for seed in SEEDS:
	set_seed(seed)

	# intialziign train and test dataset from the pickles path
	train_set = ConnectivityVectorDataset(csv_path, split="train")
	test_set = ConnectivityVectorDataset(csv_path, split="test")

	# feature size controls the weight parameter for the model
	n_features = len(train_set.feature_columns)

	# creating dataloader with batching
	g = torch.Generator()
	g.manual_seed(seed)
	train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, generator=g)
	test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, generator=g)

	# initializing the model
	model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
	# for debugging
	print(model)


	# creating the tran and test set as per the requirement of sklearn
	sklearn_train_X, sklearn_train_y = dataset_to_numpy(train_set)
	sklearn_test_X, sklearn_test_y = dataset_to_numpy(test_set)

	model.fit(sklearn_train_X, sklearn_train_y)
	test_prob = model.predict_proba(sklearn_test_X)

	auroc_score = roc_auc_score(sklearn_test_y, test_prob[:, 1])
	aupr_score = average_precision_score(sklearn_test_y, test_prob[:, 1])

	# coefficinets = model.coef_[0]
	# absolute_coefficients = np.abs(coefficinets)
	# descending_indices = np.argsort(absolute_coefficients)[::-1]
	# top_features = np.array(train_set.feature_columns)[descending_indices]
	# features_df = pd.DataFrame({
	# 	"key": train_set.feature_columns,
	# 	"coeffecients": coefficinets,
	# })
	# features_df.to_csv("./outputs/importances.csv")

	print(f"For seed = {seed}")
	print(f"\tTest AUROC = {auroc_score}")
	print(f"\tTest AUPR = {aupr_score}\n")

	all_aupr.append(aupr_score)
	all_auroc.append(auroc_score)

all_auroc, all_aupr = np.array(all_auroc), np.array(all_aupr)
print(f"Mean auroc: {np.mean(all_auroc):.4f} with std. deviation {np.std(all_auroc):.4f}")
print(f"Mean aupr: {np.mean(all_aupr):.4f} with std. deviation {np.std(all_aupr):.4f}")

if has_breakpoint:
	breakpoint()
