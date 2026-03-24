import pandas as pd
import pickle
import argparse

ALl_HEADS_KEYS = [f"layer_{l}_head_{h}" for l in range(32) for h in range(32)]

parser = argparse.ArgumentParser(description="Script to create the dataset for logistic regression task")

parser.add_argument("--breakpoint", "-b", help="breakpoint at end for debugging", action="store_true")
parser.add_argument("--pickle", nargs=2, help="creates dataset using the connectivity pickle i.e. output of extract connectivity, two arguments should be provided first one is train second is test")
parser.add_argument("--name", default="dataset", help="name for the dataset optional", required=True)
parser.add_argument("--connectivity", default="lookback", choices=["lookback", "mtopdiv"], help="connectivity measures")

args = parser.parse_args()
has_breakpoint = args.breakpoint
connectivity_pickle = args.pickle
connectivity = args.connectivity
name = args.name

train_pickle, test_pickle = connectivity_pickle

# printing the data for logs
print(f"Train Pickle = {train_pickle}")
print(f"Test Pickle = {test_pickle}")

# loading from the pickle
with open(train_pickle, "rb") as fp:
	train_data = pickle.load(fp)
with open(test_pickle, "rb") as fp:
	test_data = pickle.load(fp)

# printing for the logs
print(f"Loaded train data with {len(train_data)} samples")
print(f"Loaded test data with {len(test_data)} samples")
print(f"Connectivity being measured = {connectivity}")

connectivity_list = []
for split, split_sample in zip(["train", "test"], [train_data, test_data]):
	for sample in split_sample:
		connectivity_sample = {}
		for l in range(34):
			for h in range(34):
				try:
					key = f"layer_{l}_head_{h}"
					connectivity_sample[key] = sample["features"][f"layer_{l}_head_{h}"][connectivity]
				except:
					continue

		connectivity_sample["label"] = sample["is_hallucinating"]
		connectivity_sample["split"] = split
		connectivity_list.append(connectivity_sample)

df = pd.DataFrame(connectivity_list)
df.to_csv(f"outputs/{name}.csv", index=False)

if has_breakpoint:
	breakpoint()