import torch
import numpy as np
import pandas as pd
import argparse
from models.simple_logistic import SimpleLogisticRegression
from models.linear_logistic import LinearLogisticRegression
from utils.model import compute_test_metrices
import seaborn as sns
import matplotlib.pyplot as plt
import re

parser = argparse.ArgumentParser(description="performs logistic regression on the given csv file")
parser.add_argument("--csv", help="CSV file created using create-dataset.py")
parser.add_argument("--device", help="device to run the model", default="cuda")
parser.add_argument("--epoch", help="Epoch for training the model", default=50, type=int)
parser.add_argument("--weight-path", help="Path where the weights of neuron is to be saved")
parser.add_argument("--heatmap-name", help="Store the weight in form of heatmap, input should be the name of heatmap")
parser.add_argument("--breakpoint", "-b", help="have breakpoint after execution better for testing")
parser.add_argument("--model", help="option to choose the model")

args = parser.parse_args()
csv_path, device = args.csv, args.device
epoch = args.epoch
weight_path, heatmap = args.weight_path, args.heatmap_name
has_breakpoint = args.breakpoint
model_type = args.model

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

# intialziign train and test dataset from the pickles path
train_set = ConnectivityVectorDataset(csv_path, split="train")
test_set = ConnectivityVectorDataset(csv_path, split="test")

# feature size controls the weight parameter for the model
n_features = len(train_set.feature_columns)

# creating dataloader with batching
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# initializing the model
if model_type == "linear":
	model = LinearLogisticRegression(n_features).to(device=device)
else:
	model = SimpleLogisticRegression(n_features).to(device=device)

# for debugging
print(model)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training the model for 100 epoch
best_test_auroc = 0
best_model_bias, best_model_weight = None, None

for e in range(epoch):
	model.train()
	total_loss = 0.0

	for inputs, targets in train_dataloader:
		inputs, targets = inputs.to(device), targets.to(device)

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		total_loss+=loss.item()

	train_metrics = compute_test_metrices(model=model, dataloader=train_dataloader, criterion=criterion, device=device)
	test_metrics = compute_test_metrices(model=model, dataloader=test_dataloader, criterion=criterion, device=device)

	if test_metrics["auroc"] > best_test_auroc:
		continue
		best_model_weight = model.linear.weight.data
		best_model_bias = model.linear.bias.data

	print(f"For epoch={e+1},")
	print(f"\ttrain auroc = {train_metrics["auroc"]}, train_loss = {train_metrics["loss"]}")
	print(f"\ttest auroc = {test_metrics["auroc"]}, test_loss = {test_metrics["loss"]}\n\n")


def df_to_matrix(dataframe):
	matrix = np.zeros((32, 32))

	for key in dataframe["keys"]:
		l, h = re.match(r"layer_(\d+)_head_(\d+)_\w*").groups()
		matrix[l][h] = dataframe.loc[dataframe["key"] == key, "weight"].iloc[0]

	return matrix

def save_heatmap(matrix, name, title):
	sns.heatmap(matrix, cbar=True)
	plt.title(title)
	plt.xlabel("head")
	plt.ylabel("layer")
	plt.tight_layout()
	plt.savefig(f"./outputs/heatmaps/name.png")

if weight_path or heatmap:
	df = pd.DataFrame({
		"keys": train_set.feature_columns,
		"weights": best_model_weight.cpu().detach().flatten(),
	})

	if weight_path:
		df.to_csv(weight_path)

	if len(df["keys"]) < 32*32:
		raise Exception("All layers and heads are not provided")
	
	if heatmap:
		# if text and vision are fused then, there will be one heatmap <name>.png
		# else, there will be three heatmaps; <name>-text.png, <name>-visual.png, <name>-fused.png
		if df["keys"].values[0].endswith("_feature"):
			heatmap_matrix = df_to_matrix(df)
			save_heatmap(heatmap_matrix, name=heatmap, title="Heatmap for fused")
		else:
			subdf_text = df[df["keys"].str.endswith("text", na=False)]
			subdf_visual = df[df["keys"].str.endswith("visual", na=False)]
			
			text_matrix = df_to_matrix(subdf_text)
			visual_matrix = df_to_matrix(subdf_visual)
			fused_matrix = (text_matrix+visual_matrix)/2

			save_heatmap(text_matrix, name=f"{heatmap}_text", title="Heatmap for text")
			save_heatmap(visual_matrix, name=f"{heatmap}_visual", title="Heatmap for visual")
			save_heatmap(fused_matrix, name=f"{heatmap}_fused", title="Heatmap for fused")

if has_breakpoint:
	breakpoint()
