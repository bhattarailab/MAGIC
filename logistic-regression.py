import torch
import pandas as pd
import argparse
from models.simple_logistic import SimpleLogisticRegression
from utils.model import compute_test_metrices

parser = argparse.ArgumentParser(description="performs logistic regression on the given csv file")
parser.add_argument("--csv", help="CSV file created using create-dataset.py")
parser.add_argument("--device", help="device to run the model", default="cuda")
parser.add_argument("--epoch", help="Epoch for training the model", default=50, type=int)
parser.add_argument("--weight-path", help="Path where the weights of neuron is to be saved")

args = parser.parse_args()
csv_path, device = args.csv, args.device
epoch = args.epoch
weight_path = args.weight_path

class ConnectivityVectorDataset(torch.utils.data.Dataset):
	def __init__(self, csv_path, split="train"):
		super().__init__()
		df = pd.read_csv(csv_path)
		split_df = df[df["split"] == split]

		features_column = [s for s in split_df.columns if s.startswith("layer_")]
		self.X = split_df[features_column].to_numpy()
		self.y = split_df["label"].to_numpy()

	def __len__(self):
		return len(self.y)

	def __getitem__(self, index):
		return torch.tensor(self.X[index]), torch.tensor(self.y[index]).reshape((1, 1))

# intialziign train and test dataset from the pickles path
train_set = ConnectivityVectorDataset(csv_path, split="train")
test_set = ConnectivityVectorDataset(csv_path, split="test")

# creating dataloader with batching
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# initializing the model
model = SimpleLogisticRegression().to(device=device)
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
		best_model_weight = model.linear.weight.data
		best_model_bias = model.linear.bias.data

	print(f"For epoch={e+1},")
	print(f"\ttrain auroc = {train_metrics["auroc"]}, train_loss = {train_metrics["loss"]}")
	print(f"\ttest auroc = {test_metrics["auroc"]}, test_loss = {test_metrics["loss"]}\n\n")

if weight_path:
	df = pd.DataFrame({
		"keys": [
			*[f"layer_{i}_head_{j}_text" for i in range(32) for j in range(32)],
			*[f"layer_{i}_head_{j}_visual" for i in range(32) for j in range(32)],
		],
		"weights": best_model_weight.cpu().detach().flatten(),
	})
	df.to_csv(weight_path)

breakpoint()
