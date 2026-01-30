import torch
import pickle
from sklearn.metrics import roc_auc_score
import pandas as pd

device = "cuda"
train_connectivity_path = "./outputs/feidlermst_all_train.pkl"
test_connectivity_path = "./outputs/feidlermst_all_test.pkl"

class FiedlersDataset(torch.utils.data.Dataset):
	def __init__(self, path):
		super().__init__()

		with open(path, "rb") as fp:
			self.metadata = pickle.load(fp)

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, index):
		m = self.metadata[index]
		y = torch.tensor([1.0]) if m["is_hallucinating"] else torch.tensor([0.0])

		all_keys = [f"layer_{i}_head_{j}" for i in range(32) for j in range(32)]
		text_tensor = torch.tensor([m["connectivity"][k]["text"]["fiedler_value"] for k in all_keys], dtype=torch.float32)
		visual_tensor = torch.tensor([m["connectivity"][k]["visual"]["fiedler_value"] for k in all_keys], dtype=torch.float32)

		X = torch.concat((text_tensor, visual_tensor))

		return X, y

class SimpleLogisticRegression(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(in_features=32*32*2, out_features=1, bias=True)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		y = self.linear(x)
		y = self.sigmoid(y)

		return y

def compute_test_metrices(model, dataloader, criterion, device="cuda"):
	with torch.no_grad():
		model.eval()
		total_loss = 0.0

		actuals, predictions = [], []
		for batch_inputs, batch_targets in dataloader:
			batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
			outputs = model(batch_inputs)

			loss = criterion(outputs, batch_targets)
			total_loss += loss

			actuals.extend(batch_targets.cpu().numpy())
			predictions.extend(outputs.cpu().numpy())
		
		auroc = roc_auc_score(actuals, predictions)
		return {
			"auroc": auroc,
			"loss": total_loss/len(dataloader),
		}


# intialziign train and test dataset from the pickles path
train_set = FiedlersDataset(train_connectivity_path)
test_set = FiedlersDataset(test_connectivity_path)

# creating dataloader with batching
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# initializing the model
model = SimpleLogisticRegression().to(device=device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

EPOCH = 100
best_test_auroc = 0
best_model_bias, best_model_weight = None, None

for e in range(EPOCH):
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
		best_model_bias = model.linear.weight.bias

	print(f"For epoch={e+1},")
	print(f"\ttrain auroc = {train_metrics["auroc"]}, train_loss = {train_metrics["loss"]}")
	print(f"\ttest auroc = {test_metrics["auroc"]}, test_loss = {test_metrics["loss"]}\n\n")

df = pd.DataFrame({
	"keys": [
		*[f"layer_{i}_head_{j}_text" for i in range(32) for j in range(32)],
		*[f"layer_{i}_head_{j}_visual" for i in range(32) for j in range(32)],
	],
	"weights": best_model_weight,
})
df.to_csv("./outputs/weights.csv")
