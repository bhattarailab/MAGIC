import torch
import pickle
from sklearn.metrics import roc_auc_score

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
		y = torch.Tensor(1) if m["is_hallucinating"] else torch.Tensor(0)

		X = torch.tensor([m["connectivity"]["text"]["fiedler_value"], m["connectivity"]["visual"]["fiedler_value"]], dtype=torch.float32)

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

		actuals, targets = [], []
		for batch_inputs, batch_targets in dataloader:
			batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
			outputs = model(batch_inputs)

			loss = criterion(outputs, batch_targets)
			total_loss += loss

			actuals.extend(batch_targets.cpu().detach.to_numpy())
			targets.extend(outputs.cpu().detach.to_numpy())
		
		auroc = roc_auc_score(targets, actuals)
		return {
			"auroc": auroc,
			"loss": total_loss/len(dataloader),
		}


# intialziign train and test dataset from the pickles path
train_set = FiedlersDataset(train_connectivity_path)
test_set = FiedlersDataset(test_connectivity_path)

breakpoint()

# creating dataloader with batching
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# initializing the model
model = SimpleLogisticRegression().to(device=device)
criterion = torch.nn.BCELoss()

EPOCH = 100
for e in range(EPOCH):
	model.train()
	total_loss = 0.0
	for inputs, targets in train_dataloader:
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		model.zero_grad()

		total_loss+=loss.item()

	train_metrics = compute_test_metrices(model=model, dataloader=train_dataloader, criterion=criterion, device=device)
	test_metrics = compute_test_metrices(model=model, dataloader=test_dataloader, criterion=criterion, device=device)

	print(f"For epoch={e+1}, train auroc = {train_metrics["auroc"]}, train_loss = {train_metrics["loss"]}")
	print(f"For epoch={e+1}, test auroc = {test_metrics["auroc"]}, test_loss = {test_metrics["loss"]}")
