import torch

# intializing a dummy dataset with 200 samples and two vectors for each head
class SimpleDataset(torch.utils.data.Dataset):
	def __init__(self):
		super().__init__()
		self.x = torch.empty(200, 32*32*2).to(torch.float32)
		self.y = torch.randint(low=0, high=2, size=(200, 1)).to(torch.float32)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index):
		return (self.x[index], self.y[index])

dataset = SimpleDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

class SimpleLogisticRegression(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = torch.nn.Linear(in_features=32*32*2, out_features=1, bias=True)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		y = self.linear(x)
		y = self.sigmoid(y)

		return y

model = SimpleLogisticRegression()
criterion = torch.nn.BCELoss()

EPOCH = 100
for e in range(EPOCH):
	for inputs, targets in dataloader:
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		model.zero_grad()
