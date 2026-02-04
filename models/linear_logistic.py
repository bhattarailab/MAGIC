import torch

class LinearLogisticRegression(torch.nn.Module):
	def __init__(self, in_feautres=32*32*2, hidden_dim=512):
		super().__init__()
		self.linear1 = torch.nn.Linear(in_features=in_feautres, out_features=hidden_dim, bias=True)
		self.relu = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=1, bias=True)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		y = self.linear1(x)
		y = self.relu(y)
		y = self.linear2(y)
		y = self.sigmoid(y)

		return y
