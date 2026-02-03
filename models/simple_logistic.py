import torch

class SimpleLogisticRegression(torch.nn.Module):
	def __init__(self, in_feautres=32*32*2):
		super().__init__()
		self.linear = torch.nn.Linear(in_features=in_feautres, out_features=1, bias=True)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		y = self.linear(x)
		y = self.sigmoid(y)

		return y