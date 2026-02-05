import torch

class TwoStageModel(torch.nn.Module):
	def __init__(self, in_features=32*32):
		super().__init__()
		self.text_aggregator = torch.nn.Linear(in_features=in_features, out_features=1, bias=True)
		self.visual_aggregator = torch.nn.Linear(in_features=in_features, out_features=1, bias=True)
		self.sigmoid = torch.nn.Sigmoid()
		self.classifier = torch.nn.Linear(2, 1, bias=True)

	def forward(self, x_text, x_visual):
		y_text = self.text_aggregator(x_text)
		y_text = self.sigmoid(y_text)
		
		y_visual = self.text_aggregator(x_visual)
		y_visual = self.sigmoid(y_visual)

		y = self.classifier()