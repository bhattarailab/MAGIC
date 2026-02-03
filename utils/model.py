import torch
from sklearn.metrics import roc_auc_score

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

