import torch
import numpy as np

class FiedlerValue:
	seed = 0

	@classmethod
	def using_torch(cls, matrix, device="cuda"):
		# Better to use isinstance for type checking
		assert isinstance(matrix, np.ndarray), "Input must be a numpy array"
		
		n, m = matrix.shape
		
		# Convert input to tensor
		A = torch.tensor(matrix, device=device, dtype=torch.float32)

		# Make the matrix symmetric (bipartite adjacency matrix)
		# Shape becomes (n+m) * (n+m)
		# Structure:
		# [ 0  A ]
		# [ A.T 0 ]
		A_symmetric = torch.cat([
				torch.cat([torch.zeros((n, n), device=A.device), A], dim=1),
				torch.cat([A.T, torch.zeros((m, m), device=A.device)], dim=1),
		], dim=0)

		# 1. Calculate Degrees
		degrees = A_symmetric.sum(dim=1)

		# 2. Calculate D^(-1/2) (Inverse Square Root)
		# We use a mask to handle division by zero for isolated nodes
		d_inv_sqrt = torch.pow(degrees, -0.5)
		d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
		
		D_inv_sqrt = torch.diag(d_inv_sqrt)

		# 3. Calculate Normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
		normalized_adj = D_inv_sqrt @ A_symmetric @ D_inv_sqrt
		identity = torch.eye(A_symmetric.shape[0], device=A.device)
		
		normalized_laplacian = identity - normalized_adj

		# 4. Eigen decomposition
		# eigvalsh is correct here because the Normalized Laplacian is symmetric
		eigenvalues = torch.linalg.eigvalsh(normalized_laplacian)
		
		# The Fiedler value is the second smallest eigenvalue (index 1)
		# Note: Ensure graph is connected, otherwise this might be 0.
		fiedler_value = eigenvalues[1].item()
		
		return {
			"feidlers": fiedler_value,
		}