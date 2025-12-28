import numpy as np
import torch
from model import PointRegressor

class displacement_predictor:
    def __init__(self, weight_path: str, device: torch.device | None = None):
        # Pick device once during construction.
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model weights.
        self.regressionModel = PointRegressor().to(self.device)
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.regressionModel.load_state_dict(checkpoint["model_state"])
        self.regressionModel.eval()

    def _to_tensor(self, points: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Convert numpy -> torch and ensure float32.
        if isinstance(points, np.ndarray):
            tensor = torch.tensor(points, dtype=torch.float32)
        else:
            tensor = points.float()

        # Ensure batch dimension: (N, 2) -> (1, N, 2)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, source_points: np.ndarray | torch.Tensor,
                target_points: np.ndarray | torch.Tensor) -> torch.Tensor:
        # Convert inputs to tensors on the correct device.
        source_tensor = self._to_tensor(source_points)
        target_tensor = self._to_tensor(target_points)

        # Run inference without gradient tracking.
        with torch.no_grad():
            displacement = self.regressionModel(source_tensor, target_tensor)

        # Return displacement on CPU for easy downstream use.
        return displacement.cpu()
