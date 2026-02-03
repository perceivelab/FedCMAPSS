import torch

class PiecewiseScaler:
    def __init__(self, max_rul=125.0, normalize=125.0):
        self.max_rul = max_rul
        self.normalize = normalize
        
    def __call__(self, y):
        # Clip
        if self.max_rul is not None:
            if isinstance(y, torch.Tensor):
                y = torch.clamp(y, max=self.max_rul)
            else:
                y = min(y, self.max_rul)
        # Normalize
        if self.normalize is not None:
            y = y / self.normalize
        return y

    def inverse(self, y_norm):
        # Get back to cycles
        if self.normalize is not None:
            return y_norm * self.normalize
        return y_norm

def compute_nasa_score(y_pred, y_true):
    # Ensure shapes match and flatten
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    # Calculate Error (d = Predicted - True)
    d = y_pred - y_true
    # Initialize score tensor on the same device as inputs
    scores = torch.zeros_like(d)
    # Case 1: Early Predictions (d < 0) -> Penalty = exp(-d/13) - 1
    # Penalized less because it's safer to maintain early
    mask_early = d < 0
    scores[mask_early] = torch.exp(-d[mask_early] / 13) - 1
    # Case 2: Late Predictions (d >= 0) -> Penalty = exp(d/10) - 1
    # Penalized heavily because it leads to failure
    mask_late = d >= 0
    scores[mask_late] = torch.exp(d[mask_late] / 10) - 1
    # Return SUM for this batch (scalar item)
    return scores.sum().item()