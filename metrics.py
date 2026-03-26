import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def batch_accuracy(output: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of correct predictions in a single batch."""
    preds = output.argmax(dim=1)
    return preds.eq(targets).sum().item() / len(targets)


def top_k_accuracy(output: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Fraction of samples where the true label is in the top-k predictions."""
    _, top_k_preds = output.topk(k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_preds)
    correct = top_k_preds.eq(targets_expanded).any(dim=1).sum().item()
    return correct / len(targets)


def evaluate(
    model: torch.nn.Module,
    loader,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """
    Full evaluation pass over a DataLoader.

    Returns:
        dict with keys: accuracy, report (str), confusion_matrix (np.ndarray)
    """
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return {"accuracy": float(accuracy), "report": report, "confusion_matrix": cm}
