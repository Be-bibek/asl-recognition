from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image

from src.models.model import ASLClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 24 static ASL classes (J and Z require motion and are excluded)
ASL_CLASSES: list[str] = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in ("J", "Z")]


def load_model(
    checkpoint_path: str,
    num_classes: int = 24,
    in_channels: int = 1,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a saved ASLClassifier checkpoint and set it to eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ASLClassifier(num_classes=num_classes, in_channels=in_channels)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}  →  {device}")
    return model.to(device)


def _build_inference_transform(img_size: int, grayscale: bool) -> T.Compose:
    steps = [T.Resize((img_size, img_size))]
    if grayscale:
        steps.append(T.Grayscale(num_output_channels=1))
    steps += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    return T.Compose(steps)


def predict(
    model: nn.Module,
    image_path: str,
    class_names: list[str] = ASL_CLASSES,
    img_size: int = 28,
    grayscale: bool = True,
    top_k: int = 3,
    device: torch.device | None = None,
) -> list[dict[str, float | str]]:
    """
    Run inference on a single image file.

    Returns:
        Ranked list of top-k dicts: [{"label": "A", "confidence": 0.94}, ...]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = _build_inference_transform(img_size, grayscale)

    image = Image.open(image_path).convert("L" if grayscale else "RGB")
    tensor = transform(image).unsqueeze(0).to(device)   # (1, C, H, W)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))

    results = [
        {"label": class_names[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    logger.info(f"Prediction for '{image_path}': {results}")
    return results
