"""
train.py — entry point for model training.

Usage:
    python train.py
    python train.py --config configs/train_config.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

from src.models.model import ASLClassifier
from src.services.trainer import Trainer
from src.utils.data import build_train_transforms, get_data_loaders
from src.utils.logger import get_logger
from src.utils.metrics import evaluate
from src.services.inference import ASL_CLASSES

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/train_config.yaml") -> None:
    cfg = load_config(config_path)

    # --- Paths ---
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / cfg["output"]["checkpoint_name"])
    log_file = cfg["output"].get("log_file")

    # Reattach logger with file output now that we have the path
    get_logger("src", log_file=log_file)

    # --- Data ---
    train_loader, valid_loader = get_data_loaders(
        train_dir=cfg["data"]["train_dir"],
        valid_dir=cfg["data"]["valid_dir"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # --- Model ---
    model = ASLClassifier(
        num_classes=cfg["model"]["num_classes"],
        in_channels=cfg["data"]["in_channels"],
        channel_list=cfg["model"]["channel_list"],
        fc_dropout=cfg["model"]["fc_dropout"],
    )
    logger.info(f"Model  |  params={model.count_parameters():,}")

    # --- Augmentation (applied on-device inside Trainer) ---
    augmentation = build_train_transforms(img_size=cfg["data"]["img_size"])

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        augmentation=augmentation,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        checkpoint_path=checkpoint_path,
    )
    history = trainer.fit()

    # --- Evaluate ---
    logger.info("Running final evaluation on validation set...")
    results = evaluate(model, valid_loader, ASL_CLASSES, trainer.device)
    logger.info(f"Validation accuracy: {results['accuracy']:.4f}")
    logger.info("\n" + results["report"])

    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    main(args.config)
