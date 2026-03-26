import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.metrics import batch_accuracy

logger = get_logger(__name__)


class EarlyStopping:
    """Halt training when validation loss stops improving; persist best weights."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, checkpoint_path: str = "best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info(f"Checkpoint saved → {self.checkpoint_path}  (val_loss={val_loss:.4f})")
        else:
            self.counter += 1
            logger.info(f"No improvement. Early-stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    """
    Encapsulates the full training lifecycle:
    - Per-epoch training and validation
    - Learning-rate scheduling
    - Early stopping with automatic best-model checkpointing
    - Device-aware (CPU / CUDA / MPS auto-detected)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        augmentation=None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        patience: int = 7,
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: torch.device | None = None,
    ):
        self.device = device or self._auto_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.augmentation = augmentation
        self.epochs = epochs

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)

        logger.info(f"Trainer ready  |  device={self.device}  |  params={model.count_parameters():,}")

    @staticmethod
    def _auto_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Core epoch methods
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = total_acc = 0.0

        for x, y in tqdm(self.train_loader, desc="  train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            if self.augmentation:
                x = self.augmentation(x)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += batch_accuracy(output, y)

        n = len(self.train_loader)
        return total_loss / n, total_acc / n

    def _validate_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = total_acc = 0.0

        with torch.no_grad():
            for x, y in tqdm(self.valid_loader, desc="  valid", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                total_loss += self.loss_fn(output, y).item()
                total_acc += batch_accuracy(output, y)

        n = len(self.valid_loader)
        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run the full training loop. Returns loss/accuracy history."""
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch:03d}/{self.epochs}")

            t_loss, t_acc = self._train_epoch()
            v_loss, v_acc = self._validate_epoch()

            self.scheduler.step(v_loss)

            logger.info(
                f"  Train  loss={t_loss:.4f}  acc={t_acc:.4f}\n"
                f"  Valid  loss={v_loss:.4f}  acc={v_acc:.4f}"
            )

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            if self.stopper.step(v_loss, self.model):
                logger.info("Early stopping triggered. Restoring best weights.")
                self.model.load_state_dict(
                    torch.load(self.stopper.checkpoint_path, map_location=self.device)
                )
                break

        return history
