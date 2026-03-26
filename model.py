import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Reusable conv block: Conv2d → BatchNorm → ReLU → Dropout2d → MaxPool2d.
    Dropout2d drops entire feature maps, which is more effective than
    scalar dropout for convolutional layers.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASLClassifier(nn.Module):
    """
    Configurable CNN for ASL static hand-sign classification.

    Args:
        num_classes:  Number of output classes (default 24 — A–Z minus J and Z).
        in_channels:  1 for grayscale, 3 for RGB.
        channel_list: Output channels per ConvBlock. Controls model depth.
        fc_dropout:   Dropout probability in the fully-connected head.
    """

    def __init__(
        self,
        num_classes: int = 24,
        in_channels: int = 1,
        channel_list: list[int] | None = None,
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        if channel_list is None:
            channel_list = [32, 64, 128, 256]

        # Build conv blocks with gradually increasing dropout
        blocks = []
        ch = in_channels
        for i, out_ch in enumerate(channel_list):
            blocks.append(ConvBlock(ch, out_ch, dropout_p=0.1 + i * 0.05))
            ch = out_ch
        self.features = nn.Sequential(*blocks)

        # AdaptiveAvgPool decouples the classifier head from input resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        fc_in = channel_list[-1] * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=fc_dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
