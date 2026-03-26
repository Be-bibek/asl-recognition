"""
tests/test_model.py

Run with:  pytest tests/
"""

import torch
import pytest

from src.models.model import ASLClassifier, ConvBlock


class TestConvBlock:
    def test_output_shape(self):
        block = ConvBlock(in_channels=1, out_channels=32)
        x = torch.randn(4, 1, 28, 28)
        out = block(x)
        # MaxPool2d(2) halves spatial dims: 28 → 14
        assert out.shape == (4, 32, 14, 14)

    def test_custom_dropout(self):
        block = ConvBlock(in_channels=3, out_channels=64, dropout_p=0.4)
        assert block is not None


class TestASLClassifier:
    def test_default_forward_pass(self):
        model = ASLClassifier()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 24), f"Expected (8, 24), got {out.shape}"

    def test_custom_num_classes(self):
        model = ASLClassifier(num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 10)

    def test_rgb_input(self):
        model = ASLClassifier(in_channels=3)
        x = torch.randn(4, 3, 64, 64)   # also tests resolution independence
        out = model(x)
        assert out.shape == (4, 24)

    def test_resolution_independence(self):
        """AdaptiveAvgPool2d means the model handles any H×W."""
        model = ASLClassifier()
        for size in [28, 56, 112]:
            x = torch.randn(2, 1, size, size)
            out = model(x)
            assert out.shape == (2, 24), f"Failed at size={size}"

    def test_parameter_count(self):
        model = ASLClassifier()
        count = model.count_parameters()
        assert count > 0, "Model has no trainable parameters"

    def test_custom_channel_list(self):
        model = ASLClassifier(channel_list=[16, 32])
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 24)

    def test_eval_mode_no_grad(self):
        model = ASLClassifier().eval()
        x = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 24)
