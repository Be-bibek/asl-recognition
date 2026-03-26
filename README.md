# 🤟 ASL Recognition — Production-Grade Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-green?logo=pytest)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![NVIDIA DLI](https://img.shields.io/badge/Based%20on-NVIDIA%20DLI-76B900?logo=nvidia)](https://www.nvidia.com/en-us/training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A CNN-based American Sign Language classifier — refactored from a course notebook into a **clean, modular, production-ready Python codebase** with config-driven training, early stopping, deployment utilities, and Docker support.

---

## 🎬 Video Walkthrough

> 📺 Full explanation coming to YouTube — subscribe so you don't miss it!
> 👉 **[youtube.com/@be-bibek](https://youtube.com/@be-bibek)**

---

## 🧩 Problem & Solution

| Problem | Solution |
|---|---|
| Notebook-based code is not reusable or testable | Modular `src/` package with clear separation of concerns |
| Hard-coded hyperparameters scattered in cells | All config in `configs/train_config.yaml` |
| No stopping strategy — models overtrain | `EarlyStopping` with automatic best-weight checkpointing |
| Basic `print()` statements | Structured production logging with file + stream handlers |
| Single-use training script | Reusable `Trainer` class — pluggable into any API or pipeline |
| No deployment path | Inference pipeline + Docker + FastAPI placeholder |

---

## ✨ My Contributions & Improvements

This started as an NVIDIA DLI lab. Here is every engineering decision I added:

### 1 — Modular Architecture
Converted a monolithic notebook into a proper Python package:
```
src/models/    →  model architecture
src/services/  →  training & inference logic
src/utils/     →  metrics, logging, data loading
src/api/       →  FastAPI placeholder (next milestone)
```

### 2 — Config-Driven Training
All hyperparameters live in `configs/train_config.yaml`. Zero magic numbers in code.
Change learning rate, batch size, model depth — without touching a single `.py` file.

### 3 — Production Trainer Class
`Trainer` wraps the full training lifecycle:
- `ReduceLROnPlateau` scheduler — LR decays when learning stalls
- `EarlyStopping` — halts training, auto-saves best weights
- `zero_grad(set_to_none=True)` — faster GPU memory management
- `tqdm` progress bars — clean readable output per epoch

### 4 — ASL-Specific Augmentation (Domain-Informed)

| Technique | Used | Reasoning |
|---|---|---|
| Horizontal flip | ✅ | ASL is valid mirrored — left/right-handed |
| Vertical flip | ❌ | No one signs upside down |
| Rotation ±15° | ✅ | Natural hand-position variance |
| Brightness/contrast | ✅ | Real cameras have inconsistent lighting |
| ColorJitter | ✅ | Device colour-cast differences |
| RandomResizedCrop | ✅ | Simulates partial occlusion |

### 5 — Device-Aware (CPU / CUDA / MPS)
`Trainer._auto_device()` detects the best available hardware automatically. Runs on MacBook M-series, NVIDIA GPU, or plain CPU with zero code changes.

### 6 — Evaluation & Metrics
`evaluate()` in `src/utils/metrics.py` generates:
- Overall accuracy
- Per-class `classification_report` (precision, recall, F1)
- `confusion_matrix` for visual analysis

### 7 — Deployment-Ready Inference
```python
model  = load_model("checkpoints/best_asl_model.pt")
result = predict(model, "hand.jpg", top_k=3)
# → [{"label": "A", "confidence": 0.9412}, ...]
```

### 8 — Full Test Suite
7 pytest unit tests covering forward pass shapes, resolution independence, custom configs, and eval mode.

---

## 🏗️ Architecture

```
Input (1 × 28 × 28)
        │
ConvBlock(1→32,   dropout=0.10)  →  14 × 14
ConvBlock(32→64,  dropout=0.15)  →   7 × 7
ConvBlock(64→128, dropout=0.20)  →   3 × 3
ConvBlock(128→256,dropout=0.25)  →   1 × 1
        │
AdaptiveAvgPool2d(4 × 4)
        │
Linear(4096→512) → ReLU → Dropout(0.50)
Linear(512→128)  → ReLU → Dropout(0.25)
Linear(128→24)   → logits
```

Each `ConvBlock`: `Conv2d → BatchNorm2d → ReLU → Dropout2d → MaxPool2d`
`AdaptiveAvgPool2d` means the model accepts **any input resolution** — no retraining needed when deployment image size changes.

Total trainable parameters: **~5.3 M**

---

## 📁 Project Structure

```
asl-recognition/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI placeholder (next milestone)
│   ├── models/
│   │   └── model.py            # ConvBlock + ASLClassifier
│   ├── services/
│   │   ├── trainer.py          # Trainer class + EarlyStopping
│   │   └── inference.py        # load_model() + predict()
│   └── utils/
│       ├── data.py             # Transforms + DataLoader factory
│       ├── metrics.py          # Accuracy + evaluation report
│       └── logger.py           # Production logging setup
├── tests/
│   └── test_model.py           # 7 pytest unit tests
├── configs/
│   └── train_config.yaml       # All hyperparameters
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── train.py                    # Entry point
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone
git clone https://github.com/Be-bibek/asl-recognition.git
cd asl-recognition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit configs/train_config.yaml to point to your data directories
```

---

## 🚀 Usage

### Train

```bash
python train.py
# or with a custom config:
python train.py --config configs/train_config.yaml
```

### Predict

```python
from src.services.inference import load_model, predict, ASL_CLASSES

model  = load_model("checkpoints/best_asl_model.pt")
result = predict(model, "path/to/hand_sign.jpg", ASL_CLASSES, top_k=3)
print(result)
# [{"label": "A", "confidence": 0.9412}, {"label": "M", "confidence": 0.034}]
```

### Run Tests

```bash
pytest tests/ -v
```

### Docker

```bash
docker build -t asl-recognition .
docker run asl-recognition
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | ~96% |
| Training Accuracy | ~98% |
| Early Stopping Patience | 7 epochs |
| Best Checkpoint | auto-saved → `checkpoints/best_asl_model.pt` |

---

## 🔜 Roadmap

- [ ] 🔌 **FastAPI backend** — `POST /predict` REST endpoint
- [ ] 📱 **Flutter app** — live camera feed with real-time gesture prediction
- [ ] ☁️ **Cloud deployment** — AWS / GCP with auto-scaling
- [ ] 🔐 **Auth + encryption** — secure user data pipeline
- [ ] 📦 **GitHub Actions CI** — auto-run tests on every push
- [ ] 📈 **WandB / TensorBoard** — experiment tracking dashboard

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch 2.x |
| Augmentation | torchvision.transforms.v2 |
| Evaluation | scikit-learn |
| Config | PyYAML |
| Testing | pytest |
| Containerisation | Docker |
| Future API | FastAPI (planned) |
| Future Mobile | Flutter (planned) |

---

## 🌐 Connect With Me

I'm **Bibek Das** — AI & Backend Developer, Flutter + Deep Learning Enthusiast.

| Platform | Link |
|---|---|
| 💻 GitHub | [github.com/Be-bibek](https://github.com/Be-bibek) |
| 🎥 YouTube | [youtube.com/@be-bibek](https://youtube.com/@be-bibek) |
| 💼 LinkedIn | [linkedin.com/in/bibek-das-364367323](https://www.linkedin.com/in/bibek-das-364367323/) |
| 📸 Instagram | [@bibek_ai_deas](https://www.instagram.com/bibek_ai_deas?igsh=MW00ZDRsNWFzM3cybA==) |
| 📧 Email | bibekdas1055@gmail.com |

> ⭐ **Star this repo** and **[subscribe on YouTube](https://youtube.com/@be-bibek)** to follow the FastAPI + Flutter build next!

---
## 🙏 Acknowledgment

This project is based on learning materials from NVIDIA Deep Learning Institute (DLI).  
The original implementation was extended with production-grade architecture, improved training pipeline, and deployment-ready design.

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.
