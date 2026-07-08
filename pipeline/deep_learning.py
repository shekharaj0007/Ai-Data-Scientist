"""
Optional PyTorch training for TextCNN, LSTM, and Image CNN.
Activated automatically when torch is installed; uses GPU if available.
"""
from __future__ import annotations

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image

    from .pytorch_models import TextCNN, TextLSTM, ImageCNN

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Dataset = None
    DataLoader = None
    transforms = None
    Image = None
    TextCNN = TextLSTM = ImageCNN = None


def is_available() -> bool:
    return TORCH_AVAILABLE


def get_device_info() -> dict:
    if not TORCH_AVAILABLE:
        return {"available": False, "device": "none", "cuda": False}
    cuda = torch.cuda.is_available()
    return {
        "available": True,
        "device": "cuda" if cuda else "cpu",
        "cuda": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda else "CPU",
    }


def train_text_models(
    texts: list[str],
    labels: np.ndarray,
    model_dir: str,
    text_columns: list[str],
) -> tuple[list[dict], dict | None, object | None]:
    """Train TextCNN and TextLSTM; returns (leaderboard_rows, best_artifact, best_model_for_shap)."""
    if not TORCH_AVAILABLE or len(texts) < 16:
        return [], None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128
    vocab = _build_vocab(texts, max_vocab=15000)
    if len(vocab) < 10:
        return [], None, None

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels.astype(str))
    num_classes = len(label_encoder.classes_)

    X_idx = np.array([_text_to_ids(t, vocab, max_len) for t in texts])
    X_train, X_test, y_train, y_test = train_test_split(X_idx, y, test_size=0.2, random_state=42)

    architectures = [
        ("pytorch_text_cnn", TextCNN, "cnn"),
        ("pytorch_text_lstm", TextLSTM, "rnn"),
    ]

    leaderboard = []
    best_score = -1.0
    best_artifact = None
    best_name = None

    for name, model_cls, family in architectures:
        try:
            model = model_cls(len(vocab), num_classes).to(device)
            score = _train_text_model(model, X_train, y_train, X_test, y_test, device, epochs=6)
            leaderboard.append({
                "model": name,
                "family": family,
                "cv_mean_score": score,
                "cv_std": 0.0,
                "test_score": round(score, 4),
            })
            if score > best_score:
                best_score = score
                best_name = name
                best_artifact = {
                    "model_type": "deep_text",
                    "architecture": name,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_class": model_cls.__name__,
                    "vocab": vocab,
                    "max_len": max_len,
                    "num_classes": num_classes,
                    "label_encoder": label_encoder,
                    "text_columns": text_columns,
                    "device": str(device),
                }
        except Exception:
            continue

    return leaderboard, best_artifact, best_name


def train_image_models(
    image_paths: list[str],
    labels: np.ndarray,
    model_dir: str,
    image_columns: list[str],
) -> tuple[list[dict], dict | None, str | None]:
    if not TORCH_AVAILABLE or len(image_paths) < 20:
        return [], None, None

    valid = [(p, l) for p, l in zip(image_paths, labels) if p and os.path.isfile(p)]
    if len(valid) < 20:
        return [], None, None

    paths, labs = zip(*valid)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(np.array(labs).astype(str))

    train_paths, test_paths, y_train, y_test = train_test_split(
        list(paths), y, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_encoder.classes_)
    model = ImageCNN(num_classes).to(device)

    try:
        score = _train_image_model(model, train_paths, y_train, test_paths, y_test, device, epochs=8)
    except Exception:
        return [], None, None

    artifact = {
        "model_type": "deep_image",
        "architecture": "pytorch_image_cnn",
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_class": "ImageCNN",
        "num_classes": num_classes,
        "label_encoder": label_encoder,
        "image_columns": image_columns,
        "device": str(device),
    }

    row = {
        "model": "pytorch_image_cnn",
        "family": "cnn",
        "cv_mean_score": score,
        "cv_std": 0.0,
        "test_score": round(score, 4),
    }
    return [row], artifact, "pytorch_image_cnn"


def predict_deep(artifact: dict, records: list[dict]) -> list:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")

    model_type = artifact.get("model_type")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = artifact.get("label_encoder")

    if model_type == "deep_text":
        texts = _records_to_texts(records, artifact.get("text_columns", []))
        vocab = artifact["vocab"]
        max_len = artifact["max_len"]
        X = torch.tensor([_text_to_ids(t, vocab, max_len) for t in texts], dtype=torch.long).to(device)

        model_cls = TextCNN if artifact.get("model_class") == "TextCNN" else TextLSTM
        model = model_cls(len(vocab), artifact["num_classes"]).to(device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=1).cpu().numpy()

    elif model_type == "deep_image":
        paths = _records_to_image_paths(records, artifact.get("image_columns", []))
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensors = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        X = torch.stack(tensors).to(device)
        model = ImageCNN(artifact["num_classes"]).to(device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=1).cpu().numpy()
    else:
        raise ValueError(f"Unknown deep model type: {model_type}")

    if label_encoder is not None:
        return label_encoder.inverse_transform(preds).tolist()
    return preds.tolist()


def save_best_deep_artifact(artifact: dict, model_path: str) -> None:
    joblib.dump(artifact, model_path)


def explain_text_tokens(artifact: dict, sample_text: str | None = None, top_n: int = 10) -> list[dict]:
    """Gradient-based token saliency for PyTorch text models."""
    if not TORCH_AVAILABLE:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = artifact["vocab"]
    max_len = artifact["max_len"]
    text = sample_text or "important feature words drive predictions"
    token_ids = _text_to_ids(text, vocab, max_len)
    tokens = _ids_to_tokens(token_ids, vocab)

    model_cls = TextCNN if artifact.get("model_class") == "TextCNN" else TextLSTM
    model = model_cls(len(vocab), artifact["num_classes"]).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    emb = model.embedding(x)
    emb.retain_grad()
    if isinstance(model, TextCNN):
        emb_t = emb.transpose(1, 2)
        conv_out = [torch.relu(c(emb_t)).max(dim=2).values for c in model.convs]
        out = model.fc(model.dropout(torch.cat(conv_out, dim=1)))
    else:
        out, _ = model.lstm(emb)
        h = torch.cat((out[:, -1, : model.lstm.hidden_size], out[:, 0, model.lstm.hidden_size:]), dim=1)
        out = model.fc(model.dropout(h))

    pred_class = out.argmax(dim=1)
    out[0, pred_class].backward()
    scores = emb.grad.abs().sum(dim=2).detach().cpu().numpy()[0]

    ranked = sorted(
        [(tokens[i], float(scores[i])) for i in range(len(tokens)) if tokens[i] not in ("<pad>", "<unk>") and scores[i] > 0],
        key=lambda t: t[1],
        reverse=True,
    )[:top_n]
    return [{"token": t, "score": round(s, 4)} for t, s in ranked]


def _ids_to_tokens(ids: list[int], vocab: dict) -> list[str]:
    inv = {v: k for k, v in vocab.items()}
    return [inv.get(i, "<unk>") for i in ids]


# ── Internal helpers ───────────────────────────────────────

def _build_vocab(texts: list[str], max_vocab: int = 15000) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def _text_to_ids(text: str, vocab: dict, max_len: int) -> list[int]:
    ids = [vocab.get(w, 1) for w in _tokenize(text)][:max_len]
    ids += [0] * (max_len - len(ids))
    return ids


def _records_to_texts(records: list[dict], text_columns: list[str]) -> list[str]:
    texts = []
    for row in records:
        parts = [str(row.get(c, "")) for c in text_columns if c in row]
        texts.append(" ".join(parts) if parts else str(row.get("text", "")))
    return texts


def _records_to_image_paths(records: list[dict], image_columns: list[str]) -> list[str]:
    paths = []
    for row in records:
        for col in image_columns:
            if col in row and row[col]:
                paths.append(str(row[col]))
                break
        else:
            paths.append("")
    return paths


def _train_text_model(model, X_train, y_train, X_test, y_test, device, epochs: int = 6) -> float:
    X_tr = torch.tensor(X_train, dtype=torch.long).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te = torch.tensor(X_test, dtype=torch.long).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    batch_size = min(64, len(X_tr))
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1)
        acc = (preds == y_te).float().mean().item()
    return round(acc, 4)


if TORCH_AVAILABLE:

    class _ImagePathDataset(Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.labels[idx]


    def _train_image_model(model, train_paths, y_train, test_paths, y_test, device, epochs: int = 8) -> float:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        train_ds = _ImagePathDataset(train_paths, list(y_train), transform)
        loader = DataLoader(train_ds, batch_size=min(32, len(train_ds)), shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        test_ds = _ImagePathDataset(test_paths, list(y_test), test_transform)
        test_loader = DataLoader(test_ds, batch_size=32)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        return round(correct / max(total, 1), 4)

else:

    def _train_image_model(*args, **kwargs):
        raise RuntimeError("PyTorch not installed")
