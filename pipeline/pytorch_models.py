"""PyTorch model definitions for text and image tasks."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object  # noqa: A001


if torch is not None:

    class TextCNN(nn.Module):
        """Kim (2014) style TextCNN for sentence classification."""

        def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 128, num_filters: int = 100):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in (3, 4, 5)
            ])
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(num_filters * 3, num_classes)

        def forward(self, x):
            emb = self.embedding(x).transpose(1, 2)
            conv_out = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
            out = self.dropout(torch.cat(conv_out, dim=1))
            return self.fc(out)

    class TextLSTM(nn.Module):
        """Bidirectional LSTM for text classification."""

        def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 128, hidden_dim: int = 128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(
                embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=0.0
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            emb = self.embedding(x)
            _, (hidden, _) = self.lstm(emb)
            h = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.fc(self.dropout(h))

    class ImageCNN(nn.Module):
        """Lightweight CNN for image classification (64x64 input)."""

        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

else:
    TextCNN = TextLSTM = ImageCNN = None
