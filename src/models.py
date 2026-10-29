"""Models used in the Fashion-MNIST capstone."""
from torch import nn


class LinearBaseline(nn.Module):
    """A multinomial linear classifier over the 784 input pixels."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(28 * 28, num_classes)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.classifier(x)


class SmallCNN(nn.Module):
    """Two convolutional blocks and a small dropout-regularised head."""
    def __init__(self, num_classes: int = 10, dropout: float = 0.30) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str):
    """Create a model by its CLI name."""
    name = name.lower()
    if name == "linear":
        return LinearBaseline()
    if name == "cnn":
        return SmallCNN()
    raise ValueError(f"Unknown model {name!r}; expected 'linear' or 'cnn'.")
