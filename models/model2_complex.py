import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, padding=None):
        super().__init__()
        p = padding if padding is not None else p
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiPathParallelBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 4 == 0
        bc = out_channels // 4

        self.branch1 = ConvBNReLU(in_channels, bc, 1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels, bc, 1), ConvBNReLU(bc, bc, 3, padding=1))
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, bc, 1),
            ConvBNReLU(bc, bc, 3, padding=1),
            ConvBNReLU(bc, bc, 3, padding=1),
        )
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), ConvBNReLU(in_channels, bc, 1))

        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
        return self.relu(out + self.shortcut(x))


class ComplexCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 15, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(
            MultiPathParallelBlock(64, 128), MultiPathParallelBlock(128, 128),
            nn.Dropout2d(p=0.1), nn.MaxPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(
            MultiPathParallelBlock(128, 256), MultiPathParallelBlock(256, 256),
            nn.Dropout2d(p=0.15), nn.MaxPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            MultiPathParallelBlock(256, 512), MultiPathParallelBlock(512, 512),
            nn.Dropout2d(p=0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)


def get_model2_complex(num_classes: int = 15, dropout: float = 0.4) -> ComplexCNN:
    return ComplexCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    model = get_model2_complex(num_classes=15)
    dummy = torch.randn(2, 3, 224, 224)
    print(f"[Model 2] Output: {model(dummy).shape}")