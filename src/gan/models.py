
import torch
from torch import nn


class SelfAttention2d(nn.Module):
    """Spatial self-attention for 2D feature maps."""

    def __init__(self, channels):
        super().__init__()

        hidden = max(8, channels // 8)

        self.query = nn.Conv2d(channels, hidden, 1)
        self.key = nn.Conv2d(channels, hidden, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.shape

        query = self.query(x).view(
            batch, -1, height * width
        ).transpose(1, 2)

        key = self.key(x).view(
            batch, -1, height * width
        )

        attention = torch.softmax(
            torch.bmm(query, key), dim=-1
        )

        value = self.value(x).view(
            batch, channels, height * width
        )

        output = torch.bmm(
            value, attention.transpose(1, 2)
        )

        output = output.view(
            batch, channels, height, width
        )

        return self.gamma * output + x


class ConditionalGenerator(nn.Module):

    def __init__(self, num_classes=3, latent=64, attention=False):
        super().__init__()

        self.label = nn.Embedding(num_classes, latent)

        self.fc = nn.Linear(latent * 2, 256 * 8 * 8)

        layers = [
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        ]

        if attention:
            layers.append(SelfAttention2d(128))

        layers += [
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, z, labels):

        x = torch.cat(
            [z, self.label(labels)], dim=1
        )

        x = self.fc(x).view(
            x.size(0), 256, 8, 8
        )

        return self.net(x)


class ConditionalDiscriminator(nn.Module):

    def __init__(self, num_classes=3, attention=False):
        super().__init__()

        self.label = nn.Embedding(
            num_classes, 64 * 64
        )

        layers = [
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        ]

        if attention:
            layers.append(SelfAttention2d(128))

        layers += [
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, image, labels):

        label_map = self.label(labels).view(
            labels.size(0), 1, 64, 64
        )

        combined = torch.cat(
            [image, label_map], dim=1
        )

        return self.net(combined)
