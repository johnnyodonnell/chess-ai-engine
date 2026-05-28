"""AlphaZero-style ResNet for chess.

4 residual blocks × 96 filters. Policy head outputs 4672 logits, value
head outputs a scalar in [-1, 1] (tanh) from the side-to-move's POV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import INPUT_CHANNELS, POLICY_SIZE


N_BLOCKS = 4
N_FILTERS = 96


class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class ChessNet(nn.Module):
    def __init__(self, n_blocks=N_BLOCKS, n_filters=N_FILTERS):
        super().__init__()
        self.stem_conv = nn.Conv2d(INPUT_CHANNELS, n_filters, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(n_filters)
        self.tower = nn.ModuleList([ResBlock(n_filters) for _ in range(n_blocks)])

        # Policy head: 1×1 conv to 2 channels, then linear to POLICY_SIZE.
        self.policy_conv = nn.Conv2d(n_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, POLICY_SIZE)

        # Value head: 1×1 conv to 1 channel, linear→256→1, tanh.
        self.value_conv = nn.Conv2d(n_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.tower:
            h = block(h)

        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = self.policy_fc(p.flatten(1))

        v = F.relu(self.value_bn(self.value_conv(h)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(-1)


def n_params(net):
    return sum(p.numel() for p in net.parameters())
