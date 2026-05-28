"""Training step: pull a minibatch from the replay buffer, compute the
AlphaZero loss (policy cross-entropy + value MSE + L2), step the optimizer."""

import collections
import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Ring buffer of (state, pi, z) tuples."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = collections.deque(maxlen=capacity)

    def add_many(self, tuples):
        self.buf.extend(tuples)

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size, rng=None):
        rng = rng or np.random
        if len(self.buf) == 0:
            return None
        idxs = rng.choice(len(self.buf), size=batch_size, replace=True)
        states = np.stack([self.buf[i][0] for i in idxs])
        pis = np.stack([self.buf[i][1] for i in idxs])
        zs = np.array([self.buf[i][2] for i in idxs], dtype=np.float32)
        return states, pis, zs


def train_step(net, opt, batch, device):
    states, pis, zs = batch
    s = torch.from_numpy(states).to(device)
    pi = torch.from_numpy(pis).to(device)
    z = torch.from_numpy(zs).to(device)

    logits, v = net(s)
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(pi * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(v, z)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
    }
