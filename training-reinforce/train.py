"""REINFORCE training step over a self-play cohort.

On-policy policy gradient with a learned value baseline + entropy bonus
(matches the fox-lite REINFORCE setup):
  advantage   = z - value(state).detach()
  policy_loss = -(advantage * log pi(action|state)).mean()      # over legal moves
  value_loss  = MSE(value(state), z)
  entropy     = mean entropy of the (legal-masked) policy
  loss        = policy_loss + c_value*value_loss - c_entropy*entropy

The cohort is on-policy data from the *current* weights, and we take exactly
**one** optimizer step over the whole cohort. To keep that single step's GPU
memory bounded (cohorts can be hundreds of thousands of rows, and the GPU is
shared), the step is computed by **gradient accumulation**: the cohort is split
into `micro_batch` chunks, each chunk's mean loss is scaled by its share of the
cohort and backpropagated (accumulating grads), then a single optimizer step is
applied. The accumulated gradient equals the full-cohort mean gradient, so this
is mathematically one full-batch SGD step — just memory-bounded.
"""

import numpy as np
import torch
import torch.nn.functional as F

MASK_NEG = 1.0e9


def train_on_cohort(net, opt, cohort, device, *, micro_batch=8192,
                    c_value=1.0, c_entropy=0.05, grad_clip=1.0, rng=None):
    states = cohort["states"]
    masks = cohort["masks"]
    actions = cohort["actions"]
    z = cohort["z"]
    n = cohort["n"]
    if rng is None:
        rng = np.random.default_rng()
    perm = rng.permutation(n)

    net.train()
    opt.zero_grad(set_to_none=True)
    agg = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}

    # Use only a whole number of micro_batch-sized chunks so every forward has the
    # same shape — torch.compile then compiles one graph and reuses it, instead of
    # recompiling for a variable-size tail each cohort. The dropped remainder is
    # < micro_batch random rows (perm is shuffled), negligible for a ~200k cohort.
    if n > micro_batch:
        n_used = (n // micro_batch) * micro_batch
        chunks = [perm[i:i + micro_batch] for i in range(0, n_used, micro_batch)]
    else:
        n_used = n
        chunks = [perm]

    # One optimizer step over the cohort, accumulated chunk by chunk. Each chunk's
    # mean loss is weighted by |chunk|/n_used so the summed gradient is exactly the
    # mean gradient over the used rows.
    for idx in chunks:
        w = len(idx) / n_used
        s = torch.from_numpy(states[idx]).to(device).view(-1, 18, 8, 8)
        m = torch.from_numpy(masks[idx]).to(device)
        a = torch.from_numpy(actions[idx]).to(device)
        zz = torch.from_numpy(z[idx]).to(device).float()

        logits, v = net(s)
        masked = logits + (m - 1.0) * MASK_NEG
        logp = F.log_softmax(masked, dim=1)
        logp_a = logp.gather(1, a[:, None]).squeeze(1)
        adv = zz - v.detach()
        policy_loss = -(adv * logp_a).mean()
        value_loss = F.mse_loss(v, zz)
        p = logp.exp()
        # Entropy over legal moves only (padded logp is -inf -> p*logp is 0 there).
        ent_terms = torch.where(m > 0, p * logp, torch.zeros_like(p))
        entropy = -ent_terms.sum(dim=1).mean()
        loss = policy_loss + c_value * value_loss - c_entropy * entropy

        (loss * w).backward()

        agg["loss"] += float(loss.item()) * w
        agg["policy"] += float(policy_loss.item()) * w
        agg["value"] += float(value_loss.item()) * w
        agg["entropy"] += float(entropy.item()) * w

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    opt.step()

    agg["steps"] = 1
    return agg
