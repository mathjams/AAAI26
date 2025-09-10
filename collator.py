
import torch
import numpy as np
import random
from typing import List, Tuple
from typing import List, Tuple, Optional
def duplicate_indices(indices, repeats: int):
    return [i for i in indices for _ in range(repeats)]

def duplicate_indices_classwise(indices, labels, pos_times: int = 3, neg_times: int = 1):
    out = []
    for i in indices:
        y = 1 if labels[i] > 0 else 0
        r = pos_times if y == 1 else neg_times
        out.extend([i]*r)
    return out

def _pad_slice(seq: np.ndarray, s: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    T = len(seq)
    if T >= s + L:
        win = seq[s:s+L]
        msk = np.ones((L, 2), dtype=bool)
    else:
        deficit = s + L - T
        pad = np.zeros((deficit, 2), dtype=np.float32)
        win = np.concatenate([pad, seq[s:]], axis=0)[:L]
        m0 = np.zeros((deficit, 2), dtype=bool)
        m1 = np.ones((L - deficit, 2), dtype=bool)
        msk = np.concatenate([m0, m1], axis=0)
    return win.astype(np.float32), msk

def make_bag_windows_sliding(
    seq: np.ndarray,
    length: int,
    hop: int,
    *,
    pad_short: bool = True,
    align_last: bool = True,
    jitter_base: bool = False,
    seed: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq.ndim == 2 and seq.shape[1] == 2, "seq must be (T,2)"
    L = int(length); H = max(1, int(hop))
    T = seq.shape[0]

    rng = np.random.RandomState(seed) if seed is not None else np.random
    starts: List[int] = []

    if T >= L:
        base = int(rng.randint(0, H)) if jitter_base else 0
        base = max(0, min(base, H - 1))
        starts = list(range(base, T - L + 1, H))
        last_start = T - L
        if align_last and (len(starts) == 0 or starts[-1] != last_start):
            starts.append(last_start)
    elif pad_short and T > 0:
        starts = [0]
    else:
        return (torch.empty(0, L, 2), torch.empty(0, L, 2, dtype=torch.bool))

    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, num=max_windows, dtype=int)
        starts = [starts[i] for i in idx]

    wins, masks = zip(*[_pad_slice(seq, s, L) for s in starts])
    past_values = torch.from_numpy(np.stack(wins, axis=0))        # (N,L,2)
    past_observed_mask = torch.from_numpy(np.stack(masks, axis=0))# (N,L,2)
    return past_values, past_observed_mask

def make_bag_windows_random(
    seq: np.ndarray,
    length: int,
    n_windows: int,
    *,
    pad_short: bool = True,
    seed: Optional[int] = None,
    with_replacement: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq.ndim == 2 and seq.shape[1] == 2, "seq must be (T,2)"
    L = int(length)
    T = seq.shape[0]
    if T < L and not pad_short:
        return (torch.empty(0, L, 2), torch.empty(0, L, 2, dtype=torch.bool))

    rng = np.random.RandomState(seed) if seed is not None else np.random
    if T >= L:
        max_start = T - L
        if with_replacement:
            starts = [int(rng.randint(0, max_start + 1)) for _ in range(n_windows)]
        else:
            n = min(n_windows, max_start + 1)
            starts = rng.choice(max_start + 1, size=n, replace=False).tolist()
            starts.sort()
    else:
        starts = [0]  # will be padded

    wins, masks = zip(*[_pad_slice(seq, s, L) for s in starts])
    return torch.from_numpy(np.stack(wins, 0)), torch.from_numpy(np.stack(masks, 0))

def make_bag_windows(
    seq: np.ndarray,
    context_length: int = 64,
    stride: int = 16,
    pad_short: bool = True,
    add_noise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert seq.ndim == 2 and seq.shape[1] == 2, "Each sequence must be (T, 2)"
    L = context_length
    T = len(seq)
    seq = seq.astype(np.float32)

    starts: List[int] = []
    if T >= L:
        starts = list(range(0, T - L + 1, stride))
        last_start = T - L
        if (T - L) % stride != 0 and (len(starts) == 0 or starts[-1] != last_start):
            starts.append(last_start)
    elif pad_short:
        starts = [0]
    else:
        return (torch.empty(0, L, 2), torch.empty(0, L, 2, dtype=torch.bool))

    windows, masks = [], []
    for s in starts:
        if T >= s + L:
            win = seq[s:s+L]                                  # (L, 2)
            msk = np.ones((L, 2), dtype=bool)
        else:
            deficit = s + L - T
            pad = np.zeros((deficit, 2), dtype=np.float32)
            win = np.concatenate([pad, seq[s:]], axis=0)[:L]
            m0 = np.zeros((deficit, 2), dtype=bool)
            m1 = np.ones((L - deficit, 2), dtype=bool)
            msk = np.concatenate([m0, m1], axis=0)

        if add_noise > 0.0:
            win = win + np.random.normal(0.0, add_noise, size=win.shape).astype(np.float32)

        windows.append(win)
        masks.append(msk)

    past_values = torch.from_numpy(np.stack(windows, axis=0))        # (Ni, L, 2)
    past_observed_mask = torch.from_numpy(np.stack(masks, axis=0))   # (Ni, L, 2) bool
    return past_values, past_observed_mask

def make_bag_batches(
    bag_indices: List[int],
    batch_size_bags: int,
    shuffle: bool = True,
    seed: int = 42,
) -> List[List[int]]:
    idxs = list(bag_indices)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    return [idxs[i:i+batch_size_bags] for i in range(0, len(idxs), batch_size_bags)]

def stratified_split(indices: List[int], y: List[int], test_size: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_cls = {}
    for idx, lbl in zip(indices, y):
        by_cls.setdefault(int(lbl), []).append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_cls.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * test_size))) if len(idxs) > 1 else 1
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx
