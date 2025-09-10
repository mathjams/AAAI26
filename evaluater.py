import os, numpy as np, torch, importlib, matplotlib.pyplot as plt
from IPython.display import Image, display
import evaluater
importlib.reload(evaluater)
from collator import make_bag_windows

def _ensure_T2(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got {arr.shape}")
    if arr.shape[1] == 2:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] == 2:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"Cannot coerce to (T,2); got {arr.shape}")

def _as_1d(t):
    if torch.is_tensor(t):
        return t.reshape(1) if t.ndim == 0 else t
    return torch.as_tensor(t).reshape(-1)

def _compute_starts(T, L, stride, pad_short=True):
    if T >= L:
        starts = list(range(0, T - L + 1, stride))
        last_start = T - L
        if (T - L) % stride != 0 and (len(starts) == 0 or starts[-1] != last_start):
            starts.append(last_start)
    elif pad_short:
        starts = [0]
    else:
        starts = []
    return starts

def _plot_bag_points_two_tone(seq_xy, s, e, out_png, title):
    x, y = seq_xy[:, 0], seq_xy[:, 1]
    xc, yc = np.clip(x, 0.0, 1.0), np.clip(y, 0.0, 1.0)
    T = len(x)
    s = max(0, min(int(s), T))
    e = max(s, min(int(e), T))

    mask = np.zeros(T, dtype=bool)
    mask[s:e] = True

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(1, 1, 1)
    if (~mask).any():
        ax.scatter(xc[~mask], yc[~mask], s=14, alpha=0.9, color="blue", label="non-window")
    if mask.any():
        ax.scatter(xc[mask], yc[mask], s=22, alpha=0.95, color="red", label="positive window")

    ax.set_title(title, pad=10)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _get_seq_by_id(seq_id, sequences, all_ids=None):
    if isinstance(sequences, dict):
        return sequences[seq_id]
    try:
        return sequences[int(seq_id)]
    except Exception:
        if all_ids is None:
            raise
        id_to_idx = {bid: i for i, bid in enumerate(all_ids)}
        return sequences[id_to_idx[seq_id]]


@torch.no_grad()
def plot_all_correct_positive_bags(
    sequences,
    labels_tensor,
    val_ids,
    encoder_module,
    mil_head_module,
    context_length: int,
    stride: int,
    device: str,
    run_dir: str,
    tag: str = "",
    pad_short: bool = True,
    k: int = 1,
    threshold: float = 0.0,
    all_ids=None,
    show_first: int = 8,  
):
    os.makedirs(run_dir, exist_ok=True)
    out_dir = os.path.join(run_dir, "qual_plots")
    os.makedirs(out_dir, exist_ok=True)

    encoder_module.to(device).eval()
    mil_head_module.to(device).eval()

    asd_val_ids = [int(i) for i in val_ids if int(labels_tensor[int(i)]) == 1]
    made = []

    print(f"[INFO] Scanning {len(asd_val_ids)} ASD-labelled validation bags...")
    for seq_id in asd_val_ids:
        try:
            seq_xy = _ensure_T2(_get_seq_by_id(seq_id, sequences, all_ids=all_ids))
            pv, pm = make_bag_windows(
                seq_xy, context_length=context_length, stride=stride, pad_short=pad_short, add_noise=0.0
            )  
            if pv.shape[0] == 0:
                print(f"  - skip seq {seq_id}: no windows")
                continue

            out = encoder_module(past_values=pv.to(device), past_observed_mask=pm.to(device), return_dict=True)
            tokens = out.last_hidden_state
            inst_scores = _as_1d(mil_head_module(tokens).squeeze(-1))

            k_eff = max(1, min(k, inst_scores.numel()))
            topk_vals, _ = torch.topk(inst_scores, k_eff)
            bag_score = float(topk_vals.mean().item())
            pred = int(bag_score > threshold)
            if pred != 1:
                continue

            top_inst_idx = int(torch.argmax(inst_scores).item())
            T = seq_xy.shape[0]; L = int(context_length)
            starts = _compute_starts(T, L, stride, pad_short=pad_short)
            if not starts:
                print(f"  - skip seq {seq_id}: empty starts")
                continue
            if top_inst_idx >= len(starts):
                top_inst_idx = min(top_inst_idx, len(starts) - 1)
            s = int(starts[top_inst_idx]); e = int(min(s + L, T))

            base = f"bag{seq_id}_inst{top_inst_idx}"
            out_png = os.path.join(out_dir, f"{tag}_{base}_points_two_tone.png") if tag else os.path.join(out_dir, f"{base}_points_two_tone.png")
            _plot_bag_points_two_tone(
                seq_xy=seq_xy,
                s=s,
                e=e,
                out_png=out_png,
                title=f"Seq {seq_id} — all points (red=in positive instance)",
            )
            made.append({"seq_id": seq_id, "path": out_png, "s": s, "e": e, "bag_score": bag_score})

        except Exception as ex:
            print(f"  ! seq {seq_id}: error -> {ex}")

    print(f"[DONE] Generated {len(made)} plots in {out_dir}")
    for item in made[:show_first]:
        print(f"  - {item}")
        try:
            display(Image(filename=item["path"]))
        except Exception:
            pass
    return made
