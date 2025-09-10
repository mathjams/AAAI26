import os, glob, torch, numpy as np
from IPython.display import Image, display

def plot_confusion(cm, class_names=("TD","ASD"), normalize=False, outpath=None):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix' + (' (Normalized)' if normalize else ''))
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            txt = f'{cm[i, j]:.2f}' if normalize else int(round(cm[i, j]))
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=160); plt.close(fig)
    return fig

def _resolve_tag_and_dir(path_or_tag: str, run_dir: str = "mil_runs"):
    if path_or_tag.endswith(".csv") or path_or_tag.endswith(".pt"):
        dir_ = os.path.dirname(path_or_tag) or run_dir
        base = os.path.splitext(os.path.basename(path_or_tag))[0]
    else:
        dir_ = os.path.dirname(path_or_tag) or run_dir
        base = os.path.basename(path_or_tag)
    tag = base
    for suf in ("_bestHinge", "_bestTPF1"):
        if tag.endswith(suf):
            tag = tag[:-len(suf)]
    return dir_, tag

def _kind_to_paths(dir_, tag, kind: str):
    k = kind.lower()
    if k in ("best", "hinge", "besthinge"):
        cm_png      = os.path.join(dir_, f"{tag}_val_confusion_bestHinge.png")
        cm_norm_png = os.path.join(dir_, f"{tag}_val_confusion_bestHinge_norm.png")
        ckpt_path   = os.path.join(dir_, f"{tag}_bestHinge.pt")
        row_csv     = os.path.join(dir_, f"{tag}_bestHinge.csv")
    elif k in ("besttpf1", "tpf1", "tpweightedf1", "best_tpweighted_f1"):
        cm_png      = os.path.join(dir_, f"{tag}_val_confusion_bestTPF1.png")
        cm_norm_png = os.path.join(dir_, f"{tag}_val_confusion_bestTPF1_norm.png")
        ckpt_path   = os.path.join(dir_, f"{tag}_bestTPF1.pt")
        row_csv     = os.path.join(dir_, f"{tag}_bestTPF1.csv")
    else:
        raise ValueError(f"Unknown kind='{kind}'. Use 'best'/'hinge' or 'bestTPF1'.")
    return cm_png, cm_norm_png, ckpt_path, row_csv

def load_confusion_from_csv(path_or_tag: str,
                            mil_head=None,
                            device: str = "cpu",
                            kind: str = "best",
                            run_dir: str = "mil_runs"):
    dir_, tag = _resolve_tag_and_dir(path_or_tag, run_dir=run_dir)
    cm_png, cm_norm_png, ckpt_path, _row_csv = _kind_to_paths(dir_, tag, kind)

    shown = []
    if os.path.exists(cm_png):
        shown.append(cm_png)
    if os.path.exists(cm_norm_png):
        shown.append(cm_norm_png)
    if shown and Image is not None:
        for p in shown:
            display(Image(filename=p))

    ckpt = None
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if mil_head is not None and "mil_head" in ckpt:
            try:
                mil_head.load_state_dict(ckpt["mil_head"], strict=False)
                mil_head.to(device).eval()
            except Exception as e:
                print("[warn] Could not load mil_head state:", e)

        if (not shown) and ("val_cm" in ckpt):
            plot_confusion(np.array(ckpt["val_cm"]), class_names=("TD","ASD"),
                           normalize=False, outpath=cm_png)
            plot_confusion(np.array(ckpt["val_cm"]), class_names=("TD","ASD"),
                           normalize=True,  outpath=cm_norm_png)
            shown = [cm_png, cm_norm_png]
            if Image is not None:
                for p in shown:
                    display(Image(filename=p))
    else:
        if not shown:
            print(f"[info] No confusion PNGs or checkpoint found for tag='{tag}' and kind='{kind}'. "
                  f"Looked for: {cm_png} and {ckpt_path}")

    return ckpt, shown, tag

def load_best_checkpoint(path_or_tag, mil_head, device="cpu", run_dir="mil_runs", kind="best"):
    assert kind in ("best", "bestHinge")
    if str(path_or_tag).endswith(".csv"):
        ckpt, shown, tag = load_confusion_from_csv(path_or_tag, mil_head=mil_head, device=device, kind=kind, run_dir=run_dir)
        dir_, tag = _tag_from_csv(path_or_tag, run_dir)
        ckpt_path, _, _ = _paths_for_kind(dir_, tag, kind)
        if ckpt is None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt is not None:
            missing, unexpected = mil_head.load_state_dict(ckpt["mil_head"], strict=False)
            mil_head.eval()
            print(f"Loaded: {ckpt_path}")
            print("  state_dict | missing:", missing, "| unexpected:", unexpected)
            m = ckpt.get("val_metrics", {})
            print(f"Restored from epoch {ckpt.get('epoch','?')} | "
                  f"val_hinge={ckpt.get('val_hinge', float('nan')):.4f} | "
                  f"val_acc={ckpt.get('val_acc', float('nan')):.3f} | "
                  f"balanced_acc={m.get('balanced_acc', float('nan')):.3f} | "
                  f"P={m.get('precision', float('nan')):.3f} R={m.get('recall', float('nan')):.3f} | "
                  f"ROC-AUC={m.get('roc_auc', float('nan')):.3f} | PR-AUC={m.get('pr_auc', float('nan')):.3f}")
            if "threshold" in ckpt:
                print(f"  threshold={ckpt['threshold']}  top_k={ckpt.get('top_k','?')}  selection={ckpt.get('selection','?')}")
        return ckpt, ckpt_path if 'ckpt_path' in locals() else None, tag

    suffix = "_bestHinge.pt" if kind == "bestHinge" else "_best.pt"
    if str(path_or_tag).endswith(".pt"):
        ckpt_path = path_or_tag
        tag = os.path.basename(ckpt_path)[:-len(suffix)]
    else:
        tag = path_or_tag
        ckpt_path = os.path.join(run_dir, f"{tag}{suffix}")

    if not os.path.exists(ckpt_path):
        print(f"[info] '{ckpt_path}' not found. Searching for latest *{suffix} in '{run_dir}'.")
        candidates = sorted(glob.glob(os.path.join(run_dir, f"*{suffix}")))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints matching *{suffix} in '{run_dir}'.")
        ckpt_path = candidates[-1]
        tag = os.path.basename(ckpt_path)[:-len(suffix)]

    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = mil_head.load_state_dict(ckpt["mil_head"], strict=False)
    mil_head.eval()

    print(f"Loaded: {ckpt_path}")
    print("  state_dict | missing:", missing, "| unexpected:", unexpected)
    m = ckpt.get("val_metrics", {})
    print(f"Restored from epoch {ckpt.get('epoch','?')} | "
          f"val_hinge={ckpt.get('val_hinge', float('nan')):.4f} | "
          f"val_acc={ckpt.get('val_acc', float('nan')):.3f} | "
          f"balanced_acc={m.get('balanced_acc', float('nan')):.3f} | "
          f"P={m.get('precision', float('nan')):.3f} R={m.get('recall', float('nan')):.3f} | "
          f"ROC-AUC={m.get('roc_auc', float('nan')):.3f} | PR-AUC={m.get('pr_auc', float('nan')):.3f}")
    if "threshold" in ckpt:
        print(f"  threshold={ckpt['threshold']}  top_k={ckpt.get('top_k','?')}  selection={ckpt.get('selection','?')}")

    _, cm_png, cm_png_norm = _paths_for_kind(os.path.dirname(ckpt_path), tag, kind)
    shown = False
    if os.path.exists(cm_png):
        display(Image(filename=cm_png)); shown = True
    if os.path.exists(cm_png_norm):
        display(Image(filename=cm_png_norm)); shown = True
    if (not shown) and ("val_cm" in ckpt):
        plot_confusion(np.array(ckpt["val_cm"]), class_names=("TD","ASD"))

    return ckpt, ckpt_path, tag
