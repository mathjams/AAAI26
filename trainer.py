import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from collator import make_bag_batches, make_bag_windows
from transformers import PatchTSTModel
import torch.nn.functional as F
def _pool_scores_for_bag(scores_1d: torch.Tensor, k: int = 1, top_p: float = None) -> torch.Tensor:
    """
    scores_1d: (Ni,)
    If top_p is provided (0<p<=1), use mean of top ceil(p*Ni) scores.
    Else fall back to top-k mean with k capped by Ni.
    """
    Ni = scores_1d.numel()
    if Ni == 0:
        return scores_1d.new_tensor(0.0)
    if top_p is not None:
        n_keep = max(1, int(np.ceil(top_p * Ni)))
        top_vals, _ = torch.topk(scores_1d, n_keep)
        return top_vals.mean()
    k_eff = min(k, Ni)
    top_vals, _ = torch.topk(scores_1d, k_eff)
    return top_vals.mean()

def hinge_loss_pm1(
    scores: torch.Tensor,           
    labels_pm1: torch.Tensor,      
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    margin_pos: float = 1.0,
    margin_neg: float = 1.0,
) -> torch.Tensor:
    pos_mask = labels_pm1 > 0
    neg_mask = ~pos_mask

    if pos_mask.any():
        pos_losses = torch.clamp(margin_pos - scores[pos_mask], min=0.0)
        pos_loss = pos_losses.mean()
    else:
        pos_loss = scores.new_tensor(0.0)

    if neg_mask.any():
        neg_losses = torch.clamp(margin_neg + scores[neg_mask], min=0.0)
        neg_loss = neg_losses.mean()
    else:
        neg_loss = scores.new_tensor(0.0)

    denom = (pos_weight > 0) + (neg_weight > 0)
    return (pos_weight * pos_loss + neg_weight * neg_loss) / max(denom, 1)

@torch.no_grad()
def collect_bag_scores(
    encoder, mil_head, sequences, labels_tensor, bag_indices,
    context_length=64, stride=16, bags_per_batch=8, device="cpu",
    pad_short=True, k=1, top_p: float = None
):
    from collator import make_bag_batches, make_bag_windows

    encoder.to(device).eval()
    mil_head.to(device).eval()

    all_scores, all_labels = [], []
    batches = make_bag_batches(bag_indices, batch_size_bags=bags_per_batch, shuffle=False)

    for b_ids in batches:
        pv_list, pm_list, gids, batch_labels = [], [], [], []
        for bi, bag_id in enumerate(b_ids):
            pv, pm = make_bag_windows(
                sequences[bag_id], context_length=context_length, stride=stride,
                pad_short=pad_short, add_noise=0.0
            )
            Ni = pv.shape[0]
            if Ni == 0:
                L = context_length
                pv = torch.zeros(1, L, 2, dtype=torch.float32)
                pm = torch.zeros(1, L, 2, dtype=torch.bool)
                Ni = 1
            pv_list.append(pv); pm_list.append(pm)
            gids.append(torch.full((Ni,), bi, dtype=torch.long))
            batch_labels.append(labels_tensor[bag_id].item())

        pv_batch = torch.cat(pv_list, dim=0).to(device)
        pm_batch = torch.cat(pm_list, dim=0).to(device)
        gids     = torch.cat(gids, dim=0).to(device)
        B        = len(b_ids)
        y_pm1    = torch.tensor(batch_labels, dtype=torch.float32, device=device)

        out = encoder(past_values=pv_batch, past_observed_mask=pm_batch, return_dict=True)
        tokens = out.last_hidden_state
        instance_scores = mil_head(tokens).squeeze(-1)
        bag_scores = []
        for bi in range(B):
            mask = (gids == bi)
            sc = instance_scores[mask]
            bag_scores.append(_pool_scores_for_bag(sc, k=k, top_p=top_p))
        bag_scores = torch.stack(bag_scores)

        all_scores.append(bag_scores.detach().cpu())
        all_labels.append((y_pm1 > 0).long().cpu())

    return torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()

def compute_bag_loss(bag_scores, y_pm1, *, loss_type="bce",
                     pos_weight=None, gamma_pos=0.0, gamma_neg=2.0,
                     alpha_pos=0.5, beta_f=1.0, eps=1e-8):
    y01 = (y_pm1 > 0).float() 

    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(
            bag_scores, y01, pos_weight=pos_weight
        )

    elif loss_type == "focal_asym":
        p = torch.sigmoid(bag_scores)
        pos = - y01 * ((1 - p) ** gamma_pos) * torch.log(p.clamp_min(eps))
        neg = - (1 - y01) * (p ** gamma_neg) * torch.log((1 - p).clamp_min(eps))
        return (alpha_pos * pos + (1 - alpha_pos) * neg).mean()

    elif loss_type == "f_beta":  
        p = torch.sigmoid(bag_scores)
        tp = (p * y01).sum()
        fp = (p * (1 - y01)).sum()
        fn = ((1 - p) * y01).sum()
        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        beta2 = beta_f * beta_f
        fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)
        return 1 - fbeta  

    elif loss_type == "hinge_weighted":
        pos_w = (pos_weight if pos_weight is not None else 1.0)
        neg_w = (neg_weight if neg_weight is not None else 1.0)  # now defined
        w = torch.where(y_pm1 > 0,
                        bag_scores.new_tensor(pos_w),
                        bag_scores.new_tensor(neg_w))
        return (torch.clamp(1.0 - y_pm1 * bag_scores, min=0.0) * w).mean()

    else:
        raise ValueError(f"Unknown loss_type={loss_type}")

def train_one_epoch_manual_bag(
    encoder: PatchTSTModel,
    mil_head: nn.Module,
    sequences: List[np.ndarray],
    labels_tensor: torch.Tensor,
    bag_indices: List[int],
    context_length: int = 64,
    stride: int = 16,
    bags_per_batch: int = 8,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    pad_short: bool = True,
    add_noise: float = 0.0,
    seed: int = 42,
    k: int = 1,
    top_p: float = None,
    pos_weight: float = 1.0, neg_weight: float = 1.0,
    margin_pos: float = 1.0, margin_neg: float = 1.0,
) -> float:
    encoder.to(device).eval()
    mil_head.to(device).train()

    total_loss, total_bags = 0.0, 0
    batches = make_bag_batches(bag_indices, batch_size_bags=bags_per_batch, shuffle=True, seed=seed)

    for b_ids in batches:
        pv_list, pm_list, group_ids, batch_labels = [], [], [], []

        for bi, bag_id in enumerate(b_ids):
            pv, pm = make_bag_windows(
                sequences[bag_id], context_length=context_length, stride=stride,
                pad_short=pad_short, add_noise=add_noise
            )
            Ni = pv.shape[0]
            if Ni == 0:  
                L = context_length
                pv = torch.zeros(1, L, 2, dtype=torch.float32)
                pm = torch.zeros(1, L, 2, dtype=torch.bool)
                Ni = 1

            pv_list.append(pv)
            pm_list.append(pm)
            group_ids.append(torch.full((Ni,), bi, dtype=torch.long))
            batch_labels.append(labels_tensor[bag_id].item())

        pv_batch = torch.cat(pv_list, dim=0).to(device)   
        pm_batch = torch.cat(pm_list, dim=0).to(device)   
        gids     = torch.cat(group_ids, dim=0).to(device)
        B        = len(b_ids)
        y_pm1    = torch.tensor(batch_labels, dtype=torch.float32, device=device) 

        with torch.no_grad():
            out = encoder(past_values=pv_batch, past_observed_mask=pm_batch, return_dict=True)
            tokens = out.last_hidden_state 

        instance_scores = mil_head(tokens)  
        assert instance_scores.dim() == 1 and instance_scores.size(0) == pv_batch.size(0), \
            f"{instance_scores.shape=} vs {pv_batch.shape=}"
        bag_scores = []
        for bi in range(B):
            mask = (gids == bi)
            sc = instance_scores[mask]
            bag_scores.append(_pool_scores_for_bag(sc, k=k, top_p=locals().get("top_p", None)))
        bag_scores = torch.stack(bag_scores)        
        pos_w_val = None 
        pos_w = torch.tensor([pos_w_val], device=bag_scores.device) if pos_w_val else None
        loss = hinge_loss_pm1(bag_scores, y_pm1,
                          pos_weight=pos_weight, neg_weight=neg_weight,
                          margin_pos=margin_pos, margin_neg=margin_neg)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_bags += B

    return total_loss / max(1, total_bags)

def _safe_auc(labels_np: np.ndarray, scores_np: np.ndarray) -> Tuple[float, float]:
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = roc_auc_score(labels_np, scores_np) if len(np.unique(labels_np)) == 2 else float("nan")
        pr  = average_precision_score(labels_np, scores_np)
        return float(roc), float(pr)
    except Exception:
        return float("nan"), float("nan")

@torch.no_grad()
def evaluate_manual_bag(
    encoder: PatchTSTModel,
    mil_head: torch.nn.Module,
    sequences: List[np.ndarray],
    labels_tensor: torch.Tensor,
    bag_indices: List[int],
    context_length: int = 64,
    stride: int = 16,
    bags_per_batch: int = 8,
    device: str = "cpu",
    pad_short: bool = True,
    k: int = 1,
    threshold: float = 0.0,
    return_confusion: bool = False,
    pos_weight: float = 1.0, neg_weight: float = 1.0,
    margin_pos: float = 1.0, margin_neg: float = 1.0,
) -> Union[Tuple[float, float],
           Tuple[float, float, np.ndarray, Dict[str, float]]]:
    encoder.to(device).eval()
    mil_head.to(device).eval()

    total_hinge, total_acc, total_bags = 0.0, 0, 0
    tp = tn = fp = fn = 0

    all_scores, all_labels = [], []

    batches = make_bag_batches(bag_indices, batch_size_bags=bags_per_batch, shuffle=False)

    for b_ids in batches:
        pv_list, pm_list, group_ids, batch_labels = [], [], [], []

        for bi, bag_id in enumerate(b_ids):
            pv, pm = make_bag_windows(
                sequences[bag_id], context_length=context_length, stride=stride,
                pad_short=pad_short, add_noise=0.0
            )
            Ni = pv.shape[0]
            if Ni == 0:
                L = context_length
                pv = torch.zeros(1, L, 2, dtype=torch.float32)
                pm = torch.zeros(1, L, 2, dtype=torch.bool)
                Ni = 1

            pv_list.append(pv)
            pm_list.append(pm)
            group_ids.append(torch.full((Ni,), bi, dtype=torch.long))
            batch_labels.append(labels_tensor[bag_id].item())

        pv_batch = torch.cat(pv_list, dim=0).to(device)
        pm_batch = torch.cat(pm_list, dim=0).to(device)
        gids     = torch.cat(group_ids, dim=0).to(device)
        B        = len(b_ids)
        y_pm1    = torch.tensor(batch_labels, dtype=torch.float32, device=device)  

        out = encoder(past_values=pv_batch, past_observed_mask=pm_batch, return_dict=True)
        tokens = out.last_hidden_state          
        instance_scores = mil_head(tokens) 
        assert instance_scores.dim() == 1 and instance_scores.size(0) == pv_batch.size(0), \
            f"{instance_scores.shape=} vs {pv_batch.shape=}"

        bag_scores_list = []
        for bi in range(B):
            mask = (gids == bi)
            if mask.any():
                scores = instance_scores[mask]
                k_eff = min(k, scores.numel())
                topk_vals, _ = torch.topk(scores, k_eff)
                bag_scores_list.append(topk_vals.mean())
            else:
                bag_scores_list.append(torch.tensor(0.0, device=device))
        bag_scores = torch.stack(bag_scores_list)  

        pos_w_val = None  =
        pos_w = torch.tensor([pos_w_val], device=bag_scores.device) if pos_w_val else None
        loss = hinge_loss_pm1(bag_scores, y_pm1,
                           pos_weight=pos_weight, neg_weight=neg_weight,
                           margin_pos=margin_pos, margin_neg=margin_neg)
        total_hinge += loss.item() * B

        targets = (y_pm1 > 0).long()                  
        preds   = (bag_scores > threshold).long()    

        total_acc  += (preds == targets).sum().item()
        total_bags += B

        tp += ((preds == 1) & (targets == 1)).sum().item()
        tn += ((preds == 0) & (targets == 0)).sum().item()
        fp += ((preds == 1) & (targets == 0)).sum().item()
        fn += ((preds == 0) & (targets == 1)).sum().item()

        all_scores.append(bag_scores.detach().cpu())
        all_labels.append(targets.detach().cpu())

    val_hinge = total_hinge / max(1, total_bags)
    val_acc   = total_acc  / max(1, total_bags)

    if not return_confusion:
        return val_hinge, val_acc

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    precision     = tp / max(tp + fp, 1)
    recall        = tp / max(tp + fn, 1)       
    specificity   = tn / max(tn + fp, 1)
    balanced_acc  = 0.5 * (recall + specificity)

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()
    roc_auc, pr_auc = _safe_auc(labels_np, scores_np)

    metrics = dict(
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        balanced_acc=float(balanced_acc),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
    )
    return val_hinge, val_acc, cm, metrics
