import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel
import collator, mil

# --- 1. Patched encoder builder ---
def build_frozen_patchtst(
    repo_id: str = "namctin/patchtst_etth1_pretrain",
    num_input_channels: int = 2,
    context_length: int = 64,
    patch_length: int = 1,
    patch_stride: int = 1,
    d_model: int = None,
    use_cls_token: bool = True,
    fine_tune_last_n_layers: int = 0,  # new arg
):
    cfg = PatchTSTConfig.from_pretrained(repo_id)
    cfg.num_input_channels = num_input_channels
    cfg.context_length = context_length
    cfg.patch_length = patch_length
    cfg.patch_stride = patch_stride
    if d_model is not None:
        cfg.d_model = d_model
    cfg.use_cls_token = use_cls_token

    model = PatchTSTModel.from_pretrained(
        repo_id, config=cfg, ignore_mismatched_sizes=True
    )

    # freeze all first
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last N transformer blocks if requested
    if fine_tune_last_n_layers and hasattr(model, "encoder"):
        blocks = list(model.encoder.layers) if hasattr(model.encoder, "layers") else []
        for b in blocks[-int(fine_tune_last_n_layers):]:
            for p in b.parameters():
                p.requires_grad = True

    model.eval()
    return model, cfg