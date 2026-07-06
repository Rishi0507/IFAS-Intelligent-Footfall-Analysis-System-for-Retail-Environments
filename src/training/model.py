"""GenderClassifier: pure ViT backbone + binary classification head."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import ViTModel

class GenderClassifier(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224", num_labels=2):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        hidden_dim = self.vit.config.hidden_size  # 768
        self.head = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        self.id2label = {0: "female", 1: "male"}
        self.label2id = {"female": 0, "male": 1}
        self.vit_model_name = vit_model_name

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        return self.head(out.pooler_output)

    def embed(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        emb = out.pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)

    def freeze_backbone(self):
        for p in self.vit.parameters(): p.requires_grad = False
        for p in self.head.parameters(): p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.parameters(): p.requires_grad = True

    def param_groups(self, head_lr, backbone_lr):
        head_p, backbone_p = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            (head_p if name.startswith("head.") else backbone_p).append(p)
        groups = []
        if head_p: groups.append({"params": head_p, "lr": head_lr, "name": "head"})
        if backbone_p: groups.append({"params": backbone_p, "lr": backbone_lr, "name": "backbone"})
        return groups

    def save(self, out_dir):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_dir / "pytorch_model.bin")
        cfg = {"architectures": ["GenderClassifier"], "model_type": "vit_gender",
               "vit_model_name": self.vit_model_name, "num_labels": self.num_labels,
               "id2label": self.id2label, "label2id": self.label2id}
        with open(out_dir / "config.json", "w") as f: json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, ckpt_dir, device="cpu"):
        with open(Path(ckpt_dir) / "config.json") as f: cfg = json.load(f)
        m = cls(cfg["vit_model_name"], cfg.get("num_labels", 2))
        state = torch.load(Path(ckpt_dir) / "pytorch_model.bin", map_location=device)
        m.load_state_dict(state, strict=False)
        m.to(device).eval()
        return m
