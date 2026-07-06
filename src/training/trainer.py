"""Two-phase training loop: linear probe → full fine-tune with cosine decay."""
from __future__ import annotations
import math, os, time, json
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch, torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/run"
    phase_a_epochs: int = 3
    phase_a_head_lr: float = 1e-3
    phase_b_epochs: int = 10
    phase_b_head_lr: float = 1e-5
    phase_b_backbone_lr: float = 1e-5
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    fp16: bool = True
    log_every: int = 50
    save_best: bool = True

def _cosine_with_warmup(opt, warmup, total):
    from torch.optim.lr_scheduler import LambdaLR
    def lam(step):
        if step < warmup: return step / max(1, warmup)
        return max(0.0, 0.5*(1+math.cos(math.pi*(step-warmup)/max(1,total-warmup))))
    return LambdaLR(opt, lam)

def evaluate(model, loader, device):
    model.eval(); nc, nt, mc, mt, fc, ft = 0,0,0,0,0,0
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                logits = model(imgs)
            preds = logits.argmax(-1)
            nc += int((preds==labs).sum()); nt += labs.size(0)
            for l, p in zip(labs.tolist(), preds.tolist()):
                if l==1: mt+=1; mc+=int(p==1)
                else: ft+=1; fc+=int(p==0)
    return {"accuracy": nc/max(nt,1), "male_acc": mc/max(mt,1),
            "female_acc": fc/max(ft,1), "n_total": nt}

def train_phase(model, train_loader, val_loader, device, phase_name,
                epochs, head_lr, backbone_lr, freeze,
                weight_decay=0.01, warmup_ratio=0.1, grad_clip=1.0,
                fp16=True, log_every=50, output_dir=".", best_acc=0.0):
    if freeze: model.freeze_backbone(); print(f"[{phase_name}] FROZEN, head only")
    else: model.unfreeze_backbone(); print(f"[{phase_name}] UNFROZEN, full fine-tune")
    opt = torch.optim.AdamW(model.param_groups(head_lr, backbone_lr), weight_decay=weight_decay)
    total = epochs * len(train_loader); warmup = int(warmup_ratio * total)
    sched = _cosine_with_warmup(opt, warmup, total)
    scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and device=="cuda"))
    step, t0, history = 0, time.time(), []
    for ep in range(epochs):
        model.train(); rloss, rcorr, rtot = 0.0, 0, 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(fp16 and device=="cuda")):
                loss = nn.functional.cross_entropy(model(imgs), labs)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            rloss += loss.item()*imgs.size(0)
            rcorr += int((model(imgs).argmax(-1)==labs).sum()); rtot += imgs.size(0)
            if step % log_every == 0:
                eta = (time.time()-t0)/step*(total-step)
                print(f"  [{phase_name}] ep{ep+1}/{epochs} step {step}/{total} "
                      f"loss={rloss/rtot:.4f} acc={rcorr/rtot:.3f} lr={opt.param_groups[0]['lr']:.2e} eta={eta:.0f}s")
        m = evaluate(model, val_loader, device)
        print(f"  [{phase_name}] ep{ep+1} VAL acc={m['accuracy']:.4f} male={m['male_acc']:.3f} female={m['female_acc']:.3f}")
        history.append({"phase": phase_name, "epoch": ep+1, **m})
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            print(f"  [{phase_name}] NEW BEST, saving to {output_dir}/best")
            model.save(os.path.join(output_dir, "best"))
    return best_acc, history
