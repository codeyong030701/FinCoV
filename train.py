"""
Train the FinCoV Consensus Learning Module on the official PIG training split.

Example:
  python train.py --hand left --pig_root data/pig --device cuda \
    --out output/consensus_left/best.pt --epochs 120 --batch_size 64
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Make imports robust when script is placed at project root.
_THIS = Path(__file__).resolve()
for _cand in [_THIS.parent, *_THIS.parents]:
    if (_cand / "piano").exists():
        sys.path.insert(0, str(_cand))
        break

from fincov.data import NUM_FINGERS, parse_pig_file, notes_to_features
from fincov.consensus import ConsensusEmissionTransformer


def get_piece_id(path: Path) -> int:
    m = re.match(r"(\d+)-", path.name)
    return int(m.group(1)) if m else 999


def find_pig_files(pig_root: Path) -> List[Path]:
    pattern = re.compile(r"\d+-\d+_fingering\.txt$", re.IGNORECASE)
    out: List[Path] = []
    for root, _, fnames in os.walk(pig_root):
        for fname in sorted(fnames):
            if pattern.match(fname):
                out.append(Path(root) / fname)
    return out


def align_multiple_ground_truths(paths: List[Path], hand: str) -> Tuple[np.ndarray | None, List[np.ndarray]]:
    master_notes = parse_pig_file(paths[0], hand_filter=hand)
    if not master_notes:
        return None, []

    features = notes_to_features(master_notes)
    n = len(master_notes)
    master_map = {(round(note.onset, 2), note.pitch): i for i, note in enumerate(master_notes)}

    gts: List[np.ndarray] = []
    for path in paths:
        notes = parse_pig_file(path, hand_filter=hand)
        y = np.zeros(n, dtype=np.int64)
        for note in notes:
            key = (round(note.onset, 2), note.pitch)
            if key in master_map:
                f = int(abs(note.finger))
                if 1 <= f <= 5:
                    y[master_map[key]] = f
        gts.append(y)
    return features, gts


def gts_to_soft_targets(gts: List[np.ndarray], num_classes: int = NUM_FINGERS) -> Tuple[np.ndarray, np.ndarray]:
    """Return target_probs [L,K] and valid_mask [L]."""
    L = len(gts[0])
    counts = np.zeros((L, num_classes), dtype=np.float32)
    for gt in gts:
        for i, f in enumerate(gt):
            f = int(abs(f))
            if 1 <= f <= 5:
                counts[i, f] += 1.0
    valid = counts[:, 1:6].sum(axis=1) > 0
    targets = np.zeros_like(counts)
    denom = counts.sum(axis=1, keepdims=True)
    targets[valid] = counts[valid] / np.maximum(denom[valid], 1e-6)
    targets[:, 0] = 0.0
    return targets, valid.astype(np.float32)


class ConsensusWindowDataset(Dataset):
    def __init__(
        self,
        groups: Dict[int, List[Path]],
        hand: str,
        window: int = 32,
        stride: int = 16,
    ):
        self.samples = []
        self.window = int(window)
        self.stride = int(stride)
        self.hand = hand

        for pid, paths in sorted(groups.items()):
            alignment = align_multiple_ground_truths(paths, hand)
            if alignment[0] is None:
                continue
            feat, gts = alignment
            if not gts:
                continue
            target, valid = gts_to_soft_targets(gts)
            L = len(feat)
            if L == 0:
                continue

            starts = list(range(0, max(L - self.window + 1, 1), self.stride))
            if starts[-1] != max(0, L - self.window):
                starts.append(max(0, L - self.window))

            for st in starts:
                ed = min(st + self.window, L)
                f = feat[st:ed].copy()
                t = target[st:ed].copy()
                m = valid[st:ed].copy()
                cur = ed - st
                if cur < self.window:
                    f = np.concatenate([f, np.zeros((self.window - cur, feat.shape[1]), dtype=np.float32)], axis=0)
                    t = np.concatenate([t, np.zeros((self.window - cur, NUM_FINGERS), dtype=np.float32)], axis=0)
                    m = np.concatenate([m, np.zeros((self.window - cur,), dtype=np.float32)], axis=0)
                self.samples.append((f.astype(np.float32), t.astype(np.float32), m.astype(np.float32)))

        if not self.samples:
            raise RuntimeError("No training windows were built. Check pig_root/hand split.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        f, t, m = self.samples[idx]
        return {
            "features": torch.from_numpy(f),
            "target": torch.from_numpy(t),
            "mask": torch.from_numpy(m),
        }


def soft_ce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    loss = -(target * logp).sum(dim=-1)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    accs = []
    for batch in loader:
        features = batch["features"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        logits = model(features)
        loss = soft_ce_loss(logits, target, mask)
        losses.append(float(loss.item()))

        pred = logits.argmax(dim=-1)
        # For soft labels, use the majority finger as a simple validation proxy.
        maj = target.argmax(dim=-1)
        valid = mask > 0.5
        if valid.any():
            accs.append(float((pred[valid] == maj[valid]).float().mean().item()))
    return float(np.mean(losses)), float(np.mean(accs) if accs else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pig_root", default="data/pig")
    parser.add_argument("--hand", choices=["right", "left"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    pig_root = Path(args.pig_root)
    all_files = find_pig_files(pig_root)
    train_files = [f for f in all_files if get_piece_id(f) > 30]

    groups: Dict[int, List[Path]] = {}
    for f in train_files:
        groups.setdefault(get_piece_id(f), []).append(f)

    pids = sorted(groups.keys())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * args.val_ratio)) if args.val_ratio > 0 else 0
    val_pids = set(pids[:n_val])
    train_pids = set(pids[n_val:])

    train_groups = {pid: groups[pid] for pid in train_pids}
    val_groups = {pid: groups[pid] for pid in val_pids} if val_pids else {pid: groups[pid] for pid in train_pids}

    train_ds = ConsensusWindowDataset(train_groups, args.hand, args.window, args.stride)
    val_ds = ConsensusWindowDataset(val_groups, args.hand, args.window, args.stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConsensusEmissionTransformer(
        feat_dim=7,
        num_classes=NUM_FINGERS,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.window + 32,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data] hand={args.hand} train_pieces={len(train_groups)} val_pieces={len(val_groups)} train_windows={len(train_ds)} val_windows={len(val_ds)}")
    print(f"[model] params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features)
            loss = soft_ce_loss(logits, target, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:03d}] train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} val_majority_acc={val_acc*100:.2f}%")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "feat_dim": 7,
                    "num_classes": NUM_FINGERS,
                    "d_model": args.d_model,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dim_feedforward": args.dim_feedforward,
                    "dropout": args.dropout,
                    "max_len": args.window + 32,
                    "window": args.window,
                    "stride": args.stride,
                    "hand": args.hand,
                },
                "best_val_loss": best_val,
                "epoch": epoch,
            }, out_path)
            print(f"  [saved] {out_path} best_val_loss={best_val:.4f}")

    print(f"[done] best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
