from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from fincov.data import NUM_FINGERS, parse_pig_file, notes_to_features
from fincov.consensus import ConsensusEmissionTransformer
from fincov.hmm import NakamuraHMM2
from fincov.metrics import compute_nakamura_metrics
from fincov.decoder import decode_topk

WINDOW_SIZE = 32
INFERENCE_STRIDE = 8
LAMBDA_HMM = 1.0
LAMBDA_EMIT = 1.0
LAMBDA_TRANS = 0.3
DIVERSITY_PENALTY = 1.0


def get_piece_id(path: Path) -> int:
    m = re.match(r"(\d+)-", path.name)
    return int(m.group(1)) if m else 999


def find_pig_files(pig_root: Path) -> List[Path]:
    pat = re.compile(r"\d+-\d+_fingering\.txt$", re.I)
    out = []
    for root, _, files in os.walk(pig_root):
        for name in sorted(files):
            if pat.match(name):
                out.append(Path(root) / name)
    return out


def align_multiple_ground_truths(paths: List[Path], hand: str) -> Tuple[np.ndarray | None, List[np.ndarray]]:
    master = parse_pig_file(paths[0], hand_filter=hand)
    if not master:
        return None, []
    features = notes_to_features(master)
    index = {(round(n.onset, 2), n.pitch): i for i, n in enumerate(master)}
    gts = []
    for path in paths:
        y = np.zeros(len(master), dtype=np.int64)
        for n in parse_pig_file(path, hand_filter=hand):
            j = index.get((round(n.onset, 2), n.pitch))
            if j is not None and 1 <= abs(int(n.finger)) <= 5:
                y[j] = abs(int(n.finger))
        gts.append(y)
    return features, gts


def load_consensus(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ConsensusEmissionTransformer(
        feat_dim=int(cfg.get("feat_dim", 7)),
        num_classes=int(cfg.get("num_classes", NUM_FINGERS)),
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 4)),
        dim_feedforward=int(cfg.get("dim_feedforward", 1024)),
        dropout=0.0,
        max_len=int(cfg.get("max_len", 1024)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def consensus_log_emissions(model, features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    _, L, _ = x.shape
    accum = torch.zeros(1, L, NUM_FINGERS, device=device)
    count = torch.zeros(1, L, 1, device=device)
    seen = set()

    for start in range(0, L, INFERENCE_STRIDE):
        end = min(start + WINDOW_SIZE, L)
        start = max(0, end - WINDOW_SIZE)
        if (start, end) in seen:
            continue
        seen.add((start, end))
        prob = model(x[:, start:end]).softmax(dim=-1)
        prob[:, :, 0] = 0.0
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        accum[:, start:end] += prob
        count[:, start:end] += 1.0

    avg = accum / count.clamp(min=1.0)
    logp = torch.log(avg.clamp(min=1e-12))
    logp[:, :, 0] = -1e9
    return logp[0].cpu().numpy().astype(np.float32)


def pairwise_diversity(candidates):
    if len(candidates) < 2:
        return 0.0
    vals = [np.mean(candidates[i] != candidates[j])
            for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser(description="FinCoV PIG evaluation and Top-K fingering generation")
    p.add_argument("--hand", required=True, choices=["right", "left"])
    p.add_argument("--pig_root", default="data/pig")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--top_k", type=int, default=1)
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top_k must be >= 1")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    files = find_pig_files(Path(args.pig_root))
    train_files = [f for f in files if get_piece_id(f) > 30]
    test_files = [f for f in files if get_piece_id(f) <= 30]
    groups = {}
    for f in test_files:
        groups.setdefault(get_piece_id(f), []).append(f)

    hmm = NakamuraHMM2(alpha_1=0.556, alpha_2=0.407, lambda_1=0.474, eps=1e-4)
    hmm.fit_full_sequences(train_files, hand=args.hand)
    model = load_consensus(Path(args.checkpoint), device)

    rank_values = [{m: [] for m in ("M_gen","M_high","M_soft","M_rec")} for _ in range(args.top_k)]
    divs, uniques, oracles = [], [], []

    for _, paths in sorted(groups.items()):
        features, gts = align_multiple_ground_truths(paths, args.hand)
        if features is None:
            continue
        emission = consensus_log_emissions(model, features, device)
        candidates = decode_topk(
            emission, features, hmm, args.hand, args.top_k,
            LAMBDA_HMM, LAMBDA_EMIT, LAMBDA_TRANS, DIVERSITY_PENALTY
        )

        piece_mgen = []
        for r, pred in enumerate(candidates):
            met = compute_nakamura_metrics(pred, gts)
            piece_mgen.append(met["M_gen"])
            for name in rank_values[r]:
                rank_values[r][name].append(met[name])

        if args.top_k > 1:
            divs.append(pairwise_diversity(candidates))
            uniques.append(len({tuple(x.tolist()) for x in candidates}) / args.top_k)
            oracles.append(max(piece_mgen))

    print("\nRank | M_gen | M_high | M_soft | M_rec")
    for r, vals in enumerate(rank_values, 1):
        x = [100*np.mean(vals[m]) for m in ("M_gen","M_high","M_soft","M_rec")]
        print(f"{r:>4} | {x[0]:6.2f} | {x[1]:6.2f} | {x[2]:6.2f} | {x[3]:6.2f}")

    if args.top_k > 1:
        top1 = 100*np.mean(rank_values[0]["M_gen"])
        oracle = 100*np.mean(oracles)
        print(f"\nPairwise diversity : {100*np.mean(divs):.2f}%")
        print(f"Unique rate        : {100*np.mean(uniques):.2f}%")
        print(f"Top-{args.top_k} oracle M_gen: {oracle:.2f}%")
        print(f"Top-1 M_gen        : {top1:.2f}%")
        print(f"Oracle gain        : {oracle-top1:+.2f} pp")


if __name__ == "__main__":
    main()
