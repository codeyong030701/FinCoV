import numpy as np
import torch
from typing import List, Dict

def compute_nakamura_metrics(pred: np.ndarray, gts: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes Nakamura et al. (2020) official 4 match metrics under multiple ground truths,
    with robust skip-masking support (ignores padding / unannotated value 0 in specific annotators).
    
    Args:
        pred: 1D numpy array of estimated fingerings [N] (values 1-5)
        gts: list of 1D numpy arrays, each being one pianist's ground truth fingering [N]
             (value 0 means this pianist did not annotate this note - will be skipped)
        
    Returns:
        dictionary with keys: M_gen, M_high, M_soft, M_rec
    """
    N = len(pred)
    R = len(gts)
    if N == 0 or R == 0:
        return {"M_gen": 0.0, "M_high": 0.0, "M_soft": 0.0, "M_rec": 0.0}
    
    # 1. M_gen: Average match rate over all ground truths (skipping 0s)
    gen_accs = []
    for gt in gts:
        valid_mask = (gt >= 1) & (gt <= 5)
        if np.sum(valid_mask) > 0:
            acc = np.mean(pred[valid_mask] == gt[valid_mask])
            gen_accs.append(acc)
        else:
            gen_accs.append(0.0)
    m_gen = np.mean(gen_accs)
    
    # 2. M_high: Highest match rate among all ground truths
    m_high = np.max(gen_accs)
    
    # 3. M_soft: Soft-match rate (skipping notes where no annotator has annotations)
    soft_matches = 0
    total_valid_notes = 0
    for i in range(N):
        valid_annotators = [gt[i] for gt in gts if gt[i] >= 1 and gt[i] <= 5]
        if len(valid_annotators) > 0:
            total_valid_notes += 1
            if any(pred[i] == val for val in valid_annotators):
                soft_matches += 1
    m_soft = (soft_matches / total_valid_notes) if total_valid_notes > 0 else 0.0
    
    # 4. M_rec: Recombination Match Rate via DP, skipping 0 markers
    # dp[i, r] = minimum cost up to index i, ending by referring to pianist r.
    dp = np.zeros((N, R), dtype=np.float32)
    
    # Base case (i=0)
    for r in range(R):
        if gts[r][0] >= 1 and gts[r][0] <= 5:
            dp[0, r] = 0.0 if pred[0] == gts[r][0] else 1.0
        else:
            dp[0, r] = 1e9 # Block invalid starting state
            
    for i in range(1, N):
        for r in range(R):
            if gts[r][i] >= 1 and gts[r][i] <= 5:
                cost_sub = 0.0 if pred[i] == gts[r][i] else 1.0
            else:
                cost_sub = 1e9 # Block invalid state
                
            min_prev = float('inf')
            for r_prev in range(R):
                transition_penalty = 0.0 if r == r_prev else 1.0
                total_prev_cost = dp[i-1, r_prev] + transition_penalty
                if total_prev_cost < min_prev:
                    min_prev = total_prev_cost
                    
            dp[i, r] = min_prev + cost_sub
            
    min_total_error = np.min(dp[N-1])
    m_rec = max(0.0, (N - min_total_error) / N) if N > 0 else 0.0
    
    return {
        "M_gen": float(m_gen),
        "M_high": float(m_high),
        "M_soft": float(m_soft),
        "M_rec": float(m_rec)
    }

