from __future__ import annotations

import numpy as np
from fincov.data import parse_pig_file

class NakamuraHMM2:
    """
    Official 2nd-order Generative HMM (Nakamura et al. formulation).
    Transition (Interpolated): lambda_1 * P(f_n | f_{n-1}, f_{n-2}) + (1 - lambda_1) * P(f_n | f_{n-1})
    Emission (Factorized): P(p_n - p_{n-1} | f_{n-1}, f_n)**alpha_1 * P(p_n - p_{n-2} | f_{n-2}, f_n)**alpha_2
    """
    def __init__(self, alpha_1: float = 0.556, alpha_2: float = 0.407, lambda_1: float = 0.474, eps: float = 1e-4):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.eps = eps
        
        # Transition counts
        self.trans2_counts = np.full((5, 5, 5), self.eps)
        self.trans1_counts = np.full((5, 5), self.eps)
        
        # Emission counts
        self.emit1_counts = np.full((5, 5, 200), self.eps) # 1-step pitch difference
        self.emit2_counts = np.full((5, 5, 200), self.eps) # 2-step pitch difference
        
        self.init_counts = np.full((5, 5), self.eps)

    def fit_full_sequences(self, file_paths, hand: str = "right"):
        print(f"[NakamuraHMM2] Fitting transition and emission with alpha_1={self.alpha_1}, alpha_2={self.alpha_2}, lambda_1={self.lambda_1} (hand={hand})...")
        for path in file_paths:
            try:
                notes = parse_pig_file(path, hand_filter=hand)
            except Exception:
                continue
            if not notes: continue
            pitches = np.array([n.pitch for n in notes])
            fingers = np.array([n.finger for n in notes])
            
            for j in range(len(fingers)):
                f = fingers[j]
                if f < 1 or f > 5: continue
                f_idx = f - 1
                
                if j == 0 or fingers[j-1] < 1 or fingers[j-1] > 5:
                    continue
                elif j == 1 or fingers[j-2] < 1 or fingers[j-2] > 5:
                    self.init_counts[fingers[j-1] - 1, f_idx] += 1
                else:
                    f_prev1 = fingers[j-1] - 1
                    f_prev2 = fingers[j-2] - 1
                    
                    self.trans1_counts[f_prev1, f_idx] += 1
                    self.trans2_counts[f_prev2, f_prev1, f_idx] += 1
                    
                    # 1-step emission
                    pd1 = pitches[j] - pitches[j-1]
                    idx1 = pd1 + 100
                    if 0 <= idx1 < 200:
                        self.emit1_counts[f_prev1, f_idx, idx1] += 1
                        
                    # 2-step emission
                    pd2 = pitches[j] - pitches[j-2]
                    idx2 = pd2 + 100
                    if 0 <= idx2 < 200:
                        self.emit2_counts[f_prev2, f_idx, idx2] += 1
                        
        self.trans2_prob = self.trans2_counts / self.trans2_counts.sum(axis=2, keepdims=True)
        self.trans1_prob = self.trans1_counts / self.trans1_counts.sum(axis=1, keepdims=True)
        self.init_prob   = self.init_counts / self.init_counts.sum()
        
        self.emit1_prob = self.emit1_counts / self.emit1_counts.sum(axis=2, keepdims=True)
        self.emit2_prob = self.emit2_counts / self.emit2_counts.sum(axis=2, keepdims=True)
        
        self.log_init  = np.log(self.init_prob)
        self.log_emit1 = np.log(self.emit1_prob)
        self.log_emit2 = np.log(self.emit2_prob)

    def predict(self, features):
        feats = features.numpy()
        pitches = (feats[:, 0] * PITCH_RANGE + PITCH_MIN).astype(int)
        L = len(feats)
        if L < 2:
            return torch.ones(L, dtype=torch.long)
            
        dp = np.full((L, 5, 5), -np.inf)
        pointers = np.zeros((L, 5, 5), dtype=int)
        
        for f_prev in range(5):
            for f_curr in range(5):
                dp[1, f_prev, f_curr] = self.log_init[f_prev, f_curr]
                
        for i in range(2, L):
            pd1 = pitches[i] - pitches[i-1]
            idx1 = pd1 + 100
            if idx1 < 0 or idx1 >= 200: idx1 = 100
            
            pd2 = pitches[i] - pitches[i-2]
            idx2 = pd2 + 100
            if idx2 < 0 or idx2 >= 200: idx2 = 100
            
            for f_prev in range(5):
                for f_curr in range(5):
                    # Transition: linear interpolation
                    p_trans = self.lambda_1 * self.trans2_prob[:, f_prev, f_curr] + (1 - self.lambda_1) * self.trans1_prob[f_prev, f_curr]
                    log_p_trans = np.log(p_trans)
                    
                    # Emission: pairwise factorization
                    log_p_emit = self.alpha_1 * self.log_emit1[f_prev, f_curr, idx1] + self.alpha_2 * self.log_emit2[:, f_curr, idx2]
                    
                    scores = dp[i-1, :, f_prev] + log_p_trans + log_p_emit
                    best_prev2 = np.argmax(scores)
                    dp[i, f_prev, f_curr] = scores[best_prev2]
                    pointers[i, f_prev, f_curr] = best_prev2
                    
        pred = np.zeros(L, dtype=int)
        flat_last = dp[-1].flatten()
        best_flat = np.argmax(flat_last)
        pred[-1] = best_flat % 5
        pred[-2] = best_flat // 5
        
        for i in range(L-1, 1, -1):
            pred[i-2] = pointers[i, pred[i-1], pred[i]]
        return torch.tensor(pred + 1)

