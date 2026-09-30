from __future__ import annotations

from typing import List
import numpy as np

PARNCUTT_COMF = {
    (1, 2): (-3, 8),
    (1, 3): (-2, 10),
    (1, 4): (-1, 12),
    (1, 5): ( 1, 13),
    (2, 3): ( 1, 3),
    (2, 4): ( 1, 5),
    (2, 5): ( 2, 8),
    (3, 4): ( 1, 2),
    (3, 5): ( 1, 5),
    (4, 5): ( 1, 3),
}


def parncutt_comfort_bounds(f_prev: int, f_cur: int) -> tuple[float, float]:
    """
    Return directional (MinComf, MaxComf) for an ordered RIGHT-HAND
    finger transition. For reversed finger order, use the sign-reversed
    bounds implied by the ordered interval definition.
    """
    if f_prev == f_cur:
        return (0.0, 0.0)

    if f_prev < f_cur:
        return PARNCUTT_COMF[(f_prev, f_cur)]

    lo, hi = PARNCUTT_COMF[(f_cur, f_prev)]
    return (-hi, -lo)


def transition_regularization_cost(
    f_prev: int,
    f_cur: int,
    pitch_diff: int,
    features_i: np.ndarray,
    hand: str = "right",
) -> float:
    """
    Transition regularizer with the Parncutt et al. (1997) Stretch Rule.

    For different fingers, a signed pitch interval outside the comfortable
    range [MinComf, MaxComf] receives 2 points per excess semitone.
    The published table is for the right hand; for the left hand the pitch
    axis is mirrored before applying the same finger-relative geometry.

    The existing same-finger and fast-large-jump terms are retained so that
    this experiment changes the finger-span component only.
    """
    pd = int(pitch_diff)
    abs_pd = abs(pd)
    rest_gap = bool(features_i[13] > 0.5) if features_i.shape[0] > 13 else False
    same_onset = bool(features_i[14] > 0.5) if features_i.shape[0] > 14 else False

    penalty = 0.0

    if f_prev == f_cur:
        if abs_pd > 0 and not rest_gap and not same_onset:
            penalty += 2.0
    else:
        # Parncutt Table 1 is specified for RH. Mirror pitch direction for LH.
        ergonomic_interval = pd if hand == "right" else -pd
        min_comf, max_comf = parncutt_comfort_bounds(f_prev, f_cur)

        if ergonomic_interval > max_comf:
            penalty += 2.0 * (ergonomic_interval - max_comf)
        elif ergonomic_interval < min_comf:
            penalty += 2.0 * (min_comf - ergonomic_interval)

    # Keep the original auxiliary timing term unchanged.
    ioi_norm = float(features_i[4])
    if abs_pd > 12 and ioi_norm < 0.08:
        penalty += 1.0

    return penalty


def viterbi_hmm1_prior(
    emission_logits: np.ndarray,
    features_np: np.ndarray,
    hmm1: NakamuraHMM1,
    trans_weight: float = 1.0,
    emit_weight: float = 1.0,
    play_weight: float = 0.3,
    hand: str = "right",
) -> np.ndarray:
    """
    First-order Viterbi:
      neural emission + HMM1 transition/emission prior + playability penalty.

    emission_logits: [L,6], only classes 1..5 are used.
    returns pred [L] in {1..5}.
    """
    L = emission_logits.shape[0]
    if L == 0:
        return np.zeros(0, dtype=np.int64)

    logits = emission_logits[:, 1:6].astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    neural_logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-12)

    pitches = np.rint(features_np[:, 0] * 88 + 21).astype(np.int32)

    dp = np.full((L, 5), -1e18, dtype=np.float64)
    ptr = np.zeros((L, 5), dtype=np.int64)

    init = getattr(hmm1, "log_init", np.zeros(5))
    dp[0, :] = emit_weight * neural_logp[0, :] + trans_weight * init

    for i in range(1, L):
        pd = int(pitches[i] - pitches[i - 1])
        idx = pd + 100
        if idx < 0 or idx >= 200:
            idx = 100

        for f_idx in range(5):
            f_cur = f_idx + 1
            best_score = -1e18
            best_prev = 0

            for fp_idx in range(5):
                f_prev = fp_idx + 1

                hmm_score = 0.0
                hmm_score += float(hmm1.log_trans[fp_idx, f_idx])
                hmm_score += float(hmm1.alpha_1 * hmm1.log_emit[fp_idx, f_idx, idx])

                penalty = transition_regularization_cost(f_prev, f_cur, pd, features_np[i], hand=hand)

                score = (
                    dp[i - 1, fp_idx]
                    + trans_weight * hmm_score
                    + emit_weight * neural_logp[i, f_idx]
                    - play_weight * penalty
                )

                if score > best_score:
                    best_score = score
                    best_prev = fp_idx

            dp[i, f_idx] = best_score
            ptr[i, f_idx] = best_prev

    pred_idx = np.zeros(L, dtype=np.int64)
    pred_idx[-1] = int(np.argmax(dp[-1]))

    for i in range(L - 1, 0, -1):
        pred_idx[i - 1] = ptr[i, pred_idx[i]]

    return pred_idx + 1



def viterbi_hmm2_prior(
    emission_logits: np.ndarray,
    features_np: np.ndarray,
    hmm2: NakamuraHMM2,
    trans_weight: float = 1.0,
    emit_weight: float = 1.0,
    play_weight: float = 0.3,
    hand: str = "right",
) -> np.ndarray:
    """
    Second-order Viterbi:
      neural emission + HMM2 transition/emission prior + playability penalty.

    State is (f_{i-1}, f_i). This better matches Nakamura HMM2.
    """
    L = emission_logits.shape[0]
    if L == 0:
        return np.zeros(0, dtype=np.int64)
    if L == 1:
        return np.array([int(np.argmax(emission_logits[0, 1:6])) + 1], dtype=np.int64)

    logits = emission_logits[:, 1:6].astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    neural_logp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-12)

    pitches = np.rint(features_np[:, 0] * 88 + 21).astype(np.int32)

    dp = np.full((L, 5, 5), -1e18, dtype=np.float64)
    ptr = np.zeros((L, 5, 5), dtype=np.int64)

    # initialize pair at i=1
    log_init = getattr(hmm2, "log_init", np.zeros((5, 5)))
    for f0 in range(5):
        for f1 in range(5):
            pd1 = int(pitches[1] - pitches[0])
            penalty = transition_regularization_cost(f0 + 1, f1 + 1, pd1, features_np[1], hand=hand)
            dp[1, f0, f1] = (
                emit_weight * (neural_logp[0, f0] + neural_logp[1, f1])
                + trans_weight * float(log_init[f0, f1])
                - play_weight * penalty
            )

    for i in range(2, L):
        pd1 = int(pitches[i] - pitches[i - 1])
        idx1 = pd1 + 100
        if idx1 < 0 or idx1 >= 200:
            idx1 = 100

        pd2 = int(pitches[i] - pitches[i - 2])
        idx2 = pd2 + 100
        if idx2 < 0 or idx2 >= 200:
            idx2 = 100

        for f_prev in range(5):
            for f_cur in range(5):
                best_score = -1e18
                best_prev2 = 0

                for f_prev2 in range(5):
                    # HMM2 transition interpolation.
                    p_trans = (
                        hmm2.lambda_1 * hmm2.trans2_prob[f_prev2, f_prev, f_cur]
                        + (1.0 - hmm2.lambda_1) * hmm2.trans1_prob[f_prev, f_cur]
                    )
                    hmm_score = np.log(float(p_trans) + 1e-12)

                    hmm_score += float(hmm2.alpha_1 * hmm2.log_emit1[f_prev, f_cur, idx1])
                    hmm_score += float(hmm2.alpha_2 * hmm2.log_emit2[f_prev2, f_cur, idx2])

                    penalty = transition_regularization_cost(
                        f_prev + 1, f_cur + 1, pd1, features_np[i], hand=hand
                    )

                    score = (
                        dp[i - 1, f_prev2, f_prev]
                        + trans_weight * hmm_score
                        + emit_weight * neural_logp[i, f_cur]
                        - play_weight * penalty
                    )

                    if score > best_score:
                        best_score = score
                        best_prev2 = f_prev2

                dp[i, f_prev, f_cur] = best_score
                ptr[i, f_prev, f_cur] = best_prev2

    pred_idx = np.zeros(L, dtype=np.int64)
    best_flat = int(np.argmax(dp[-1].reshape(-1)))
    pred_idx[-2] = best_flat // 5
    pred_idx[-1] = best_flat % 5

    for i in range(L - 1, 1, -1):
        pred_idx[i - 2] = ptr[i, pred_idx[i - 1], pred_idx[i]]

    return pred_idx + 1



def decode_topk(
    base_emission: np.ndarray,
    features_np: np.ndarray,
    hmm2,
    hand: str,
    top_k: int = 1,
    lambda_hmm: float = 1.0,
    lambda_emit: float = 1.0,
    lambda_trans: float = 0.3,
    diversity_penalty: float = 1.0,
) -> List[np.ndarray]:
    """FinCoV structured decoding. Candidate 1 is unchanged for every top_k."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    L = len(features_np)
    used_counts = np.zeros((L, 6), dtype=np.float32)
    candidates = []

    for rank in range(top_k):
        emission = base_emission.copy()
        if rank > 0:
            emission -= diversity_penalty * used_counts
            emission[:, 0] = -1e9

        pred = viterbi_hmm2_prior(
            emission_logits=emission,
            features_np=features_np,
            hmm2=hmm2,
            trans_weight=lambda_hmm,
            emit_weight=lambda_emit,
            play_weight=lambda_trans,
            hand=hand,
        )
        candidates.append(pred)
        used_counts[np.arange(L), pred] += 1.0

    return candidates
