# FinCoV

**FinCoV: Multi-Expert Consensus Learning and Diversity-Aware Structured Decoding for Automatic Piano Fingering**

## Overview

Automatic piano fingering does not necessarily have a single correct solution,
as different experts may assign different fingerings to the same score.
FinCoV formulates automatic piano fingering as a **multi-reference structured
prediction problem** and addresses three aspects of piano fingering:

1. incorporating **different expert fingering choices**,
2. maintaining **sequence-level consistency**, and
3. generating **multiple distinct fingering alternatives**.

FinCoV consists of three modules:

- **Consensus Learning Module**  
  Represents multiple expert annotations as note-level finger distributions
  and learns consensus emission probabilities.

- **Fingering Generation Module**  
  Combines the learned consensus emissions with HMM transition statistics and
  transition regularization, and uses Viterbi decoding to generate a globally
  coherent fingering sequence.

- **Diverse Fingering Generation Module**  
  Repeatedly performs structured decoding with an accumulated overlap penalty
  to generate multiple distinct fingering alternatives.

## Architecture

<p align="center">
  <img src="fig/frame.png" width="100%">
</p>

## Data Preparation

Please download the PIG dataset from its official source.

Following the official PIG split, 120 pieces from the miscellaneous subset
are used for training, while 30 pieces from the Bach, Mozart, and Chopin
subsets are used for testing.

The dataset directory should be organized as follows:

```text
data/
└── pig/
    └── ...

## Results

### PIG Benchmark

We evaluate FinCoV on the official PIG test split using the four
multi-reference annotation-matching metrics:
$M_{\mathrm{gen}}$, $M_{\mathrm{high}}$, $M_{\mathrm{soft}}$, and
$M_{\mathrm{rec}}$.

| Hand | M_gen ↑ | M_high ↑ | M_soft ↑ | M_rec ↑ |
|------|--------:|---------:|---------:|--------:|
| Right | 65.75 | 73.04 | 89.84 | 81.88 |
| Left | 70.69 | 77.31 | 89.34 | 83.05 |
| **Both** | **68.22** | **75.18** | **89.59** | **82.46** |

The reported results are averaged over three random seeds.

### Diverse Fingering Generation

FinCoV can generate multiple distinct fingering sequences for the same
musical score. With Top-10 decoding, the generated candidate set achieves:

| Metric | Result |
|--------|-------:|
| Top-1 $M_{\mathrm{gen}}$ | 68.22 |
| Top-10 Oracle $M_{\mathrm{gen}}$ | 68.83 |
| Average Pairwise Diversity | 59.11% |
| Unique Rate | 100.00% |

The Top-K oracle is used only for evaluation and selects the candidate with
the highest reference agreement after generation.

## Training

Train the Consensus Learning Module separately for each hand:

```bash
python train.py \
  --hand right \
  --pig_root data/pig \
  --device cuda \
  --out output/consensus_right/best.pt
