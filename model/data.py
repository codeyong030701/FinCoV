"""
piano/dataset.py  –  PIG (PIano fingerinG) dataset loader.

PIG dataset format (each hand, each piece, each annotator):
  Each line: OnsetTime OffsetTime Pitch FingerNumber
  FingerNumber: 1-5 (thumb=1 … pinky=5)
  Files: <piece_id>-<annotator_id>_fingering.txt  (right/left separate)

Directory layout expected:
  data/pig/
    ├── PianoFingeringDataset_v1.02/
    │   ├── fingering_anno/
    │   │   ├── 001-1_fingering.txt
    │   │   └── ...
    │   └── ...

Usage:
    from fincov.dataset import PigDataset
    ds = PigDataset("data/pig", hand="right", window=32)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
FINGER_PAD  = 0   # padding / unknown token
FINGER_MIN  = 1   # thumb
FINGER_MAX  = 5   # pinky
NUM_FINGERS = 6   # 0=pad, 1-5=fingers  (vocab size)

PITCH_MIN   = 21   # A0
PITCH_MAX   = 108  # C8
PITCH_RANGE = PITCH_MAX - PITCH_MIN + 1  # 88

MAX_IOI     = 4.0  # seconds – inter-onset interval clamp
MAX_DUR     = 4.0  # seconds – note duration clamp


# ── Note dataclass ────────────────────────────────────────────────────────────
@dataclass
class Note:
    onset:  float   # seconds
    offset: float   # seconds
    pitch:  int     # MIDI pitch (21-108)
    finger: int     # 1-5  (0 = unknown)

    @property
    def duration(self) -> float:
        return self.offset - self.onset


# ── PIG file parser ───────────────────────────────────────────────────────────
def note_name_to_midi(name: str) -> Optional[int]:
    """Convert PIG note name (e.g., C4, F#3, Bb4, Eb5) to MIDI integer pitch."""
    try:
        sharp_map = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                     "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                     "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}
        m = re.match(r"([A-G][#b]?)(\d+)", name)
        if not m:
            return None
        return (int(m.group(2)) + 1) * 12 + sharp_map[m.group(1)]
    except Exception:
        return None

def parse_pig_file(path: Path, hand_filter: Optional[str] = None) -> List[Note]:
    """
    Parse a single PIG v1.2 annotation file.

    Column layout (tab-separated):
      0:idx  1:onset  2:offset  3:NoteName  4:vel  5:?  6:hand  7:finger

      hand:   0 = right hand,  1 = left hand
      finger: positive = right hand (1-5), negative = left hand (-1 to -5)
              special: "4_1" means finger substitution → take first value

    Args:
        path:        path to fingering .txt file
        hand_filter: "right" | "left" | None (both)
    """
    notes: List[Note] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                onset   = float(parts[1])
                offset  = float(parts[2])
                pitch   = note_name_to_midi(parts[3])
                if pitch is None:
                    continue
                hand    = int(parts[6])   # 0=right, 1=left

                # Apply hand filter
                if hand_filter == "right" and hand != 0:
                    continue
                if hand_filter == "left" and hand != 1:
                    continue

                # Finger: may be "4_1" (substitution) → take first digit
                finger_str = parts[7].split("_")[0]
                finger  = abs(int(finger_str))
                if finger < FINGER_MIN or finger > FINGER_MAX:
                    finger = FINGER_PAD
                notes.append(Note(onset, offset, pitch, finger))
            except (ValueError, IndexError):
                continue
    notes.sort(key=lambda n: n.onset)
    return notes


# ── Feature extraction ────────────────────────────────────────────────────────
FINCOV_FEAT_DIM = 7

def notes_to_features(notes):
    """Exact 7-D FinCoV note representation used in the paper."""
    N = len(notes)
    feat = np.zeros((N, FINCOV_FEAT_DIM), dtype=np.float32)
    BLACK_KEYS = {1, 3, 6, 8, 10}

    for i, note in enumerate(notes):
        pc = note.pitch % 12
        prev = notes[i - 1] if i > 0 else None
        ioi = note.onset - prev.onset if prev is not None else 0.0

        feat[i, 0] = (note.pitch - PITCH_MIN) / PITCH_RANGE
        feat[i, 1] = pc / 11.0
        feat[i, 2] = (note.pitch // 12 - 1) / 8.0
        feat[i, 3] = min(max(note.duration, 0.0), MAX_DUR) / MAX_DUR
        feat[i, 4] = min(max(ioi, 0.0), MAX_IOI) / MAX_IOI
        feat[i, 5] = 1.0 if pc in BLACK_KEYS else 0.0
        feat[i, 6] = i / max(N - 1, 1)

    return feat

class PigDataset(Dataset):
    """
    Sliding-window dataset over PIG fingering sequences.

    Args:
        pig_root:  path to PIG dataset root (contains fingering_anno/ or similar)
        hand:      "right" | "left" | "both"
        window:    number of notes per sample
        stride:    sliding window stride (default = window // 2)
        augment:   if True, randomly transpose pitch at train time
    """

    def __init__(
        self,
        pig_root: str | Path,
        hand: str = "right",
        window: int = 32,
        stride: Optional[int] = None,
        augment: bool = False,
    ):
        self.pig_root = Path(pig_root)
        self.hand     = hand.lower()
        self.window   = window
        self.stride   = stride if stride is not None else window // 2
        self.augment  = augment

        # Gather all annotation files
        self._files: List[Path] = self._find_files()
        if not self._files:
            raise FileNotFoundError(
                f"No PIG annotation files found under {self.pig_root}.\n"
                f"Download PIG dataset and place it at: {self.pig_root}\n"
                f"See: https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/"
            )

        # Pre-load all sequences → list of (features, fingers)
        self._samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build_samples()

        print(
            f"[PigDataset] hand={hand}  files={len(self._files)}  "
            f"window={window}  samples={len(self._samples)}"
        )

    def _find_files(self) -> List[Path]:
        """Recursively find PIG annotation .txt files.
        Hand filtering is done inside parse_pig_file, not by filename.
        """
        pattern = re.compile(r"\d+-\d+_fingering\.txt$", re.IGNORECASE)
        files: List[Path] = []
        for root, _, fnames in os.walk(self.pig_root):
            for fname in sorted(fnames):
                if pattern.match(fname):
                    files.append(Path(root) / fname)
        return files

    def _build_samples(self):
        """Slide window over every piece and collect (feat, finger) pairs."""
        hand_filter = None if self.hand == "both" else self.hand
        for path in self._files:
            try:
                notes = parse_pig_file(path, hand_filter=hand_filter)
            except Exception as e:
                print(f"[warn] skip {path}: {e}")
                continue
            if not notes:
                continue

            # Skip if too many unknown fingers
            valid = sum(1 for n in notes if n.finger != FINGER_PAD)
            if valid < len(notes) * 0.5:
                continue

            feat    = notes_to_features(notes)                 # [N, 7]
            fingers = np.array([n.finger for n in notes],      # [N]
                               dtype=np.int64)
            N = len(notes)
            if N < self.window:
                # pad
                pad_feat = np.zeros((self.window - N, feat.shape[1]), dtype=np.float32)
                pad_fing = np.zeros(self.window - N, dtype=np.int64)
                feat    = np.concatenate([feat, pad_feat], axis=0)
                fingers = np.concatenate([fingers, pad_fing], axis=0)
                self._samples.append((feat, fingers))
            else:
                for start in range(0, N - self.window + 1, self.stride):
                    self._samples.append((
                        feat[start:start + self.window].copy(),
                        fingers[start:start + self.window].copy(),
                    ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        feat, fingers = self._samples[idx]
        feat    = torch.from_numpy(feat)       # [W, 7]
        fingers = torch.from_numpy(fingers)    # [W]  int64, values 0-5

        if self.augment:
            # Random pitch transpose ±6 semitones (clamp to valid range)
            shift = torch.randint(-6, 7, (1,)).item()
            feat[:, 0] = (feat[:, 0] * PITCH_RANGE + PITCH_MIN + shift
                          - PITCH_MIN).clamp(0, PITCH_RANGE) / PITCH_RANGE
            feat[:, 1] = ((feat[:, 1] * 11 + shift) % 12) / 11.0

        return {
            "features": feat,       # [W, 7]  float32
            "fingers":  fingers,    # [W]     int64
        }


# ── Utility: collate for DataLoader ──────────────────────────────────────────
def collate_fn(batch):
    features = torch.stack([b["features"] for b in batch])  # [B, W, 7]
    fingers  = torch.stack([b["fingers"]  for b in batch])  # [B, W]
    return {"features": features, "fingers": fingers}


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/pig"
    ds   = PigDataset(root, hand="right", window=32)
    sample = ds[0]
    print(f"features shape: {sample['features'].shape}")   # [32, 7]
    print(f"fingers  shape: {sample['fingers'].shape}")    # [32]
    print(f"fingers  values: {sample['fingers'].tolist()}")
