"""SAC agent (stable-baselines3 ``SAC``, ``MlpPolicy``) trained against the NPC field.

A thin wrapper over the SAC path of ``scripts/train_vs_npcs.py``: one expansion-masked learning
seat against the four self-check NPCs, homes shuffled each episode, argmax logit dims bounded to
[-1, 1], per-tick profit scaled by 1e-3, and the observation scale folded into the first layers so
the saved zip takes RAW observations (the scorer applies no VecNormalize). ``buffer_size`` is
capped at ``min(steps, 1e6)`` because the default 1M-transition buffer peaks near 11 GB.

Run from the repo root:
    python agents/Yavkata/sac/train_sac.py --steps 300000 --out agents/Yavkata/sac/checkpoint.zip
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from train_vs_npcs import main as train_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return train_main(["yavkata_sac", "--algo", "sac", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
