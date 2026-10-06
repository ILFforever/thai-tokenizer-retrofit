"""Where trained models live.

Trained models are large (up to ~1 GB each) and can be rebuilt by the
training scripts, so they are kept outside the project folder, which syncs
to the cloud: ~/thaitok_models, or $THAITOK_MODELS. Commands and results
still name them "models/<folder>"; `model_dir` maps that name to the real
location.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS = Path(os.environ.get("THAITOK_MODELS", Path.home() / "thaitok_models"))


def model_dir(path: str | Path) -> Path:
    """"models/<folder>" -> MODELS/<folder>; any other relative path is
    relative to the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "models":
        return MODELS.joinpath(*p.parts[1:])
    return ROOT / p
