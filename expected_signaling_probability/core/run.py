"""Cache-aware Monte Carlo runner."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from expected_signaling_probability.core.cache import DEFAULT_RUNS_DIR, RunCache
from expected_signaling_probability.core.config import DrawConfig
from expected_signaling_probability.core.draw import draw
from expected_signaling_probability.core.samples import Samples


def run(cfg: DrawConfig, n: int, master_seed: int = 0, runs_dir: Path = DEFAULT_RUNS_DIR, use_cache: bool = True) -> Samples:
    """Return ``n`` draws (indices ``0..n-1``) for ``cfg``, computing only the ones not cached."""
    cache = RunCache(cfg, master_seed, runs_dir) if use_cache else None
    values = np.empty(n)
    todo = [i for i in range(n) if cache is None or i not in cache]

    try:
        for i in tqdm(todo, desc=f"{cfg.slug()} seed={master_seed}", leave=False, disable=not todo, mininterval=1.0, ascii=True):
            value = draw(cfg, master_seed, i)
            if cache is not None:
                cache.set(i, value)
            values[i] = value
        if cache is not None:
            for i in range(n):
                if i not in todo:
                    values[i] = cache.get(i)  # type: ignore[assignment]
    finally:
        if cache is not None:
            cache.close()

    return Samples(config=cfg, master_seed=master_seed, values=values)


def sweep(cfgs: Iterable[DrawConfig], n: int, master_seed: int = 0, runs_dir: Path = DEFAULT_RUNS_DIR, use_cache: bool = True) -> list[Samples]:
    return [run(cfg, n, master_seed=master_seed, runs_dir=runs_dir, use_cache=use_cache) for cfg in cfgs]
