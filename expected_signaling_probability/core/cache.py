"""On-disk cache of per-draw values, one CSV (+ JSON sidecar) per (config, master seed).

Layout::

    data/runs/<slug>_seed=<master_seed>.csv    columns: index,value
    data/runs/<slug>_seed=<master_seed>.json   config dump + provenance

Rows are appended as they are computed, so an interrupted run loses at most
``flush_every`` draws and resumes where it stopped.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import qutip as qt

from expected_signaling_probability.core.config import DrawConfig

DEFAULT_RUNS_DIR = Path("data/runs")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


class RunCache:
    def __init__(self, cfg: DrawConfig, master_seed: int, runs_dir: Path = DEFAULT_RUNS_DIR, flush_every: int = 25):
        self.cfg = cfg
        self.master_seed = master_seed
        stem = f"{cfg.slug()}_seed={master_seed}"
        self.csv_path = runs_dir / f"{stem}.csv"
        self.json_path = runs_dir / f"{stem}.json"
        self.flush_every = flush_every
        self._values: dict[int, float] = {}
        self._pending = 0
        self._fh = None
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if not self.csv_path.exists():
            return
        with self.csv_path.open() as fh:
            header = fh.readline()
            assert header.strip() == "index,value", f"unexpected header in {self.csv_path}: {header!r}"
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                idx, val = line.split(",")
                self._values[int(idx)] = float(val)

    def _write_sidecar(self) -> None:
        if self.json_path.exists():
            return
        meta = {
            "config": self.cfg.to_dict(),
            "master_seed": self.master_seed,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "qutip": qt.__version__,
            "git_commit": _git_commit(),
        }
        self.json_path.write_text(json.dumps(meta, indent=2) + "\n")

    def _open(self) -> None:
        if self._fh is None:
            new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._fh = self.csv_path.open("a")
            if new_file:
                self._fh.write("index,value\n")
            self._write_sidecar()

    # ------------------------------------------------------------------ api
    def __contains__(self, index: int) -> bool:
        return index in self._values

    def get(self, index: int) -> float | None:
        return self._values.get(index)

    def set(self, index: int, value: float) -> None:
        self._open()
        assert self._fh is not None
        self._values[index] = value
        self._fh.write(f"{index},{value!r}\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
        self._pending = 0

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        self._pending = 0

    def __enter__(self) -> "RunCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._values)
