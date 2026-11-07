import json

import numpy as np

from expected_signaling_probability import DrawConfig, run, summary_table
from expected_signaling_probability.core.cache import RunCache


def test_run_caches_and_resumes(tmp_path):
    cfg = DrawConfig(d_A=2, d_B=2)
    s1 = run(cfg, 5, master_seed=0, runs_dir=tmp_path)
    s2 = run(cfg, 8, master_seed=0, runs_dir=tmp_path)  # extends, reuses first 5
    assert np.allclose(s1.values, s2.values[:5])
    assert s2.n == 8

    cache = RunCache(cfg, 0, tmp_path)
    assert len(cache) == 8
    meta = json.loads(cache.json_path.read_text())
    assert meta["config"]["d_A"] == 2 and meta["master_seed"] == 0

    fresh = run(cfg, 8, master_seed=0, runs_dir=tmp_path, use_cache=False)
    assert np.allclose(fresh.values, s2.values)

    other_seed = run(cfg, 8, master_seed=1, runs_dir=tmp_path)
    assert not np.allclose(other_seed.values, s2.values)


def test_samples_statistics(tmp_path):
    cfg = DrawConfig(d_A=2, d_B=2)
    s = run(cfg, 10, runs_dir=tmp_path)
    assert abs(s.sem - np.std(s.values, ddof=1) / np.sqrt(10)) < 1e-12
    assert s.q25 <= s.median <= s.q75
    df = summary_table([s])
    assert set(["mean", "sem", "median", "q25", "q75", "n", "d_A"]) <= set(df.columns)
