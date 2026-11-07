# Expected Signaling Probability

$$
\mathcal{S}_{A \to B} =
D_{\text{Tr}}\left(
\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB}) \right],
\text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB})) \right]
\right),
\qquad
\langle \mathcal{S} \rangle_{A \to B} =
\underset{\rho, \mathcal{E}, \mathcal{A}}{\mathbb{E}}
\Big[ \mathcal{S}_{A \to B} \Big]
$$

## Getting started
```
pip install uv
uv sync
uv run pytest
uv run expected_signaling_probability/experiments/headline_symmetric.py
uv run expected_signaling_probability/experiments/headline_asymmetric.py
uv run expected_signaling_probability/experiments/robustness.py
```

## API

A `DrawConfig` fully declares one Monte Carlo draw (up to the seed):

```python
from expected_signaling_probability import *

cfg = DrawConfig(
    d_A=8, d_B=2,
    direction=Direction.A_TO_B,
    state=StateEnsemble.HAAR_PURE,          # HAAR_PURE | HS_MIXED | PRODUCT_HS
    local_op=LocalOpEnsemble.HAAR_UNITARY,  # HAAR_UNITARY | BCSZ
    global_op=GlobalOpEnsemble.LUDERS,      # HAAR_UNITARY | LUDERS | BCSZ
    global_rank=None,                       # Kraus rank, BCSZ only; None = full
    quantity=Quantity.SIGNALING,            # SIGNALING (S) | TRANSMISSION (T, erasure baseline)
)
samples = run(cfg, n=1000, master_seed=0)   # cached, resumable
print(samples)                              # mean ± sem, median, IQR
samples = sweep([cfg.with_(d_A=d) for d in range(2, 33)], n=1000)
```

### Seeding
Draw `i` of a run uses `SeedSequence([master_seed, i, SCHEME_VERSION]).spawn(4)`; the
state, the second state (for `T`), the local operation and the global operation
each get their own child generator, so they are statistically independent.
Bump `SCHEME_VERSION` in `core/config.py` whenever the meaning of a draw changes;
it is part of every cache key.

### Cache
One file pair per `(config, master_seed)` under `data/runs/`:
`<slug>_seed=<master_seed>.csv` (columns `index,value`, appended as computed) and a
`.json` sidecar with the config, qutip version and git commit. `run(cfg, n)` computes
only indices not yet on disk, so extending `n` or resuming an interrupted run is free.

### Ensembles (`core/ensembles.py`)
| | ensemble | definition |
|---|---|---|
| state | `HAAR_PURE` | Haar random pure state on $\mathcal H_A \otimes \mathcal H_B$ |
| | `HS_MIXED` | Hilbert–Schmidt measure (full-rank Ginibre) |
| | `PRODUCT_HS` | $\rho_A \otimes \rho_B$, each Hilbert–Schmidt |
| local op | `HAAR_UNITARY` | Haar unitary on the sender's factor |
| | `BCSZ` | full-rank BCSZ CPTP map on the sender's factor |
| global op | `HAAR_UNITARY` | Haar unitary on $\mathcal H_A \otimes \mathcal H_B$ (Kraus rank 1) |
| | `LUDERS` | $\rho \mapsto \sum_k P_k \rho P_k$, rank-1 projectors onto a Haar random basis of $\mathcal H_A \otimes \mathcal H_B$ |
| | `BCSZ` | BCSZ CPTP map on $\mathcal H_A \otimes \mathcal H_B$ with Kraus rank `global_rank` |

## Main package — [QuTiP](https://qutip.org/citing.html)
```
@misc{lambert2024qutip,
    title={QuTiP 5: The Quantum Toolbox in Python},
    author={Neill Lambert and Eric Giguère and Paul Menczel and Boxi Li and Patrick Hopf and Gerardo Suárez and Marc Gali and Jake Lishman and Rushiraj Gadhvi and Rochisha Agarwal and Asier Galicia and Nathan Shammah and Paul Nation and J. R. Johansson and Shahnawaz Ahmed and Simon Cross and Alexander Pitchford and Franco Nori},
    year={2024},
    eprint={2412.04705},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```
