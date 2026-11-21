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
make test       # pytest
make lint       # ruff
make figures    # regenerate plots/fig_*.pdf|png and plots/exponents.{md,tex} from the run cache
```

## Results

Signalling $\mathcal S$ is read against the transmission $\mathcal T$, the trace distance between Bob's outputs for two
independent inputs, evaluated on the same draw $(\rho, \sigma, \mathcal A, \mathcal E)$:

$$
\mathcal T_{A\to B} = D_{\mathrm{Tr}}\big(\mathrm{Tr}_A[\mathcal E(\rho)],\ \mathrm{Tr}_A[\mathcal E(\sigma)]\big),
\qquad
\mathcal L_{A\to B} = D_{\mathrm{Tr}}\big(\mathrm{Tr}_A[\mathcal E(\rho)],\ \mathrm{Tr}_A[\mathcal E(\mathcal A_B(\rho))]\big),
$$

where $\mathcal L$ is Bob's own-side sensitivity (the same local unitary applied on Bob's factor).

| figure | script | what it shows |
|---|---|---|
| `plots/fig_symmetric` | `experiments/fig_symmetric.py` | $\langle\mathcal S\rangle$ and $\langle\mathcal T\rangle$ vs $d=d_A=d_B$; $\mathcal S$ sits on the $\mathcal T$ curve in every ensemble, flat for unitary |
| `plots/fig_ratio` | `experiments/fig_ratio.py` | $\mathrm{med}(\mathcal S/\mathcal T)$ with IQR; $\to 1$ for Haar inputs, plateaus at $\approx 0.8$ (unitary) / $0.7$ (COMS) for product inputs, where $\mathrm{med}(\mathcal S/\mathcal L)$ is $1.00$ / $0.82$, flat in $d$ |
| `plots/fig_asymmetric` | `experiments/fig_asymmetric.py` | $d_B=2$, both directions, per ensemble, with the Page guides below |
| `plots/exponents.md` | `experiments/table_exponents.py` | fitted exponents against the Page rule |

**Page rule.** $\mathcal S$ and $\mathcal T$ both scale as $\sqrt{d_{\text{receiver}} / d_{\text{discarded}}}$ with
$d_{\text{discarded}} = d_{\text{sender}} \cdot K$, $K$ the Kraus rank of the global ensemble ($1$, $D$, $D^2$).
At $d_B = 2$ this predicts $d_A^{-1/2}, d_A^{-1}, d_A^{-3/2}$ for $A\to B$ (unitary, COMS, BCSZ) and
saturation, saturation, $d_A^{-1/2}$ for $B\to A$; the fits agree within error.

## API

A `DrawConfig` fully declares one Monte Carlo draw (up to the seed):

```python
from expected_signaling_probability import *

cfg = DrawConfig(
    d_A=8, d_B=2,
    direction=Direction.A_TO_B,
    state=StateEnsemble.HAAR_PURE,          # HAAR_PURE | HS_MIXED | PRODUCT_HS
    local_op=LocalOpEnsemble.HAAR_UNITARY,  # HAAR_UNITARY | BCSZ
    global_op=GlobalOpEnsemble.COMS,        # HAAR_UNITARY | COMS | BCSZ | WEAK_COUPLING
    global_rank=None,                       # Kraus rank, BCSZ only; None = full
    coupling=None,                          # interaction strength g, WEAK_COUPLING only
    quantity=Quantity.SIGNALING,            # SIGNALING (S) | TRANSMISSION (T) | LOCAL (L) | RATIO (S/T) | RATIO_LOCAL (S/L)
)
samples = run(cfg, n=1000, master_seed=0)   # cached, resumable
print(samples)                              # mean ± sem, median, IQR
samples = sweep([cfg.with_(d_A=d) for d in range(2, 33)], n=1000)
```

Ratios are evaluated pointwise on the same draw and reported as median with IQR; `S`, `T`, `L` as mean with SEM.

### Seeding
Draw `i` of a run uses `SeedSequence([master_seed, i, SCHEME_VERSION]).spawn(4)`; the
state, the second state (for `T`), the local operation and the global operation
each get their own child generator, so they are statistically independent. `L` re-uses a fresh copy
of the local-operation stream on the receiver's side.
Bump `SCHEME_VERSION` in `core/config.py` whenever the meaning of a draw changes;
it is part of every cache key.

### Cache
One file pair per `(config, master_seed)` under `data/runs/` (tracked in git, so every figure
is reproducible without re-running):
`<slug>_seed=<master_seed>.csv` (columns `index,value`, appended as computed) and a
`.json` sidecar with the config, qutip version and git commit. `run(cfg, n)` computes
only indices not yet on disk, so extending `n` or resuming an interrupted run is free.

### Ensembles (`core/ensembles.py`)
| | ensemble | definition | Kraus rank |
|---|---|---|---|
| state | `HAAR_PURE` | Haar random pure state on $\mathcal H_A \otimes \mathcal H_B$ | |
| | `HS_MIXED` | Hilbert–Schmidt measure (full-rank Ginibre) | |
| | `PRODUCT_HS` | $\rho_A \otimes \rho_B$, each Hilbert–Schmidt | |
| local op | `HAAR_UNITARY` | Haar unitary on the sender's factor | |
| | `BCSZ` | full-rank BCSZ CPTP map on the sender's factor | |
| global op | `HAAR_UNITARY` | Haar unitary on $\mathcal H_A \otimes \mathcal H_B$ | 1 |
| | `COMS` | complete orthogonal measurement superoperator $\rho \mapsto \sum_k P_k \rho P_k$, $P_k$ rank-1 projectors onto a Haar random basis (Beckman et al.) | $D$ |
| | `BCSZ` | BCSZ CPTP map on $\mathcal H_A \otimes \mathcal H_B$ with Kraus rank `global_rank` (Bruzda et al.) | $D^2$ (full) |
| | `WEAK_COUPLING` | $(U_A \otimes U_B)\, e^{-i g H_{AB}}$, $H_{AB}$ GUE with spectrum on $[-2, 2]$ | 1 |

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
