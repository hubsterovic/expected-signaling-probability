# Expected Signaling Probability

$$
\mathcal{S}_{A \to B} =
D_{\text{Tr}}\left(
\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB}) \right],
\text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB})) \right]
\right)
$$

$$
\langle \mathcal{S} \rangle_{A \to B} =
\underset{\rho, \mathcal{E}, \mathcal{A}}{\mathbb{E}}
\Big[
\mathcal{S}_{A \to B}
\Big]
$$



## Getting Started Locally
### 1. Install [uv](https://docs.astral.sh/uv/)
Follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/) or simply use pip
```
pip install uv
```
### 2. Install dependencies 
```
uv sync
```
### 3. Run an experimemt
```
uv run expected_signaling_probability/experiments/symmetric_expected_signaling.py
uv run expected_signaling_probability/experiments/asymmetric_expected_signaling.py
```


## Main Package - [QuTiP](https://qutip.org/citing.html)

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