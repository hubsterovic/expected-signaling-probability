# Expected Signaling Probability
$$ \mathcal{S}_{A \to B} = \text{D}_{\text{Tr}}(\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB})\right], \text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB}))\right]) $$
$$\langle \mathcal{S} \rangle_{A \to B} = \underset{\rho,\mathcal{E}, \mathcal{A}}{\mathbb{E}} \Big[ \text{D}_{\text{Tr}}(\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB})\right], \text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB}))\right])  \Big]$$


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
uv run expected_signaling_probability/experiments/symmetric_expected_singaling.py
uv run expected_signaling_probability/experiments/asymmetric_expected_singaling.py
```