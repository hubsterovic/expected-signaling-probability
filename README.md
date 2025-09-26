# Expected Signaling Probability
$$ \mathcal{S}_{A \to B} = \text{D}_{\text{Tr}}(\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB})\right], \text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB}))\right]) $$
$$\langle \mathcal{S} \rangle_{A \to B} = \underset{\rho,\mathcal{E}, \mathcal{A}}{\mathbb{E}} \Big[ \text{D}_{\text{Tr}}(\text{Tr}_A\left[ \mathcal{E}_{AB}(\rho_{AB})\right], \text{Tr}_A\left[ \mathcal{E}_{AB}(\mathcal{A}_A(\rho_{AB}))\right])  \Big]$$


## TimeComp 
Sampling Large Superoperators takes a long time
```
D=64  --> ~3.25s
D=81  --> ~11.42s
D=100 --> ~38.29s
D=121 --> ~127.45s
D=144 --> ~560.81s
``` 