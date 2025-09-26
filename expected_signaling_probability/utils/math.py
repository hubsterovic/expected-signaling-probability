import qutip as qt
from enum import Enum
from tqdm import tqdm


class Direction(Enum):
    A_TO_B = "A to B"
    B_TO_A = "B to A"


def _direction_to_ptrace_index(direction: Direction) -> int:
    return 0 if direction == Direction.A_TO_B else 1


def compute_signaling_probability(
    initial_state: qt.Qobj,
    local_operation: qt.Qobj,
    global_superoperator: qt.Qobj,
    direction: Direction = Direction.A_TO_B,
) -> float:
    altered_initial_state = local_operation(initial_state)

    final_state = global_superoperator(initial_state)
    final_altered_state = global_superoperator(altered_initial_state)

    reduced_final_state = qt.ptrace(final_state, _direction_to_ptrace_index(direction))
    reduced_final_altered_state = qt.ptrace(
        final_altered_state, _direction_to_ptrace_index(direction)
    )

    tr_dist = qt.tracedist(reduced_final_state, reduced_final_altered_state)
    return tr_dist


def generate_random_dm(d_A: int, d_B: int, seed: int = 42) -> qt.Qobj:
    random_dm = qt.rand_dm(dimensions=[d_A, d_B], seed=seed)
    return random_dm


def generate_random_superoperator(d_A: int, d_B: int, seed: int = 42) -> qt.Qobj:
    return qt.rand_super_bcsz([d_A, d_B], seed=seed)


def generate_random_local_superoperator(
    d_A: int, d_B: int, direction: Direction, seed: int = 42
) -> qt.Qobj:
    match direction:
        case Direction.A_TO_B:
            local_superoperator = qt.super_tensor(
                qt.rand_super_bcsz(d_A, seed=seed), qt.to_super(qt.identity(d_B))
            )
        case Direction.B_TO_A:
            local_superoperator = qt.super_tensor(
                qt.to_super(qt.identity(d_A)), qt.rand_super_bcsz(d_B, seed=seed)
            )
    return local_superoperator


def expected_signaling_probability(
    n_samples: int, d_A: int, d_B: int, direction: Direction, seed: int = 42
) -> tuple[float, list[float]]:
    tr_dists: list[float] = []
    for _ in tqdm(
        range(n_samples),
        desc=f"Computing <S>_{direction} ({d_A=}, {d_B=})",
        leave=False,
    ):
        initial_state = generate_random_dm(d_A, d_B, seed)
        local_operation = generate_random_local_superoperator(d_A, d_B, direction, seed)
        global_superoperator = generate_random_superoperator(d_A, d_B, seed)
        tr_dist = compute_signaling_probability(
            initial_state, local_operation, global_superoperator, direction
        )
        tr_dists.append(tr_dist)
    return sum(tr_dists) / n_samples, tr_dists
