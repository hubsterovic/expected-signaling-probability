from expected_signaling_probability.utils.directions import Direction
from expected_signaling_probability.utils.caching import Cache
from tqdm import tqdm
import qutip as qt

_CACHE = Cache()


def compute_signaling_probability(
    initial_state: qt.Qobj,
    local_operation: qt.Qobj,
    global_superoperator: qt.Qobj,
    direction: Direction = Direction.A_TO_B,
) -> float:
    altered_initial_state = local_operation(initial_state)

    final_state = global_superoperator(initial_state)
    final_altered_state = global_superoperator(altered_initial_state)

    reduced_final_state = qt.ptrace(final_state, direction.to_ptrace_index())
    reduced_final_altered_state = qt.ptrace(
        final_altered_state, direction.to_ptrace_index()
    )

    tr_dist = qt.tracedist(reduced_final_state, reduced_final_altered_state)
    return tr_dist


def generate_random_dm(d_A: int, d_B: int, seed: int | None = None) -> qt.Qobj:
    random_dm = qt.rand_dm(dimensions=[d_A, d_B], seed=seed)  # type: ignore
    return random_dm


def generate_random_superoperator(
    d_A: int, d_B: int, seed: int | None = None
) -> qt.Qobj:
    return qt.rand_super_bcsz([d_A, d_B], seed=seed)  # type: ignore


def generate_random_local_superoperator(
    d_A: int, d_B: int, direction: Direction, seed: int | None = None
) -> qt.Qobj:
    if direction == Direction.A_TO_B:
        local_superoperator = qt.super_tensor(
            qt.rand_super_bcsz(d_A, seed=seed),  # type: ignore
            qt.to_super(qt.identity(d_B)),
        )
    elif direction == Direction.B_TO_A:
        local_superoperator = qt.super_tensor(
            qt.to_super(qt.identity(d_A)),
            qt.rand_super_bcsz(d_B, seed=seed),  # type: ignore
        )
    return local_superoperator


def _one_shot_signaling_probability(
    d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = _CACHE
) -> float:
    if cache:
        cached_result = cache.get(d_A, d_B, direction, seed)
        if cached_result is not None:
            return cached_result

    initial_state = generate_random_dm(d_A, d_B, seed)
    local_operation = generate_random_local_superoperator(d_A, d_B, direction, seed)
    global_superoperator = generate_random_superoperator(d_A, d_B, seed)
    tr_dist = compute_signaling_probability(
        initial_state, local_operation, global_superoperator, direction
    )

    if cache:
        cache.set(d_A, d_B, direction, seed, tr_dist)

    return tr_dist


def expected_signaling_probability(
    n_samples: int,
    d_A: int,
    d_B: int,
    direction: Direction,
    cache: Cache | None = _CACHE,
    _initial_seed_state: int = 0,
) -> tuple[float, list[float]]:
    seed = _initial_seed_state
    tr_dists: list[float] = []

    for _ in tqdm(
        range(n_samples),
        desc=f"Computing <S>_{direction.value} ({d_A=}, {d_B=})",
        leave=False,
    ):
        seed += 1

        tr_dist = _one_shot_signaling_probability(d_A, d_B, direction, seed, cache)
        tr_dists.append(tr_dist)

    return sum(tr_dists) / n_samples, tr_dists
