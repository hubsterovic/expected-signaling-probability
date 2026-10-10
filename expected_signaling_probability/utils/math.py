from expected_signaling_probability.utils.params import ExtraParams, StateEnsemble, DEFAULT_EXTRA_PARAMS
from expected_signaling_probability.utils.directions import Direction
from expected_signaling_probability.utils.caching import (
    Cache, 
    SIGNALING_CACHE, 
    TRANSMISSION_CACHE,
    CORRELATION_CACHE,
)
from tqdm import tqdm
import qutip as qt
import numpy as np


def _compute_signaling_probability(
    initial_state: qt.Qobj,
    local_operation: qt.Qobj,
    global_superoperator: qt.Qobj,
    direction: Direction = Direction.A_TO_B,
) -> float:
    altered_initial_state = local_operation(initial_state)

    final_state = global_superoperator(initial_state)
    final_altered_state = global_superoperator(altered_initial_state)

    reduced_final_state = qt.ptrace(final_state, direction.to_ptrace_index())
    reduced_final_altered_state = qt.ptrace(final_altered_state, direction.to_ptrace_index())

    tr_dist = qt.tracedist(reduced_final_state, reduced_final_altered_state)
    return tr_dist


def generate_random_dm(d_A: int, d_B: int, seed: int | None = None, state_ensemble: StateEnsemble = StateEnsemble.GINIBRE) -> qt.Qobj:
    if state_ensemble == StateEnsemble.PURE:
        ket = qt.rand_ket(dimensions=[d_A, d_B], seed=seed)  # type: ignore
        return ket * ket.dag()
    if state_ensemble == StateEnsemble.PRODUCT:
        ket_A = qt.rand_ket(d_A, seed=seed)  # type: ignore
        ket_B = qt.rand_ket(d_B, seed=seed + 1 if seed is not None else None)  # type: ignore
        return qt.tensor(ket_A * ket_A.dag(), ket_B * ket_B.dag())
    random_dm = qt.rand_dm(dimensions=[d_A, d_B], seed=seed)  # type: ignore
    return random_dm


def _random_cptp_or_unitary(dimensions, seed, rank) -> qt.Qobj:
    if rank == 1:
        U = qt.rand_unitary(dimensions, seed=seed)  # type: ignore
        return qt.to_super(U)
    return qt.rand_super_bcsz(dimensions, seed=seed, rank=rank)  # type: ignore


def generate_random_superoperator(d_A: int, d_B: int, seed: int | None = None, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> qt.Qobj:
    return _random_cptp_or_unitary([d_A, d_B], seed, extra_params.superoperator_rank)


def generate_random_local_superoperator(d_A: int, d_B: int, direction: Direction, seed: int | None = None, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> qt.Qobj:
    if direction == Direction.A_TO_B:
        local_superoperator = qt.super_tensor(
            _random_cptp_or_unitary(d_A, seed, extra_params.superoperator_rank),
            qt.to_super(qt.identity(d_B)),
        )
    elif direction == Direction.B_TO_A:
        local_superoperator = qt.super_tensor(
            qt.to_super(qt.identity(d_A)),
            _random_cptp_or_unitary(d_B, seed, extra_params.superoperator_rank),
        )
    return local_superoperator


def _one_shot_signaling_probability(d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = SIGNALING_CACHE, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> float:
    if cache and (cached_result := cache.get(d_A, d_B, direction, seed, extra_params)):
        return cached_result

    initial_state = generate_random_dm(d_A, d_B, seed=3 * seed, state_ensemble=extra_params.state_ensemble)
    local_operation = generate_random_local_superoperator(d_A, d_B, direction, seed=3 * seed + 1, extra_params=extra_params)
    global_superoperator = generate_random_superoperator(d_A, d_B, seed=3 * seed + 2, extra_params=extra_params)
    tr_dist = _compute_signaling_probability(initial_state, local_operation, global_superoperator, direction)

    if cache:
        cache.set(d_A, d_B, direction, seed, tr_dist, extra_params)

    return tr_dist


def expected_signaling_probability(
    n_samples: int,
    d_A: int,
    d_B: int,
    direction: Direction,
    cache: Cache | None = SIGNALING_CACHE,
    extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS,
    _initial_seed_state: int = 0,
) -> np.typing.NDArray:
    seed = _initial_seed_state
    tr_dists: list[float] = []

    if cache is not None:
        cache.warm(d_A, d_B, direction, extra_params)

    for _ in tqdm(
        range(n_samples),
        desc=f"Computing <S>_{direction.value} ({d_A=}, {d_B=})",
        leave=False,
    ):
        seed += 1

        tr_dist = _one_shot_signaling_probability(d_A, d_B, direction, seed, cache, extra_params)
        tr_dists.append(tr_dist)

    # Close the warmed cache to flush updates
    if cache is not None:
        cache.close()

    return np.array(tr_dists)



# ------------------------------------------------------------
#                            extra                        
# ------------------------------------------------------------

def _compute_transmission_probability(initial_state_one: qt.Qobj, initial_state_two: qt.Qobj, global_superoperator: qt.Qobj, direction: Direction) -> float:
    final_state_one = global_superoperator(initial_state_one)
    final_state_two = global_superoperator(initial_state_two)
    reduced_final_state_one = qt.ptrace(final_state_one, direction.to_ptrace_index())
    reduced_final_state_two = qt.ptrace(final_state_two, direction.to_ptrace_index())
    tr_dist = qt.tracedist(reduced_final_state_one, reduced_final_state_two)
    return tr_dist


def _one_shot_transmission_probability(d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = TRANSMISSION_CACHE, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> float:
    if cache and (cached_result := cache.get(d_A, d_B, direction, seed, extra_params)):
        return cached_result

    initial_state_one = generate_random_dm(d_A, d_B, seed)
    initial_state_two = generate_random_dm(d_A, d_B, seed + 1)
    global_superoperator = generate_random_superoperator(d_A, d_B, seed, extra_params)
    tr_dist = _compute_transmission_probability(initial_state_one, initial_state_two, global_superoperator, direction)

    if cache:
        cache.set(d_A, d_B, direction, seed, tr_dist, extra_params)

    return tr_dist

def expected_transmission_probability(
    n_samples: int,
    d_A: int,
    d_B: int,
    direction: Direction,
    cache: Cache | None = TRANSMISSION_CACHE,
    extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS,
    _initial_seed_state: int = 0,
) -> np.typing.NDArray:
    seed = _initial_seed_state
    tr_dists: list[float] = []

    if cache is not None:
        cache.warm(d_A, d_B, direction, extra_params)

    for _ in tqdm(
        range(n_samples),
        desc=f"Computing <T>_{direction.value} ({d_A=}, {d_B=})",
        leave=False,
    ):
        seed += 1
        tr_dist = _one_shot_transmission_probability(d_A, d_B, direction, seed, cache, extra_params)
        tr_dists.append(tr_dist)

    if cache is not None:
        cache.close()

    return np.array(tr_dists)



def _compute_correlation_probability(initial_state_one: qt.Qobj, initial_state_two: qt.Qobj, direction: Direction) -> float:
    reduced_initial_state_one = qt.ptrace(initial_state_one, direction.to_ptrace_index())
    reduced_initial_state_two = qt.ptrace(initial_state_two, direction.to_ptrace_index())
    tr_dist = qt.tracedist(reduced_initial_state_one, reduced_initial_state_two)
    return tr_dist

def _one_shot_correlation_probability(d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = CORRELATION_CACHE, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> float:
    if cache and (cached_result := cache.get(d_A, d_B, direction, seed, extra_params)):
        return cached_result

    initial_state_one = generate_random_dm(d_A, d_B, seed)
    initial_state_two = generate_random_dm(d_A, d_B, seed + 1)
    tr_dist = _compute_correlation_probability(initial_state_one, initial_state_two, direction)

    if cache:
        cache.set(d_A, d_B, direction, seed, tr_dist, extra_params)

    return tr_dist


def expected_correlation_probability(
    n_samples: int,
    d_A: int,
    d_B: int,
    direction: Direction,
    cache: Cache | None = CORRELATION_CACHE,
    extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS,
    _initial_seed_state: int = 0,
) -> np.typing.NDArray:
    seed = _initial_seed_state
    tr_dists: list[float] = []

    if cache is not None:
        cache.warm(d_A, d_B, direction, extra_params)

    for _ in tqdm(
        range(n_samples),
        desc=f"Computing <C>_{direction.value} ({d_A=}, {d_B=})",
        leave=False,
    ):
        seed += 1
        tr_dist = _one_shot_correlation_probability(d_A, d_B, direction, seed, cache, extra_params)
        tr_dists.append(tr_dist)

    if cache is not None:
        cache.close()

    return np.array(tr_dists)