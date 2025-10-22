from expected_signaling_probability.utils.params import ExtraParams, DEFAULT_EXTRA_PARAMS
from expected_signaling_probability.utils.directions import Direction
from expected_signaling_probability.utils.caching import (
    Cache, 
    SIGNALING_CACHE, 
    REDUCED_DISTINGUISHABILITY_CACHE,
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


def generate_random_dm(d_A: int, d_B: int, seed: int | None = None) -> qt.Qobj:
    random_dm = qt.rand_dm(dimensions=[d_A, d_B], seed=seed)  # type: ignore
    return random_dm


def generate_random_superoperator(d_A: int, d_B: int, seed: int | None = None, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> qt.Qobj:
    return qt.rand_super_bcsz([d_A, d_B], seed=seed, rank=extra_params.superoperator_rank)  # type: ignore


def generate_random_local_superoperator(d_A: int, d_B: int, direction: Direction, seed: int | None = None, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> qt.Qobj:
    if direction == Direction.A_TO_B:
        local_superoperator = qt.super_tensor(
            qt.rand_super_bcsz(d_A, seed=seed, rank=extra_params.superoperator_rank),  # type: ignore
            qt.to_super(qt.identity(d_B)),
        )
    elif direction == Direction.B_TO_A:
        local_superoperator = qt.super_tensor(
            qt.to_super(qt.identity(d_A)),
            qt.rand_super_bcsz(d_B, seed=seed, rank=extra_params.superoperator_rank),  # type: ignore
        )
    return local_superoperator


def _one_shot_signaling_probability(d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = SIGNALING_CACHE, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> float:
    if cache and (cached_result := cache.get(d_A, d_B, direction, seed, extra_params)):
        return cached_result

    initial_state = generate_random_dm(d_A, d_B, seed)
    local_operation = generate_random_local_superoperator(d_A, d_B, direction, seed, extra_params)
    global_superoperator = generate_random_superoperator(d_A, d_B, seed, extra_params)
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


def _compute_reduced_distinguishability(initial_state_one: qt.Qobj, initial_state_two: qt.Qobj, global_superoperator: qt.Qobj, direction: Direction) -> float:
    final_state_one = global_superoperator(initial_state_one)
    final_state_two = global_superoperator(initial_state_two)
    reduced_final_state_one = qt.ptrace(final_state_one, direction.to_ptrace_index())
    reduced_final_state_two = qt.ptrace(final_state_two, direction.to_ptrace_index())
    tr_dist = qt.tracedist(reduced_final_state_one, reduced_final_state_two)
    return tr_dist


def _one_shot_reduced_distinguishability(d_A: int, d_B: int, direction: Direction, seed: int, cache: Cache | None = REDUCED_DISTINGUISHABILITY_CACHE, extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS) -> float:
    if cache and (cached_result := cache.get(d_A, d_B, direction, seed, extra_params)):
        return cached_result

    initial_state_one = generate_random_dm(d_A, d_B, seed)
    initial_state_two = generate_random_dm(d_A, d_B, seed + 1)
    global_superoperator = generate_random_superoperator(d_A, d_B, seed, extra_params)
    tr_dist = _compute_reduced_distinguishability(initial_state_one, initial_state_two, global_superoperator, direction)

    if cache:
        cache.set(d_A, d_B, direction, seed, tr_dist, extra_params)

    return tr_dist

def expected_reduced_distinguishability(
    n_samples: int,
    d_A: int,
    d_B: int,
    direction: Direction,
    cache: Cache | None = REDUCED_DISTINGUISHABILITY_CACHE,
    extra_params: ExtraParams = DEFAULT_EXTRA_PARAMS,
    _initial_seed_state: int = 0,
) -> np.typing.NDArray:
    seed = _initial_seed_state
    tr_dists: list[float] = []

    if cache is not None:
        cache.warm(d_A, d_B, direction, extra_params)

    for _ in tqdm(
        range(n_samples),
        desc=f"Computing R_{direction.value} ({d_A=}, {d_B=})",
        leave=False,
    ):
        seed += 1
        tr_dist = _one_shot_reduced_distinguishability(d_A, d_B, direction, seed, cache, extra_params)
        tr_dists.append(tr_dist)

    if cache is not None:
        cache.close()

    return np.array(tr_dists)

