"""
simulated annealing followed by nelder-mead refinement
"""

import math
import random
from collections.abc import Callable

from nelder_mead import nelder_mead
from search_result import SearchResult


def hybrid_annealing_simplex(
    objective: Callable[[list[float]], float],
    x0: list[float],
    bounds: list[tuple[float, float]],
    max_evals: int = 1500,
    initial_temp: float = 0.1,
    x_tolerance: float = 1e-4,
    f_tolerance: float = 1e-4,
) -> SearchResult:
    """
    explore using simulated annealing and refine using nelder-mead

    :param callable objective: objective function to minimize
    :param x0: the initial vertex vector
    :param bounds: a list of bound pairs (lower, upper) for each coordinate
    :param int max_evals: maximum objective evaluations across both phases
    :param initial_temp: the initial annealing temperature
    :param x_tolerance: position tolerance for nelder-mead convergence
    :param f_tolerance: score tolerance for nelder-mead convergence
    """
    n = len(x0)

    if n == 0:
        raise ValueError("x0 must contain at least one coordinate")

    if max_evals < 1:
        raise ValueError("max_evals must be positive")

    if initial_temp <= 0:
        raise ValueError("initial_temp must be positive")

    if x_tolerance < 0 or f_tolerance < 0:
        raise ValueError("tolerances cannot be negative")

    if len(bounds) != n or any(low >= high for low, high in bounds):
        raise ValueError("provide one valid bound pair per coordinate")

    cache = {}
    num_evals = 0

    def evaluate(vertex):
        nonlocal num_evals

        key = tuple(vertex)
        if key not in cache:
            cache[key] = objective(vertex.copy())
            num_evals += 1
        return cache[key]

    current_vertex = [
        max(low, min(high, coordinate))
        for coordinate, (low, high) in zip(x0, bounds)
    ]
    current_value = evaluate(current_vertex)

    best_vertex = current_vertex.copy()
    best_score = current_value

    # allocate 70 percent to annealing, including the initial evaluation
    annealing_budget = max(1, int(max_evals * 0.70))
    annealing_steps = annealing_budget - 1

    for step in range(annealing_steps):
        # cool linearly while keeping the temperature positive
        temperature = initial_temp * (1.0 - step / annealing_steps)

        # perturb one randomly chosen coordinate
        candidate_vertex = current_vertex.copy()

        i = random.randrange(n)
        low, high = bounds[i]
        step_size = 0.05 * (high - low)

        candidate_vertex[i] += random.gauss(0, step_size)
        candidate_vertex[i] = max(
            low, min(high, candidate_vertex[i])
        )

        candidate_value = evaluate(candidate_vertex)

        value_change = candidate_value - current_value

        # accept improvements and probabilistically accept worsening moves
        if (
            value_change <= 0
            or random.random() < math.exp(-value_change / temperature)
        ):
            current_vertex = candidate_vertex.copy()
            current_value = candidate_value

        # preserve the best solution separately from the current position
        if candidate_value < best_score:
            best_vertex = candidate_vertex.copy()
            best_score = candidate_value

    print("annealing best decode rate:", -best_score)
    remaining_evals = max_evals - num_evals

    # initialize a simplex only if the remaining budget can evaluate it
    if remaining_evals >= n + 1:
        simplex_result = nelder_mead(
            objective=objective,
            x0=best_vertex,
            bounds=bounds,
            max_evals=remaining_evals,
            x_tolerance=x_tolerance,
            f_tolerance=f_tolerance,
            cache=cache,
        )

        num_evals += simplex_result.num_evals

        if simplex_result.best_score < best_score:
            best_vertex = simplex_result.best_vertex.copy()
            best_score = simplex_result.best_score

        search_outcome = f"nelder-mead: {simplex_result.search_outcome}"
    else:
        search_outcome = "insufficient remaining evaluations for a simplex"

    return SearchResult(
        best_score=best_score,
        best_vertex=best_vertex,
        num_evals=num_evals,
        search_outcome=search_outcome,
    )
