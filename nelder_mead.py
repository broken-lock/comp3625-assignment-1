"""
a custom implementation of the nelder-mead simplex algorithm
"""

import math
from collections.abc import Callable

from search_result import SearchResult


def transform(
    fixed_point: list[float],
    point_to_transform: list[float],
    scale_factor: float,
) -> list[float]:
    """
    scale a point's displacement relative to a fixed point
    """
    return [
        fixed + scale_factor * (coordinate - fixed)
        for fixed, coordinate in zip(fixed_point, point_to_transform)
    ]


def nelder_mead(
    objective: Callable[[list[float]], float],
    x0: list[float],
    simplex: list[list[float]] | None = None,
    bounds: list[tuple[float, float]] | None = None,
    max_evals: int | None = None,
    x_tolerance: float = 1e-4,
    f_tolerance: float = 1e-4,
    cache: dict[tuple[float, ...], float] | None = None,
) -> SearchResult:
    """
    minimize an objective using the nelder-mead simplex algorithm

    :param callable objective: objective function to minimize
    :param x0: the initial vertex vector when no simplex is supplied
    :param simplex: the initial simplex to start the search with
    :param bounds: a list of bound pairs (lower, upper) for each coordinate
    :param int max_evals: maximum number of objective function evaluations
    :param x_tolerance: maximum distance from the best vertex for convergence
    :param f_tolerance: maximum difference between the best and worst scores for convergence
    :param cache: optional shared scores for this objective, keyed by exact coordinates
    """
    # reuse scores only for the same deterministic objective
    if cache is None:
        cache = {}

    n = len(x0)

    if n == 0:
        raise ValueError("x0 must contain at least one coordinate")

    if max_evals is not None and max_evals < 1:
        raise ValueError("max_evals must be positive or None")

    if x_tolerance < 0 or f_tolerance < 0:
        raise ValueError("tolerances cannot be negative")

    if bounds is not None:
        if len(bounds) != n or any(low >= high for low, high in bounds):
            raise ValueError("provide one valid bound pair per coordinate")

    reflection = 1.0
    expansion = 2.0
    contraction = 0.5
    shrink = 0.5

    num_evals = 0
    best_score = math.inf
    best_vertex = []

    class EvaluationLimitReached(Exception):
        pass

    def clip(vertex):
        if bounds is None:
            return vertex.copy()

        return [
            max(low, min(high, coordinate))
            for coordinate, (low, high) in zip(vertex, bounds)
        ]

    def evaluate(vertex):
        nonlocal num_evals, best_score, best_vertex

        vertex = clip(vertex)
        key = tuple(vertex)

        # cached scores do not consume objective evaluations
        if key not in cache:
            if max_evals is not None and num_evals >= max_evals:
                raise EvaluationLimitReached
            cache[key] = objective(vertex.copy())
            num_evals += 1

        value = cache[key]

        # retain the best evaluated point even if an operation is interrupted
        if not best_vertex or value < best_score:
            best_vertex = vertex.copy()
            best_score = value

        return vertex, value

    if simplex is None:
        start = clip(x0)
        vertices = [start.copy()]

        # create one additional vertex along each coordinate direction
        for i in range(n):
            vertex = start.copy()

            if bounds is not None:
                low, high = bounds[i]
                step_size = 0.05 * (high - low)

                if vertex[i] + step_size <= high:
                    vertex[i] += step_size
                else:
                    vertex[i] -= step_size
            else:
                step_size = 0.05 * abs(vertex[i]) if vertex[i] != 0 else 0.05
                vertex[i] += step_size

            vertices.append(vertex)
    else:
        if len(simplex) != n + 1 or any(len(vertex) != n for vertex in simplex):
            raise ValueError("the simplex must have n+1 vertices of length n")

        vertices = [vertex.copy() for vertex in simplex]

    try:
        # store each vertex together with its objective value
        simplex = [evaluate(vertex) for vertex in vertices]

        while True:
            simplex.sort(key=lambda entry: entry[1])

            current_best, best_value = simplex[0]
            worst_vertex, worst_value = simplex[-1]
            second_worst_value = simplex[-2][1]

            score_spread = worst_value - best_value
            vertex_spread = max(
                math.dist(vertex, current_best)
                for vertex, _ in simplex
            )

            if (
                score_spread <= f_tolerance
                and vertex_spread <= x_tolerance
            ):
                search_outcome = "position and score tolerances met"
                break

            # calculate the centroid excluding the worst vertex
            centroid = [
                sum(vertex[i] for vertex, _ in simplex[:-1]) / n
                for i in range(n)
            ]

            # reflection
            reflected_vertex, reflected_value = evaluate(
                transform(centroid, worst_vertex, -reflection)
            )

            if best_value <= reflected_value < second_worst_value:
                simplex[-1] = (reflected_vertex, reflected_value)
                continue

            # expansion
            if reflected_value < best_value:
                expanded_vertex, expanded_value = evaluate(
                    transform(centroid, reflected_vertex, expansion)
                )

                if expanded_value < reflected_value:
                    simplex[-1] = (expanded_vertex, expanded_value)
                else:
                    simplex[-1] = (reflected_vertex, reflected_value)

                continue

            # outside contraction
            if reflected_value < worst_value:
                contracted_vertex, contracted_value = evaluate(
                    transform(centroid, reflected_vertex, contraction)
                )

                if contracted_value <= reflected_value:
                    simplex[-1] = (contracted_vertex, contracted_value)
                    continue

            # inside contraction
            else:
                contracted_vertex, contracted_value = evaluate(
                    transform(centroid, worst_vertex, contraction)
                )

                if contracted_value < worst_value:
                    simplex[-1] = (contracted_vertex, contracted_value)
                    continue

            # shrink every vertex except the current best
            simplex = [simplex[0]] + [
                evaluate(transform(current_best, vertex, shrink))
                for vertex, _ in simplex[1:]
            ]

    except EvaluationLimitReached:
        search_outcome = "maximum evaluations reached"

    return SearchResult(
        best_score=best_score,
        best_vertex=best_vertex,
        num_evals=num_evals,
        search_outcome=search_outcome,
    )
