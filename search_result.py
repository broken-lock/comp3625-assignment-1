"""
The shared result returned by the search algorithms.
"""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """
    the best solution found and the reason the search stopped

    best_score is the objective value, not necessarily the decode rate
    num_evals counts objective calls made during this search
    """

    best_score: float
    best_vertex: list[float]
    num_evals: int
    search_outcome: str
