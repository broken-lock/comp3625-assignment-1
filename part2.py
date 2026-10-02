import random
import numpy as np
from time import perf_counter

from translator import UniversalTranslator
from hybrid_annealing_simplex import hybrid_annealing_simplex
import util


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    translator = UniversalTranslator(n_dim=10)

    start_time = perf_counter()

    result = hybrid_annealing_simplex(
        objective=util.objective,
        x0=[0.5] * 10,
        bounds=[(0, 1)] * 10,
        max_evals=3000,
        initial_temp=0.1,
    )

    elapsed = perf_counter() - start_time

    print("best settings:", result.best_vertex)
    print("best decode rate:", -result.best_score)
    print("total settings tried:", translator.n_settings_tried())
    print("runtime in seconds:", elapsed)