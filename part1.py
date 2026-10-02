from translator import UniversalTranslator
from hybrid_annealing_simplex import hybrid_annealing_simplex
import util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # create the translator before calling the objective
    translator = UniversalTranslator(n_dim=2)

    util.evaluated_settings.clear()
    util.decode_rates.clear()

    # run the search before plotting its evaluations
    result = hybrid_annealing_simplex(
        objective=util.objective,
        x0=[0.5, 0.5],
        bounds=[(0, 1), (0, 1)],
        max_evals=1500,
        initial_temp=0.1,
    )

    print("best settings:", result.best_vertex)
    print("best decode rate:", -result.best_score)
    print("total settings tried:", translator.n_settings_tried())

    # plot every evaluated two-knob configuration
    knob_1 = [settings[0] for settings in util.evaluated_settings]
    knob_2 = [settings[1] for settings in util.evaluated_settings]

    fig, ax = plt.subplots()

    points = ax.scatter(
        knob_1,
        knob_2,
        c=util.decode_rates,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=20,
    )

    ax.set_xlabel("knob 1 setting")
    ax.set_ylabel("knob 2 setting")
    ax.set_title("evaluated settings and decode rates")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid()

    fig.colorbar(points, ax=ax, label="decode rate")
    fig.tight_layout()
    fig.savefig("part1_search.png", dpi=200)
    plt.close(fig)