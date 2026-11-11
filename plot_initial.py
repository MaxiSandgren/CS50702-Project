"""Plot first-solution results the way Fig. 1b (cage) and Fig. 2d (bookshelf small) of the FCIT* paper do.

    python plot_initial.py cage_results/<run folder>/results.csv

Top:    share of trials that have found a solution by time t, with a 99 % Clopper-Pearson band.
Bottom: the median first solution, drawn as a square at (median time, median path length)
        with 99 % nonparametric intervals.
Unsolved trials count as infinitely slow and infinitely long, like in the paper.
Writes initial_solutions.png and summary.json next to results.csv, and prints the Table I numbers.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

ALPHA = 0.01  # 99 % intervals, as in the paper
GREEN = "#4e9a2a"  # close to the FCIT* green in the paper
# Axis ranges read off the paper's figures, so ours can sit next to theirs
X_RANGE = {"cage": (5e-5, 1e2), "bookshelf_small": (1.5e-5, 1e2)}
Y_RANGE = {"cage": (4, 26), "bookshelf_small": (4, 11)}
PAPER = {"cage": "Fig. 1b", "bookshelf_small": "Fig. 2d"}


def as_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def median_interval(values):
    """Median and nonparametric (order statistic) 99 % interval."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    lo = int(max(binom.ppf(ALPHA / 2, n, 0.5) - 1, 0))
    hi = int(min(binom.ppf(1 - ALPHA / 2, n, 0.5), n - 1))
    return float(np.median(v)), float(v[lo]), float(v[hi])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Path to results.csv")
    args = parser.parse_args()
    folder = args.results.parent
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    scene = metadata.get("scene", "bookshelf_small")

    with args.results.open(newline="", encoding="utf-8") as source:
        trials = list(csv.DictReader(source))
    solved = [t["solved"].strip().lower() == "true" for t in trials]
    times = np.array([as_float(t["planning_ms"]) / 1000 if ok else math.inf for t, ok in zip(trials, solved)])
    lengths = np.array([as_float(t["path_length"]) if ok else math.inf for t, ok in zip(trials, solved)])
    n = len(trials)
    problems = len({t["problem"] for t in trials})

    t_med, t_lo, t_hi = median_interval(times)
    c_med, c_lo, c_hi = median_interval(lengths)
    summary = {
        "scene": scene, "trials": n, "problems": problems,
        "solved_pct": 100 * sum(solved) / n,
        "median_first_solution_ms": 1000 * t_med, "median_first_solution_ms_99ci": [1000 * t_lo, 1000 * t_hi],
        "median_first_path_length": c_med, "median_first_path_length_99ci": [c_lo, c_hi],
        "cpu": metadata.get("cpu", "unknown"),
        "note": "Unsolved trials count as infinite time and length, like the paper.",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1 = X_RANGE.get(scene, (1e-5, 1e2))
    grid = np.logspace(math.log10(x0), math.log10(x1), 600)
    k = np.searchsorted(np.sort(times), grid, side="right")
    lo = np.where(k > 0, beta.ppf(ALPHA / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - ALPHA / 2, k + 1, n - k), 1.0)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 3.6), sharex=True,
                                      gridspec_kw={"hspace": 0.08})
    for ax in (top, bottom):
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    top.fill_between(grid, 100 * lo, 100 * hi, step="post", color=GREEN, alpha=0.2, linewidth=0)
    top.step(grid, 100 * k / n, where="post", color=GREEN, linewidth=2.5)
    top.set(ylim=(0, 100), yticks=[0, 25, 50, 75, 100], ylabel="Success [%]")

    if math.isfinite(t_med) and math.isfinite(c_med):
        bottom.errorbar(t_med, c_med,
                        xerr=[[t_med - t_lo], [t_hi - t_med if math.isfinite(t_hi) else 0]],
                        yerr=[[c_med - c_lo], [c_hi - c_med if math.isfinite(c_hi) else 0]],
                        fmt="s", color=GREEN, markersize=7, capsize=3, elinewidth=1.5)
    y0, y1 = Y_RANGE.get(scene, (0, max(1.0, 1.2 * c_med if math.isfinite(c_med) else 1.0)))
    finite = [v for v in (c_lo, c_hi) if math.isfinite(v)]
    if finite:
        y0, y1 = min(y0, 0.95 * min(finite)), max(y1, 1.05 * max(finite))
    bottom.set(xscale="log", xlim=(x0, x1), ylim=(y0, y1), xlabel="Computation time [s]", ylabel="Cost")

    fig.suptitle(f"Ours, FCIT*, {scene.replace('_', ' ')}: {problems} problems × {n // max(problems, 1)} trials "
                 f"(compare {PAPER.get(scene, 'the paper')})", x=0.01, ha="left", fontsize=11, color="#52514e")
    out = folder / "initial_solutions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Solved {summary['solved_pct']:.1f} % of {n} trials ({problems} problems)")
    print(f"Median first solution time: {summary['median_first_solution_ms']:.1f} ms "
          f"(99 % interval {1000 * t_lo:.1f} to {1000 * t_hi:.1f} ms)")
    print(f"Median first path length: {c_med:.2f} (99 % interval {c_lo:.2f} to {c_hi:.2f})")
    print(f"Plot: {out}")
    print(f"Summary: {folder / 'summary.json'}")


if __name__ == "__main__":
    main()
