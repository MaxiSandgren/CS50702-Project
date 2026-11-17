"""Plot success and best path cost over time for one problem."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import beta, binom

from plot_success import median_interval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Convergence results folder")
    args = parser.parse_args(argv)
    metadata = json.loads((args.folder / "metadata.json").read_text())
    if metadata["config"].get("mode") != "convergence":
        parser.error("Choose a results folder created with --mode convergence.")
    trials = [json.loads(path.read_text())
              for path in sorted(args.folder.glob("problem_*_trial_*.json"))]
    if len(trials) != metadata["trials_per_problem"] or len(metadata["problem_indices"]) != 1:
        parser.error("The folder must contain all trials for exactly one problem.")

    end = metadata["config"]["timeout_seconds"]
    events = [event for trial in trials for event in trial.get("improvements", [])]
    start = min(1e-4, end / 10, min((e["time_s"] for e in events), default=end) / 10)
    times = np.unique(np.concatenate((
        np.geomspace(start, end, 400),
        [e["time_s"] for e in events],
        [end],
    )))
    costs = np.full((len(trials), len(times)), np.inf)
    for index, trial in enumerate(trials):
        curve = trial.get("improvements", [])
        if not curve:
            continue
        event_times = np.array([event["time_s"] for event in curve])
        event_costs = np.array([event["cost"] for event in curve])
        if np.any(np.diff(event_times) < 0) or np.any(np.diff(event_costs) >= 0):
            parser.error("Improvement logs must have increasing times and decreasing costs.")
        positions = np.searchsorted(event_times, times, side="right") - 1
        present = positions >= 0
        costs[index, present] = event_costs[positions[present]]

    solved = np.isfinite(costs).sum(axis=0)
    success = 100 * solved / len(trials)
    success_lower = np.zeros(len(times))
    success_upper = np.full(len(times), 100.0)
    present = solved > 0
    success_lower[present] = 100 * beta.ppf(0.005, solved[present], len(trials) - solved[present] + 1)
    incomplete = solved < len(trials)
    success_upper[incomplete] = 100 * beta.ppf(
        0.995, solved[incomplete] + 1, len(trials) - solved[incomplete])

    ordered = np.sort(costs, axis=0)
    middle = len(trials) // 2
    median = ordered[middle].copy()
    if len(trials) % 2 == 0:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    rank = int(binom.ppf(0.005, len(trials), 0.5))
    lower = ordered[rank].copy()
    upper = ordered[-rank - 1].copy()
    if 2 ** (1 - len(trials)) > 0.01:
        lower[:] = np.nan
        upper[:] = np.nan
    median[~np.isfinite(median)] = np.nan
    lower[~np.isfinite(lower)] = np.nan
    upper[~np.isfinite(upper)] = np.nan

    table = args.folder / "convergence_over_time.csv"
    with table.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(("time_s", "solved_by_time", "success_pct", "success_lower_99_pct",
                         "success_upper_99_pct", "median_cost", "cost_lower_99", "cost_upper_99"))
        writer.writerows(zip(times, solved, success, success_lower, success_upper, median, lower, upper))

    figure = Figure(figsize=(8, 5.5), layout="constrained")
    success_axis, cost_axis = figure.subplots(2, 1, sharex=True)
    success_axis.step(times, success, where="post", color="#538d29",
                      label=f"FCIT* ({len(trials)} trials)")
    success_axis.fill_between(times, success_lower, success_upper, step="post",
                              color="#538d29", alpha=0.2)
    success_axis.set(ylabel="Success [%]", ylim=(-3, 103), yticks=range(0, 101, 25))
    success_axis.legend(loc="lower right")
    cost_axis.step(times, median, where="post", color="#538d29", label="Median best path cost")
    cost_axis.fill_between(times, lower, upper, step="post", color="#538d29",
                           alpha=0.2, label="99% confidence interval")
    first = [trial["improvements"][0] for trial in trials if trial.get("improvements")]
    if first:
        first_times = [event["time_s"] for event in first]
        first_costs = [event["cost"] for event in first]
        time_interval = median_interval(first_times)
        cost_interval = median_interval(first_costs)
        time_median = statistics.median(first_times)
        cost_median = statistics.median(first_costs)
        cost_axis.errorbar(
            time_median, cost_median,
            xerr=[[time_median - time_interval[0]], [time_interval[1] - time_median]] if time_interval else None,
            yerr=[[cost_median - cost_interval[0]], [cost_interval[1] - cost_median]] if cost_interval else None,
            fmt="+", color="#223c10", markersize=10, capsize=3, label="Median initial solution")
    cost_axis.set(xscale="log", xlim=(start, end), xlabel="Computation time [s]",
                  ylabel="Path cost (joint-space length)")
    cost_axis.legend(loc="upper right", fontsize=9)
    for axis in (success_axis, cost_axis):
        axis.grid(True, which="major", alpha=0.3)
    figure.suptitle(f"{metadata['config'].get('environment', 'bookshelf_small').replace('_', ' ').capitalize()} — problem {metadata['problem_indices'][0]}")
    destination = args.folder / "convergence.png"
    figure.savefig(destination, dpi=180)
    print(f"Convergence data saved to {table}")
    print(f"Convergence plot saved to {destination}")


if __name__ == "__main__":
    main()
