"""Plot first-solution success and path cost against computation time."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics
from scipy.stats import beta, binom


def median_interval(values):
    """Exact two-sided, distribution-free 99% interval using order statistics."""
    ordered = sorted(values)
    rank = int(binom.ppf(0.005, len(ordered), 0.5))
    if 2 ** (1 - len(ordered)) > 0.01:
        return None
    return ordered[rank], ordered[-rank - 1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to results.csv")
    args = parser.parse_args(argv)
    with args.results.open(newline="", encoding="utf-8") as source:
        trials = list(csv.DictReader(source))
    if not trials:
        parser.error("The results CSV has no trials.")

    solution_times = Counter(
        float(trial["planning_ms"]) / 1000
        for trial in trials
        if trial["solved"].lower() == "true"
    )
    if any(not math.isfinite(t) or t <= 0 for t in solution_times):
        parser.error("Solved trials must have positive, finite planning times.")
    solutions = [
        (float(trial["planning_ms"]) / 1000, float(trial["path_length"]))
        for trial in trials
        if trial["solved"].lower() == "true"
    ]
    if any(not math.isfinite(cost) or cost < 0 for _, cost in solutions):
        parser.error("Solved trials must have finite, nonnegative path lengths.")

    folder = args.results.parent
    metadata_path = folder / "metadata.json"
    timeout = 0.0
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        timeout = float(metadata["config"]["timeout_seconds"])
    last_solution = max(solution_times, default=0.0)
    end = max(timeout, last_solution) or 1.0
    start = min(0.0001, end / 10, min(solution_times, default=end) / 10)
    cutoffs = sorted({start, end, *solution_times})

    solved = 0
    counts = []
    for cutoff in cutoffs:
        solved += solution_times[cutoff]
        counts.append((cutoff, solved, len(trials) - solved, 100 * solved / len(trials)))
    lower = [100 * beta.ppf(0.005, row[1], len(trials) - row[1] + 1)
             if row[1] else 0.0 for row in counts]
    upper = [100 * beta.ppf(0.995, row[1] + 1, len(trials) - row[1])
             if row[1] < len(trials) else 100.0 for row in counts]

    table_path = folder / "success_over_time.csv"
    with table_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("time_s", "solved_by_time", "unsolved_by_time", "success_pct",
                         "success_lower_99_pct", "success_upper_99_pct"))
        writer.writerows((*row, lo, hi) for row, lo, hi in zip(counts, lower, upper))

    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 3.5), layout="constrained")
    axis = figure.subplots()
    axis.step(cutoffs, [row[3] for row in counts], where="post", color="#538d29",
              label=f"FCIT* ({len(trials)} trials)")
    axis.fill_between(cutoffs, lower, upper, step="post", color="#538d29", alpha=0.2)
    axis.set(xscale="log", xlim=(start, end), ylim=(-3, 103),
             yticks=range(0, 101, 25), xlabel="Computation time [s]", ylabel="Success [%]")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend(loc="lower right")
    plot_path = folder / "success_over_time.png"
    figure.savefig(plot_path, dpi=180)
    print(f"Counts saved to {table_path}")
    print(f"Plot saved to {plot_path}")

    combined = Figure(figsize=(8, 5.5), layout="constrained")
    success_axis, cost_axis = combined.subplots(2, 1, sharex=True)
    success_axis.step(cutoffs, [row[3] for row in counts], where="post", color="#538d29",
                      label=f"FCIT* ({len(trials)} trials)")
    success_axis.fill_between(cutoffs, lower, upper, step="post", color="#538d29", alpha=0.2)
    success_axis.set(ylim=(-3, 103), yticks=range(0, 101, 25), ylabel="Success [%]")
    success_axis.legend(loc="lower right")
    if solutions:
        times, costs = zip(*solutions)
        time_median = statistics.median(times)
        cost_median = statistics.median(costs)
        time_interval = median_interval(times)
        cost_interval = median_interval(costs)
        xerr = [[time_median - time_interval[0]], [time_interval[1] - time_median]] if time_interval else None
        yerr = [[cost_median - cost_interval[0]], [cost_interval[1] - cost_median]] if cost_interval else None
        cost_axis.errorbar(time_median, cost_median, xerr=xerr, yerr=yerr,
                           fmt="+", markersize=10, capsize=4, color="#538d29",
                           label="Median with 99% confidence intervals" if time_interval
                           else "Median (too few trials for finite 99% intervals)")
        cost_axis.legend(loc="upper right")
    else:
        cost_axis.text(0.5, 0.5, "No solutions", ha="center", transform=cost_axis.transAxes)
    cost_axis.set(xscale="log", xlim=(start, end), xlabel="Computation time [s]",
                  ylabel="Path cost (joint-space length)")
    for panel in (success_axis, cost_axis):
        panel.grid(True, which="major", alpha=0.3)
    combined_path = folder / "success_and_cost.png"
    combined.savefig(combined_path, dpi=180)
    cost_table_path = folder / "first_solution_cost.csv"
    with cost_table_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("computation_time_s", "first_solution_path_cost"))
        writer.writerows(sorted(solutions))
    print(f"Cost data saved to {cost_table_path}")
    print(f"Combined plot saved to {combined_path}")


if __name__ == "__main__":
    main()
