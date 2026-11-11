"""Plot cumulative first-solution success from a bookshelf experiment CSV."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to results.csv")
    args = parser.parse_args()
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

    table_path = folder / "success_over_time.csv"
    with table_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("time_s", "solved_by_time", "unsolved_by_time", "success_pct"))
        writer.writerows(counts)

    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 3.5), layout="constrained")
    axis = figure.subplots()
    axis.step(cutoffs, [row[3] for row in counts], where="post", color="#538d29",
              label=f"FCIT* ({len(trials)} trials)")
    axis.set(xscale="log", xlim=(start, end), ylim=(-3, 103),
             yticks=range(0, 101, 25), xlabel="Computation time [s]", ylabel="Success [%]")
    axis.grid(True, which="major", alpha=0.3)
    axis.legend(loc="lower right")
    plot_path = folder / "success_over_time.png"
    figure.savefig(plot_path, dpi=180)
    print(f"Counts saved to {table_path}")
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
