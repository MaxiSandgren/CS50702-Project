"""Test stopping rules offline on FCIT* batch logs.

Each rule replays a log batch by batch and only sees the batches so far, so it
stops at the same batch it would have stopped at inside the planner.

  python stopping_rules.py bookshelf_results/RUN_FOLDER cage_results/RUN_FOLDER

The folders must come from whateverpy.py --mode convergence --trace.
Settings are picked on half the runs (tune) and reported on the other half (test).
Best setting: most time saved with median extra path cost at most --max-extra-cost (2 %).
Writes stopping_rules_results.csv, stopping_rules_runs.csv and stopping_rules.png to stopping_results/.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parent


# ---------- Reading the logs ----------

def read_trace(path):
    """Return the batches as (time_s, cost) and the run's end time and reason."""
    batches, end_time, end_reason = [], None, "no_end_line (killed)"
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["event"] == "batch":
                batches.append((float(row["time_s"]), float(row["best_cost"])))
            elif row["event"].startswith("end_"):
                end_time, end_reason = float(row["time_s"]), row["event"][4:]
    if end_time is None and batches:
        end_time = batches[-1][0]
    return batches, end_time, end_reason


def lower_bound(problem):
    """Straight-line distance from start to goal in joint space. No path is shorter."""
    return math.dist(problem["start"], problem["goals"][0])


def load_runs(folders):
    runs = []
    for folder in folders:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        scene = metadata["config"].get("environment", "bookshelf_small")
        dataset = json.loads((ROOT / "bookshelf_data" / f"{scene}.json").read_text(encoding="utf-8"))
        problems = {p["index"]: p for p in dataset["problems"]}
        traces = sorted((folder / "traces").glob("problem_*_trial_*.csv"))
        if not traces:
            raise SystemExit(f"No batch logs in {folder / 'traces'}. Run whateverpy.py with --trace.")
        for path in traces:
            _, problem, _, trial = path.stem.split("_")
            batches, end_time, end_reason = read_trace(path)
            runs.append(dict(scene=scene, problem=int(problem), trial=int(trial),
                             batches=batches, end_time=end_time, end_reason=end_reason,
                             lower_bound=lower_bound(problems[int(problem)])))
    return runs


# ---------- The rules ----------
# Each rule gets the batches so far as (time_s, cost) and the run, and returns
# True to stop. Rules are only asked once the run has a solution.

def plateau(epsilon, patience, min_batches):
    """Stop when the cost has not dropped by epsilon (relative) for `patience` batches.
    Same logic as Maxi's C++ version: compare with the last real improvement."""
    def rule(history, run):
        solved = [cost for _, cost in history if math.isfinite(cost)]
        reference, stagnant = solved[0], 0
        for cost in solved[1:]:
            if (reference - cost) / reference >= epsilon:
                reference, stagnant = cost, 0
            else:
                stagnant += 1
        return len(solved) - 1 >= min_batches and stagnant >= patience
    rule.name = "Plateau"
    rule.setting = f"eps={epsilon:g}, m={patience}, min={min_batches}"
    return rule


def gain_per_second(rate, window):
    """Stop when the relative cost saved per second over the last `window` batches is below `rate`."""
    def rule(history, run):
        solved = [(t, cost) for t, cost in history if math.isfinite(cost)]
        if len(solved) <= window:
            return False
        (t0, c0), (t1, c1) = solved[-1 - window], solved[-1]
        return (c0 - c1) / c1 / max(t1 - t0, 1e-9) < rate
    rule.name = "Gain per second"
    rule.setting = f"rate={rate:g}/s, window={window}"
    return rule


def gap_to_lower_bound(gap):
    """Stop when the cost is within `gap` (relative) of the straight-line distance."""
    def rule(history, run):
        return history[-1][1] <= (1 + gap) * run["lower_bound"]
    rule.name = "Gap to lower bound"
    rule.setting = f"gap={gap:g}"
    return rule


def curve_fit(min_gain, min_points=5):
    """Fit cost = a + b / batch to the batches since the first solution. The fitted a
    is the predicted final cost. Stop when the predicted remaining gain is small."""
    def rule(history, run):
        points = [(k + 1, cost) for k, (_, cost) in enumerate(history) if math.isfinite(cost)]
        if len(points) < min_points:
            return False
        k, cost = np.array(points, dtype=float).T
        b, a = np.polyfit(1 / k, cost, 1)
        current = cost[-1]
        return b <= 0 or (current - a) / current < min_gain
    rule.name = "Curve fit"
    rule.setting = f"gain={min_gain:g}"
    return rule


def all_rules():
    rules = [plateau(e, m, mb) for e in (0.005, 0.01, 0.02) for m in (3, 5, 10) for mb in (0, 10)]
    rules += [gain_per_second(r, w) for r in (1, 0.1, 0.01, 0.001) for w in (3, 5, 10)]
    rules += [gap_to_lower_bound(g) for g in (0.05, 0.1, 0.2, 0.3, 0.5)]
    rules += [curve_fit(g) for g in (0.002, 0.005, 0.01, 0.02, 0.05)]
    return rules


# ---------- Replay ----------

def replay(rule, run):
    """Return (time, cost, stopped_early, seconds spent in the rule per check)."""
    history, spent, checks = [], 0.0, 0
    for batch in run["batches"]:
        history.append(batch)
        if not math.isfinite(batch[1]):
            continue
        tick = time.perf_counter()
        stop = rule(history, run)
        spent += time.perf_counter() - tick
        checks += 1
        if stop:
            return batch[0], batch[1], True, spent / checks
    return run["end_time"], run["batches"][-1][1], False, spent / max(checks, 1)


def baselines(run):
    first = next(batch for batch in run["batches"] if math.isfinite(batch[1]))
    return {"First solution": (first[0], first[1]),
            "Full run": (run["end_time"], run["batches"][-1][1])}


def summarize(rows):
    saved = [r["time_saved_pct"] for r in rows]
    extra = [r["extra_cost_pct"] for r in rows]
    return dict(runs=len(rows),
                stopped_early_pct=100 * statistics.mean(r["stopped_early"] for r in rows),
                median_time_saved_pct=statistics.median(saved),
                mean_time_saved_pct=statistics.mean(saved),
                median_extra_cost_pct=statistics.median(extra),
                p90_extra_cost_pct=float(np.percentile(extra, 90)),
                rule_us_per_check=1e6 * statistics.mean(r["rule_seconds"] for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+", help="Results folders with a traces/ subfolder")
    parser.add_argument("--max-extra-cost", type=float, default=2.0,
                        help="Best setting = most time saved with median extra cost at most this %% (default 2)")
    parser.add_argument("--out", type=Path, default=Path("stopping_results"), help="Output folder (default: stopping_results)")
    args = parser.parse_args(argv)

    runs = [run for run in load_runs(args.folders)]
    unsolved = [run for run in runs if not any(math.isfinite(c) for _, c in run["batches"])]
    runs = [run for run in runs if run not in unsolved]
    for scene in sorted({run["scene"] for run in runs}):
        scene_runs = [run for run in runs if run["scene"] == scene]
        # Tune on odd problems, test on even ones. With only one problem, split by trial.
        by_problem = len({run["problem"] for run in scene_runs}) > 1
        for run in scene_runs:
            run["split"] = "tune" if (run["problem"] if by_problem else run["trial"]) % 2 else "test"

    per_run = []
    for run in runs:
        full_time, full_cost = run["end_time"], run["batches"][-1][1]
        results = [(name, "", t, c, name == "First solution", 0.0) for name, (t, c) in baselines(run).items()]
        results += [(rule.name, rule.setting, *replay(rule, run)) for rule in all_rules()]
        for name, setting, t, c, early, seconds in results:
            per_run.append(dict(
                scene=run["scene"], split=run["split"], problem=run["problem"], trial=run["trial"],
                rule=name, setting=setting, stop_time_s=t, stop_cost=c, stopped_early=early,
                full_time_s=full_time, full_cost=full_cost, end_reason=run["end_reason"],
                time_saved_pct=100 * (1 - t / full_time), extra_cost_pct=100 * (c / full_cost - 1),
                rule_seconds=seconds))

    summary = []
    groups = {}
    for row in per_run:
        groups.setdefault((row["scene"], row["rule"], row["setting"], row["split"]), []).append(row)
    for (scene, rule, setting, split), rows in sorted(groups.items()):
        summary.append(dict(scene=scene, rule=rule, setting=setting, split=split, **summarize(rows)))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stopping_rules_runs.csv", per_run), ("stopping_rules_results.csv", summary)):
        with (args.out / name).open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Pick each rule's best setting on the tune half, report it on the test half.
    best = []
    for scene in sorted({row["scene"] for row in summary}):
        print(f"\n{scene}: {sum(r['scene'] == scene for r in runs)} solved runs"
              f" ({sum(r['scene'] == scene for r in unsolved)} unsolved left out)")
        print(f"  {'rule':<19} {'best setting (on tune)':<28} {'time saved':>11} {'extra cost':>11}"
              f" {'p90 extra':>10} {'stopped':>8} {'us/check':>9}")
        order = ["First solution", "Full run", "Plateau", "Gain per second", "Gap to lower bound", "Curve fit"]
        for rule in order:
            tune = [row for row in summary if row["scene"] == scene and row["rule"] == rule and row["split"] == "tune"]
            allowed = [row for row in tune if row["median_extra_cost_pct"] <= args.max_extra_cost]
            pick = (max(allowed, key=lambda row: (row["median_time_saved_pct"], -row["median_extra_cost_pct"]))
                    if allowed else min(tune, key=lambda row: row["median_extra_cost_pct"]))
            test = next((row for row in summary if row["scene"] == scene and row["rule"] == rule
                         and row["setting"] == pick["setting"] and row["split"] == "test"), pick)
            best.append(test)
            print(f"  {rule:<19} {pick['setting'] or '-':<28} {test['median_time_saved_pct']:>10.1f}%"
                  f" {test['median_extra_cost_pct']:>10.2f}% {test['p90_extra_cost_pct']:>9.2f}%"
                  f" {test['stopped_early_pct']:>7.0f}% {test['rule_us_per_check']:>9.1f}")
    print("\nTime saved and extra cost are medians over the test runs, compared with the full run.")
    plot(summary, best, args.out / "stopping_rules.png")
    print(f"Saved {args.out / 'stopping_rules_results.csv'}, stopping_rules_runs.csv and stopping_rules.png")


def plot(summary, best, destination):
    """Time saved against extra cost for every setting on the test runs, one panel per scene."""
    from matplotlib.figure import Figure

    colors = {"Plateau": "#2a78d6", "Gain per second": "#eb6834",
              "Gap to lower bound": "#1baf7a", "Curve fit": "#eda100"}
    markers = {"Plateau": "o", "Gain per second": "s", "Gap to lower bound": "^", "Curve fit": "D"}
    scenes = sorted({row["scene"] for row in summary})
    figure = Figure(figsize=(6 * len(scenes), 4.5), layout="constrained")
    axes = figure.subplots(1, len(scenes), squeeze=False)[0]
    for axis, scene in zip(axes, scenes):
        test = [row for row in summary if row["scene"] == scene and row["split"] == "test"]
        for rule, color in colors.items():
            points = [row for row in test if row["rule"] == rule]
            if not points:
                continue
            axis.scatter([row["median_extra_cost_pct"] for row in points],
                         [row["median_time_saved_pct"] for row in points],
                         s=40, color=color, marker=markers[rule], edgecolors="white",
                         linewidths=1.5, label=rule, zorder=3)
        picks = [row for row in best if row["scene"] == scene and row["rule"] in colors]
        axis.scatter([row["median_extra_cost_pct"] for row in picks],
                     [row["median_time_saved_pct"] for row in picks], s=160, facecolors="none",
                     edgecolors="#3a3a38", linewidths=1.2, label="Best setting per rule", zorder=4)
        for row in test:
            if row["rule"] in ("First solution", "Full run"):
                axis.scatter(row["median_extra_cost_pct"], row["median_time_saved_pct"],
                             s=60, color="#888888", marker="x", zorder=3)
                axis.annotate(row["rule"], (row["median_extra_cost_pct"], row["median_time_saved_pct"]),
                              xytext=(6, -10), textcoords="offset points", fontsize=8, color="#3a3a38")
        axis.set(title=scene.replace("_", " ").capitalize(), xlabel="Extra path cost vs full run [%] (median)",
                 ylabel="Time saved vs full run [%] (median)", ylim=(-5, 105))
        axis.grid(True, alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(loc="lower right", fontsize=8, frameon=False)
    figure.suptitle("Stopping rules on the test runs (up and left is better)")
    figure.savefig(destination, dpi=180)


if __name__ == "__main__":
    main()
