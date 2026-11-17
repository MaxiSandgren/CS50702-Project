Jan's fantastic friend codex try to make the project

## Experiment modes

Run all 100 problems five times, stopping each trial at its first solution:

```sh
python whateverpy.py
```

Run one problem 100 times and track improved path costs over time:

```sh
python whateverpy.py --mode convergence --problem 1
```

Both modes default to a 100-second per-trial planning limit. For a short
convergence check, use `--trials 10 --timeout 0.5`.
The first mode saves `success_and_cost.png`. Convergence mode saves
`convergence.png` and `convergence_over_time.csv` in its new results folder.
Each trial JSON stores the improvement timestamps and costs. Convergence
cost curves count unsolved runs as infinite cost, and omit nonfinite median
values or confidence bounds. The initial-solution cross summarizes successful
runs. Both plots use 99% confidence intervals.

The paper's precise single bookshelf-small query is not identified here.
Problem 1 is the demonstration choice. Current sampler offsets, software,
machine, and finite sample/iteration limits can differ from the paper.
Callbacks and path recovery contribute to recorded convergence timing.
The JSON status and `solve_elapsed_ms` record sample/iteration-limit exits.
When a trial exits early, its final incumbent is retained on the graph.

Replot a completed convergence run with:

```sh
python plot_convergence.py bookshelf_results/RUN_FOLDER
```


For cage problem 1, run:

```sh
python whateverpy.py --mode convergence --environment cage --problem 1
```

Cage runs save their JSON, CSV and graphs in `cage_results`.
The default environment remains `bookshelf_small`.


## Worker failure recovery

Convergence workers stop gracefully at 2048 MiB peak RAM by default. Set
`--memory-limit-mb` to choose a different limit. Such trials retain their
best path and history and record `memory_limit`. Unexpected worker exits
record `worker_exit` and preserve received improvements, with no final path
when the worker could not return it. The next trial then runs normally.

Progress is checkpointed atomically. To continue an interrupted experiment:

```sh
python whateverpy.py --resume cage_results/RUN_FOLDER
```

Saved trial JSON files are reused, the CSV is rebuilt, and an unfinished trial
is rerun. Resume loads the original experiment settings and records the new
memory guard in its history. A folder lock prevents simultaneous resume runs.
