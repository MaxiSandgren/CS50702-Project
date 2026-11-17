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


## Batch logs and stopping rules

`fcit.hh` can write one CSV line after every batch (time, batch, best cost,
samples) plus a last line saying why the run ended. After changing `fcit.hh`,
rebuild `local_fcit.abi3.so` inside the container:

```sh
git clone --depth 1 --branch v0.6.4 https://github.com/KavrakiLab/vamp.git /tmp/vamp-src
pip install /tmp/vamp-src -Ccmake.define.CMAKE_PROJECT_vamp_INCLUDE=$PWD/local_fcit.cmake
```

Log several problems with `--trace` (one CSV per trial in `RESULTS/traces`):

```sh
python whateverpy.py --mode convergence --environment cage --problems 1-20 --trials 5 --trace --skip-stride 1000000
```

Test the stopping rules on the logs, without rerunning the planner:

```sh
python stopping_rules.py cage_results/RUN_FOLDER bookshelf_results/RUN_FOLDER
```

Settings are picked on the odd problems and reported on the even ones. The
script prints the best setting per rule and saves `stopping_rules_results.csv`,
`stopping_rules_runs.csv` and `stopping_rules.png` in `stopping_results/`.
