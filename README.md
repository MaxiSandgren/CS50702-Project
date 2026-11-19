# FCIT* replication and stopping rules (CS572)

## Setup (once per computer)

1. Start Docker Desktop. In VS Code, install the **Dev Containers** extension.
2. Open this folder in VS Code, press F1 and choose **Dev Containers: Reopen in Container**.
   The first time takes about 5 to 10 minutes: it installs everything and compiles our FCIT*.
   It is ready when the terminal prints `local_fcit is built and ready`.
3. Run all commands below in the VS Code terminal, inside the container.

After changing `fcit.hh`, compile it again with `bash build_fcit.sh`.

## Folders

| Folder | What is in it |
|---|---|
| `data/` | The MotionBenchMaker problems (bookshelf small and cage) |
| `results/replication/` | The finished replication runs (Table I, Fig. 3) |
| `results/runs/` | **New runs end up here**, one folder per scene and run |
| `results/stopping_rules/` | Output of `stopping_rules.py` |
| `results/old/` | Old test runs, kept for reference |

## Running FCIT*

First solution, all 100 problems, 5 trials each (Table I):

```sh
python whateverpy.py --environment cage
python plot_initial.py results/runs/cage/RUN_FOLDER/results.csv
```

Cost over time, one problem, 100 trials (Fig. 3):

```sh
python whateverpy.py --mode convergence --environment cage --problem 1
```

Default limits: 100 s per trial and 2048 MiB of RAM per trial (`--timeout`, `--memory-limit-mb`).
To continue a run that crashed or was stopped: `python whateverpy.py --resume results/runs/cage/RUN_FOLDER`

## Stopping rules

Log several problems with `--trace` (one CSV per trial, one line per batch):

```sh
python whateverpy.py --mode convergence --environment cage --problems 1-20 --trials 5 --trace --skip-stride 1000000
```

Test the four rules on the logs, without running the planner again:

```sh
python stopping_rules.py results/runs/cage/RUN_FOLDER results/runs/bookshelf_small/RUN_FOLDER
```

Settings are picked on the odd problems and reported on the even ones. The script prints the
best setting per rule and saves a table and a plot in `results/stopping_rules/`.

## Differences from the paper

The paper's seeds, its exact Fig. 3 problem and some FCIT* settings are not published, so we
use VAMP 0.6.4 defaults, Halton sampler offsets per trial and problem 1. Machines also differ.
