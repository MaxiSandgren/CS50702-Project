# Bookshelf-small FCIT* experiment

Run `python whateverpy.py` in the existing VAMP container. This runs original
MotionBenchMaker Panda bookshelf-small problem 1 five times. Outputs go to a new
timestamped `bookshelf_results` folder. Open the HTML file in a Windows browser
for an animated collision-sphere replay; drag to rotate and scroll to zoom.

Use `python whateverpy.py --problem 57` for another original scene, or
`python whateverpy.py --all --no-replay` for all 100 scenes (500 trials).
Each call has a 100-second watchdog, so a full run can take considerable time.
The CSV is flushed after every trial. JSON files retain every returned path.

## Relationship to the research paper

Source: Wilson et al., *Nearest-Neighbourless Asymptotically Optimal Motion
Planning with Fully Connected Informed Trees (FCIT*)*, ICRA 2025, Section IV:
https://www.kavrakilab.org/publications/wilson2025-fcit.pdf

Matched: Panda 7-DoF model, original MBM bookshelf-small scene geometry and
start/goal queries, FCIT*, five trials per scene, and a 100-second timeout value.
All-scene mode selects all 100 original problems. No synthetic shelf geometry
or replacement start/goal positions are used. FCIT paths are not shortcut-smoothed.

This is NOT an exact numerical reproduction of Table I or Fig. 3:

- Uses installed VAMP 0.6.4, not a recovered paper-specific implementation.
- Batch size 1000 and iteration/sample caps 100000 are current-library defaults;
  the paper does not specify these exact values. The caps may terminate a run
  before 100 seconds; this is recorded as `iteration_or_sample_limit`.
- Uses Halton sampling with skip offsets 0, 100000, 200000, etc. These offsets
  are reproducible choices, not recovered paper seeds.
- The installed Python binding lacks a timed solve and convergence callbacks.
  A parent-process watchdog limits initial-solution calls. It cannot capture
  continuous anytime improvement; this script does not reproduce Fig. 3.
- Problem 1 is a demonstration choice, not an asserted match to the Fig. 3b query.
- Reported medians are among successful trials, explicitly labeled as such;
  they are not claimed to use the paper's aggregation/confidence intervals.
- Replay timing is illustrative; no trajectory velocity/acceleration profiling.

## Data provenance

https://github.com/KavrakiLab/vamp/blob/27cb9b66ebd8d101ce88d330f3b028398afc1cfa/resources/panda/problems.tar.bz2

Archive SHA-256:
`4e1509e99c53da769ee6a338cf1173ecf601e4ec16f24081d3894ef358243b69`

The bundled JSON contains the 100 bookshelf-small scene/request pairs only.
Conversion follows the upstream `resources/problem_tar_to_pkl_json.py` format:
joint names determine order, box dimensions are converted to half extents, and
object and primitive transforms are composed. The upstream `problem_dict_to_vamp`
function builds collision geometry. The replay draws shelf boxes and the Panda's
actual VAMP collision spheres. Cylinders are displayed with sphere stacks.

Validation performed: Python syntax; problem 1 trial 1 solved in the existing
container; every waypoint and motion segment of that solution passed VAMP
collision checks; replay HTML generated. Additional run checks were not executed.
