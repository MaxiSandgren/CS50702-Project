"""Run FCIT* on the original MotionBenchMaker Panda scenes (bookshelf_small or cage).

Run normally for problem 1, five trials and a Python 3D plot (PNG).
  python whateverpy.py --problem 57
  python whateverpy.py --all
  python whateverpy.py --scene cage --all --no-replay

Paper: https://www.kavrakilab.org/publications/wilson2025-fcit.pdf (Sec. IV).
This reproduces the scene and initial-solution task, NOT Fig. 3 convergence.
The paper does not specify exact sampler seeds or all FCIT tuning values.
Current VAMP defaults and reproducible Halton skip offsets are recorded below.
The Python binding has no timed solve: a separate process enforces a 100-second
wall-clock watchdog; the library also has finite sample/iteration limits.
"""
import argparse
import csv
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import statistics
import time
import traceback
import vamp

ROOT = Path(__file__).resolve().parent
# Scene name -> (problem file, results folder)
SCENES = {
    "bookshelf_small": (ROOT / "bookshelf_data" / "bookshelf_small.json", ROOT / "bookshelf_results"),
    "cage": (ROOT / "cage_data" / "cage.json", ROOT / "cage_results"),
}


def worker(connection, problem, config, trial):
    try:
        from local_fcit import solve

        #Checking Validity
        env = vamp.problem_dict_to_vamp(problem)
        start, goals = problem["start"], problem["goals"]
        if not vamp.panda.validate(start, env) or not all(vamp.panda.validate(g, env) for g in goals):
            connection.send(("result", {"status": "invalid_start_or_goal", "solved": False}))
            return
        settings = vamp.FCITSettings(vamp.FCITNeighborParams(vamp.panda.dimension(), vamp.panda.space_measure()))
        settings.batch_size = config["batch_size"]
        settings.max_iterations = config["max_iterations"]
        settings.max_samples = config["max_samples"]
        settings.optimize = False  # Report initial solution, with no shortcutting.
        sampler = vamp.panda.halton()
        sampler.skip(trial * config["skip_stride"])
        connection.send(("ready", None))
        connection.recv()  # Parent starts the watchdog immediately before solve.
        result = solve(start, goals, env, settings, sampler)
        result_data = {
            "solved": bool(result.solved), "status": "solved" if result.solved else "iteration_or_sample_limit",
            "planning_ms": result.nanoseconds / 1e6,
            "iterations": int(result.iterations), "graph_size": list(result.size),
            "path_length": float(result.path.cost()) if result.solved else None,
            "path": result.path.numpy().tolist() if result.solved else [],
        }
        # Report immediately: validation and visualization are outside planner timing.
        connection.send(("result", result_data))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_trial(problem, config, trial):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, problem, config, trial))
    process.start()
    child.close()
    try:
        if not parent.poll(30):
            raise RuntimeError("Planner worker failed to initialize within 30 seconds.")
        kind, payload = parent.recv()
        if kind == "ready":
            parent.send("solve")
            if not parent.poll(config["timeout_seconds"]):
                return {"status": "timeout", "solved": False, "path": []}
            kind, payload = parent.recv()
        if kind == "error":
            raise RuntimeError(payload)
        if kind != "result":
            raise RuntimeError(f"Unexpected worker message: {kind}")
        if payload.get("planning_ms", 0) > 1000 * config["timeout_seconds"]:
            return {"status": "timeout", "solved": False, "path": []}
        return payload
    finally:
        if process.is_alive():
            process.join(timeout=0.1)
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()


def replay(problem, path, destination):
    """Save a Python 3D plot of the scene at the start, midpoint and goal."""
    import vamp
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Works inside Docker without a desktop display.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    env = vamp.problem_dict_to_vamp(problem)
    if not all(vamp.panda.validate(q, env) for q in path):
        raise RuntimeError("Returned path contains a colliding waypoint.")
    if not all(vamp.panda.validate_motion(a, b, env) for a, b in zip(path, path[1:])):
        raise RuntimeError("Returned path contains an invalid motion segment.")
    plan = vamp.panda.Path()
    for q in path:
        plan.append(q)
    plan.interpolate_to_n_states(120)
    configurations = plan.numpy()
    frames = [np.array([[s.x, s.y, s.z, s.r] for s in vamp.panda.fk(q)])
              for q in configurations]

    def transform(points, obj):
        x, y, z, w = obj["orientation_quat_xyzw"]
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return np.asarray(points) @ rotation.T + obj["position"]

    boxes = []
    face_ids = [[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5]]
    for obj in problem["box"]:
        h = obj["half_extents"]
        vertices = transform([[h[0]*(1 if i&1 else -1), h[1]*(1 if i&2 else -1),
                               h[2]*(1 if i&4 else -1)] for i in range(8)], obj)
        boxes.append(vertices)
    all_points = np.concatenate([f[:, :3] for f in frames] + boxes)
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    radius = max(np.ptp(all_points, axis=0)) / 2 + .12
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 12), np.linspace(0, np.pi, 8))
    unit_sphere = np.stack([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)], axis=-1)

    def sphere(ax, position, r, color):
        points = unit_sphere*r + position
        ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], color=color,
                        linewidth=0, alpha=.95, shade=True)

    fig = plt.figure(figsize=(15, 6), layout="constrained")
    for panel, (index, title) in enumerate([(0, "Start"), (len(frames)//2, "Mid-path"), (len(frames)-1, "Goal")], 1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        for vertices in boxes:
            ax.add_collection3d(Poly3DCollection([vertices[ids] for ids in face_ids],
                                facecolors="#9caaba", edgecolors="#596c7d", alpha=.18, linewidths=.6))
        for obj in problem["cylinder"]:
            theta, height = np.meshgrid(np.linspace(0, 2*np.pi, 24), [-obj["length"]/2, obj["length"]/2])
            local = np.stack([obj["radius"]*np.cos(theta), obj["radius"]*np.sin(theta), height], axis=-1)
            points = transform(local, obj)
            ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], color="#df9b38")
            ax.add_collection3d(Poly3DCollection([points[0], points[1]], facecolors="#df9b38"))
        for obj in problem["sphere"]:
            sphere(ax, obj["position"], obj["radius"], "#df9b38")
        for x, y, z, r in frames[index]:
            sphere(ax, [x, y, z], r, "#148e90")
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=title)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=24, azim=-125)
    fig.suptitle(f"Panda / {problem['problem'].replace('_', ' ')} / problem {problem['index']} — FCIT* initial path", fontsize=16)
    fig.supxlabel("Robot shown using VAMP collision spheres. Snapshots along the path, not execution time.", fontsize=10)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def cpu_name():
    """Processor model, so the results say which computer they come from."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=sorted(SCENES), default="bookshelf_small", help="MotionBenchMaker scene")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--problem", type=int, default=1, help="Original MBM problem index, 1..100 (default: 1)")
    selection.add_argument("--all", action="store_true", help="Run all 100 scenes, five trials each by default")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=100.0, help="Per-call watchdog seconds (paper: 100)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Installed VAMP default, not specified in paper")
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--skip-stride", type=int, default=100000, help="Halton offset between trials; not original paper seeds")
    parser.add_argument("--no-replay", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.timeout <= 0 or args.batch_size < 1 or args.max_iterations < 1:
        parser.error("Trials, timeout, batch size and max iterations must be positive.")
    if args.max_samples < args.batch_size + 2 or args.skip_stride < 0:
        parser.error("max-samples must be at least batch-size + 2; skip-stride cannot be negative.")
    try:
        version = importlib.metadata.version("vamp-planner")
    except importlib.metadata.PackageNotFoundError:
        parser.exit(1, "Run this script inside the working VAMP container, not Windows Anaconda.\n")
    data_file, results_root = SCENES[args.scene]
    dataset = json.loads(data_file.read_text(encoding="utf-8"))
    problems = dataset["problems"] if args.all else [p for p in dataset["problems"] if p["index"] == args.problem]
    if not problems:
        parser.error("Problem index must be between 1 and 100.")
    config = dict(timeout_seconds=args.timeout, batch_size=args.batch_size, max_iterations=args.max_iterations,
                  max_samples=args.max_samples, skip_stride=args.skip_stride, optimize=False, sampler="halton")
    out = results_root / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "paper": "https://www.kavrakilab.org/publications/wilson2025-fcit.pdf", "dataset_source": dataset["source"],
        "archive_sha256": dataset["archive_sha256"], "vamp_version": version,
        "platform": platform.platform(), "cpu": cpu_name(), "scene": args.scene, "config": config, "trials_per_problem": args.trials,
        "problem_indices": [p["index"] for p in problems],
        "limitations": ["Initial-solution FCIT only; not the Fig. 3 anytime convergence experiment.",
                        "Exact paper seeds and tuning are unavailable; current-library defaults and Halton offsets are explicit.",
                        "100-second watchdog is external; finite sample/iteration caps may stop planning earlier.",
                        "Current VAMP and this computer differ from the paper's software/hardware; timings need not match."],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"FCIT* / Panda / {args.scene}: {len(problems)} problem(s), {args.trials} trials each", flush=True)
    print(f"VAMP {version}; {args.timeout:g}s watchdog; batch={args.batch_size}; samples/iterations={args.max_samples}/{args.max_iterations}")
    print("Original MBM geometry and queries. Current-library defaults and Halton offsets; NOT an exact paper replication.", flush=True)
    rows = []
    fields = ["problem", "trial", "halton_skip", "solved", "status", "planning_ms", "path_length", "iterations"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for problem in problems:
            replay_saved = False
            for trial in range(args.trials):
                result = run_trial(problem, config, trial)
                result.update(problem=problem["index"], trial=trial + 1, halton_skip=trial * args.skip_stride)
                rows.append(result)
                writer.writerow(result)
                csvfile.flush()
                (out / f"problem_{problem['index']:04d}_trial_{trial+1}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                detail = f"{result['planning_ms']:.3f} ms, path length {result['path_length']:.4f}" if result["solved"] else result["status"]
                print(f"  Problem {problem['index']:03d}, trial {trial+1}: {detail}", flush=True)
                if result["solved"] and not replay_saved and not args.no_replay:
                    destination = out / f"{args.scene}_{problem['index']:04d}.png"
                    replay(problem, result["path"], destination)
                    print(f"  Python plot: {destination}", flush=True)
                    replay_saved = True
    solved = [r for r in rows if r["solved"]]
    print(f"\nSolved {len(solved)}/{len(rows)} trials ({100*len(solved)/len(rows):.1f}%).")
    if solved:
        print(f"Median among solved trials: {statistics.median(r['planning_ms'] for r in solved):.3f} ms; "
              f"path length {statistics.median(r['path_length'] for r in solved):.4f}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
