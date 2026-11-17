"""Run FCIT* on the original MotionBenchMaker Panda bookshelf_small scenes.



Run normally for all 100 problems, five trials each, then plot success and cost.

  python whateverpy.py --problem 57

  python whateverpy.py --all



Paper: https://www.kavrakilab.org/publications/wilson2025-fcit.pdf (Sec. IV).

Use --mode convergence --problem 1 for one problem with 100 optimizing trials.

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





def worker(connection, problem, config, trial):
    try:
        import vamp
        from local_fcit import solve
        env = vamp.problem_dict_to_vamp(problem)
        start, goals = problem["start"], problem["goals"]
        if not vamp.panda.validate(start, env) or not all(vamp.panda.validate(g, env) for g in goals):
            connection.send(("result", {"status": "invalid_start_or_goal", "solved": False}))
            return
        settings = vamp.FCITSettings(vamp.FCITNeighborParams(vamp.panda.dimension(), vamp.panda.space_measure()))
        settings.batch_size = config["batch_size"]
        settings.max_iterations = config["max_iterations"]
        settings.max_samples = config["max_samples"]
        settings.optimize = config["optimize"]
        improvements = []
        memory_exceeded = False
        def stop_for_memory():
            nonlocal memory_exceeded
            import resource
            peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            memory_exceeded = peak_mb >= config.get("memory_limit_mb", 2048)
            return memory_exceeded
        def record_improvement(seconds, cost):
            event = {"time_s": seconds, "cost": cost}
            improvements.append(event)
            connection.send(("progress", event))
        sampler = vamp.panda.halton()
        sampler.skip(trial * config["skip_stride"])
        connection.send(("ready", None))
        connection.recv()  # Parent starts the watchdog immediately before solve.
        if config["optimize"]:
            from local_fcit import solve_anytime
            result = solve_anytime(start, goals, env, settings, sampler,
                                   config["timeout_seconds"], record_improvement, stop_for_memory)
        else:
            result = solve(start, goals, env, settings, sampler)
        result_data = {
            "solved": bool(result.solved), "status": "solved" if result.solved else "iteration_or_sample_limit",
            "planning_ms": result.nanoseconds / 1e6,
            "iterations": int(result.iterations), "graph_size": list(result.size),
            "path_length": float(result.path.cost()) if result.solved else None,
            "path": result.path.numpy().tolist() if result.solved else [],
        }
        if config["optimize"]:
            elapsed_ms = result_data["planning_ms"]
            result_data.update(
                improvements=improvements, solve_elapsed_ms=elapsed_ms,
                final_path_length=result_data["path_length"],
                planning_ms=improvements[0]["time_s"] * 1000 if improvements else None,
                path_length=improvements[0]["cost"] if improvements else None,
                status="memory_limit" if memory_exceeded else
                       "time_limit" if elapsed_ms >= 1000 * config["timeout_seconds"]
                       else "iteration_or_sample_limit")
        result_data["memory_limit_mb"] = config.get("memory_limit_mb", 2048)
        # Report immediately: validation and visualization are outside planner timing.
        connection.send(("result", result_data))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()







def save_trial(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_trial(problem, config, trial, checkpoint_path=None):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, problem, config, trial))
    label = f"Problem {problem['index']:03d}, trial {trial + 1}"
    if config["optimize"]:
        print(f"  {label}: initializing planner and sampler...", flush=True)
    process.start()
    child.close()
    improvements = []
    started = None

    def interrupted_result(status):
        return {
            "status": status, "solved": bool(improvements), "path": [],
            "path_available": False, "improvements": improvements,
            "planning_ms": improvements[0]["time_s"] * 1000 if improvements else None,
            "path_length": improvements[0]["cost"] if improvements else None,
            "final_path_length": improvements[-1]["cost"] if improvements else None,
            "solve_elapsed_ms": (time.monotonic() - started) * 1000 if started else None,
            "worker_exit_code": process.exitcode,
            "memory_limit_mb": config.get("memory_limit_mb", 2048),
        }

    try:
        if not parent.poll(30):
            raise RuntimeError("Planner worker failed to initialize within 30 seconds.")
        kind, payload = parent.recv()
        if kind == "ready":
            if config["optimize"]:
                print(f"  {label}: searching, budget {config['timeout_seconds']:g}s", flush=True)
            started = time.monotonic()
            deadline = started + config["timeout_seconds"] + (5 if config["optimize"] else 0)
            parent.send("solve")
            while True:
                remaining = max(0, deadline - time.monotonic())
                if not parent.poll(min(remaining, 5)):
                    if time.monotonic() < deadline:
                        if config["optimize"]:
                            best = (f"best cost {improvements[-1]['cost']:.4f}"
                                    if improvements else "searching for first solution")
                            print(f"  {label}: {time.monotonic() - started:.1f}s elapsed, {best}",
                                  flush=True)
                        continue
                    return interrupted_result("watchdog_timeout")
                kind, payload = parent.recv()
                if kind != "progress":
                    break
                improvements.append(payload)
                if checkpoint_path is not None:
                    save_trial(checkpoint_path, interrupted_result("in_progress"))
                print(f"  {label}: improvement at {payload['time_s']:.3f}s, "
                      f"best cost {payload['cost']:.4f}", flush=True)
        if kind == "error":
            raise RuntimeError(payload)
        if kind != "result":
            raise RuntimeError(f"Unexpected worker message: {kind}")
        if not config["optimize"] and (payload.get("planning_ms") or 0) > 1000 * config["timeout_seconds"]:
            return {"status": "timeout", "solved": False, "path": []}
        return payload
    except (EOFError, BrokenPipeError, ConnectionResetError):
        process.join(timeout=1)
        print(f"  {label}: worker exited (code {process.exitcode}); "
              "preserving received improvements and continuing.", flush=True)
        return interrupted_result("worker_exit")
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

    fig.suptitle(f"Panda / {problem['problem'].replace('_', ' ')} / problem {problem['index']} â€” FCIT* initial path", fontsize=16)

    fig.supxlabel("Robot shown using VAMP collision spheres. Snapshots along the path, not execution time.", fontsize=10)

    fig.savefig(destination, dpi=160)

    plt.close(fig)





def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument("--problem", type=int, default=None, help="Run one original MBM problem, 1..100 (default: all 100)")

    selection.add_argument("--all", action="store_true", help="Run all 100 scenes, five trials each by default")

    parser.add_argument("--mode", choices=("initial", "convergence"), default="initial",

                        help="initial: all problems × 5 trials; convergence: one problem × 100 trials")

    parser.add_argument("--environment", choices=("bookshelf_small", "cage"),
                        default="bookshelf_small", help="MotionBenchMaker environment")
    parser.add_argument("--trials", type=int, default=None)

    parser.add_argument("--timeout", type=float, default=100.0, help="Per-call watchdog seconds (paper: 100)")

    parser.add_argument("--batch-size", type=int, default=1000, help="Installed VAMP default, not specified in paper")

    parser.add_argument("--max-iterations", type=int, default=100000)

    parser.add_argument("--max-samples", type=int, default=100000)

    parser.add_argument("--skip-stride", type=int, default=100000, help="Halton offset between trials; not original paper seeds")

    parser.add_argument("--no-replay", action="store_true", default=True, help="Skip scene replay (default)")

    parser.add_argument("--replay", dest="no_replay", action="store_false", help="Also save scene replays")

    parser.add_argument("--resume", type=Path, help="Resume a saved results folder")
    parser.add_argument("--memory-limit-mb", type=float, default=2048,
                        help="Peak convergence-worker RAM limit in MiB (default: 2048)")
    args = parser.parse_args()
    resume_metadata = None
    if args.resume is not None:
        args.resume = args.resume.resolve()
        resume_metadata = json.loads((args.resume / "metadata.json").read_text(encoding="utf-8"))
        saved = resume_metadata["config"]
        for key in ("batch_size", "max_iterations", "max_samples", "skip_stride", "mode", "environment"):
            if key in saved:
                setattr(args, key, saved[key])
        args.timeout = saved["timeout_seconds"]
        args.trials = resume_metadata["trials_per_problem"]
        indices = resume_metadata["problem_indices"]
        args.problem = indices[0] if len(indices) == 1 else None
        args.all = len(indices) > 1
    if args.mode == "convergence":
        if args.all:
            parser.error("Convergence mode uses one problem. Select it with --problem.")
        if args.problem is None:
            args.problem = 1
    if args.trials is None:
        args.trials = 100 if args.mode == "convergence" else 5
    if not 0 < args.memory_limit_mb < float("inf"):
        parser.error("The memory limit must be positive and finite.")
    if args.trials < 1 or args.timeout <= 0 or args.batch_size < 1 or args.max_iterations < 1:
        parser.error("Trials, timeout, batch size and max iterations must be positive.")
    if args.max_samples < args.batch_size + 2 or args.skip_stride < 0:
        parser.error("max-samples must be at least batch-size + 2; skip-stride cannot be negative.")
    from plot_success import main as plot_results
    try:
        version = importlib.metadata.version("vamp-planner")
    except importlib.metadata.PackageNotFoundError:
        parser.exit(1, "Run this script inside the working VAMP container.\n")
    dataset_path = ROOT / "bookshelf_data" / f"{args.environment}.json"
    if not dataset_path.is_file():
        dataset_path = ROOT / f"{args.environment}.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    indices = resume_metadata["problem_indices"] if resume_metadata else (
        [p["index"] for p in dataset["problems"]] if args.all or args.problem is None else [args.problem])
    problems = [p for p in dataset["problems"] if p["index"] in indices]
    if not problems or len(problems) != len(indices):
        parser.error("Problem index must be between 1 and 100.")
    config = dict(timeout_seconds=args.timeout, batch_size=args.batch_size,
                  max_iterations=args.max_iterations, max_samples=args.max_samples,
                  skip_stride=args.skip_stride, optimize=args.mode == "convergence",
                  mode=args.mode, environment=args.environment, sampler="halton",
                  memory_limit_mb=args.memory_limit_mb)
    out = args.resume if args.resume is not None else (
        ROOT / ("cage_results" if args.environment == "cage" else "bookshelf_results")
        / time.strftime("%Y%m%d-%H%M%S"))
    if args.resume is None:
        out.mkdir(parents=True, exist_ok=False)
    import fcntl
    run_lock = (out / ".run.lock").open("a")
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This results folder already has a running experiment.")
    rows = []
    completed = set()
    if args.resume is not None:
        for problem in problems:
            for trial in range(args.trials):
                path = out / f"problem_{problem['index']:04d}_trial_{trial+1}.json"
                if path.is_file():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result.get("problem") != problem["index"] or result.get("trial") != trial + 1:
                        parser.error(f"Trial identifiers do not match {path.name}.")
                    rows.append(result)
                    completed.add((problem["index"], trial + 1))
        print(f"Resuming {out}: keeping {len(completed)} saved trials.", flush=True)
    if resume_metadata is not None:
        metadata = resume_metadata
        metadata.setdefault("resume_history", []).append(
            {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
             "memory_limit_mb": args.memory_limit_mb, "completed_trials": len(completed)})
    else:
        metadata = {
            "paper": "https://www.kavrakilab.org/publications/wilson2025-fcit.pdf",
            "dataset_source": dataset["source"], "archive_sha256": dataset["archive_sha256"],
            "vamp_version": version, "planner": "local_fcit (compiled from fcit.hh)",
            "platform": platform.platform(), "config": config,
            "trials_per_problem": args.trials, "problem_indices": indices,
            "limitations": [
                "Current machine, software, sampler offsets, and finite planner limits differ from the paper.",
                "Convergence callbacks and path recovery contribute to timing.",
                "Convergence trials stop at the time, sample/iteration, or peak-memory limit.",
                "Partial worker-exit results can contain cost history without a returned path.",
            ],
        }
    save_trial(out / "metadata.json", metadata)
    print(f"FCIT* / Panda / {args.environment}: {len(problems)} problem(s), {args.trials} trials each", flush=True)
    print(f"VAMP {version}; {args.timeout:g}s budget; samples/iterations="
          f"{args.max_samples}/{args.max_iterations}; convergence RAM limit {args.memory_limit_mb:g} MiB", flush=True)
    fields = ["problem", "trial", "halton_skip", "solved", "status", "planning_ms", "path_length", "iterations"]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        csvfile.flush()
        for problem in problems:
            replay_saved = False
            for trial in range(args.trials):
                if (problem["index"], trial + 1) in completed:
                    continue
                checkpoint = out / f".problem_{problem['index']:04d}_trial_{trial+1}.checkpoint.json"
                result = run_trial(problem, config, trial, checkpoint_path=checkpoint)
                result.update(problem=problem["index"], trial=trial + 1, halton_skip=trial * args.skip_stride)
                save_trial(out / f"problem_{problem['index']:04d}_trial_{trial+1}.json", result)
                checkpoint.unlink(missing_ok=True)
                rows.append(result)
                writer.writerow(result)
                csvfile.flush()
                detail = (f"{result['planning_ms']:.3f} ms, path length {result['path_length']:.4f}"
                          if result["solved"] else result["status"])
                if config["optimize"]:
                    elapsed = result.get("solve_elapsed_ms")
                    duration = f"{elapsed / 1000:.3f}s" if elapsed is not None else "unknown duration"
                    final_cost = result.get("final_path_length")
                    cost = f"{final_cost:.4f}" if final_cost is not None else "none"
                    detail = f"finished after {duration}; best cost {cost}; {result['status']}"
                print(f"  Problem {problem['index']:03d}, trial {trial+1}: {detail}", flush=True)
                if result.get("path") and not replay_saved and not args.no_replay:
                    destination = out / f"{args.environment}_{problem['index']:04d}.html"
                    replay(problem, result["path"], destination)
                    replay_saved = True
    solved = [r for r in rows if r["solved"]]
    print(f"\nSolved {len(solved)}/{len(rows)} trials ({100*len(solved)/len(rows):.1f}%).")
    if solved:
        print(f"Median first solution among solved trials: "
              f"{statistics.median(r['planning_ms'] for r in solved):.3f} ms; "
              f"path length {statistics.median(r['path_length'] for r in solved):.4f}")
    print(f"Results: {out}")
    if args.mode == "convergence":
        from plot_convergence import main as plot_convergence
        plot_convergence([str(out)])
    else:
        plot_results([str(out / "results.csv")])








if __name__ == "__main__":

    main()

