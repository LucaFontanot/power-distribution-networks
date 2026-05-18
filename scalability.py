import json
import os
import time as _time
from datetime import datetime

import numpy as np
from generator import Generator
from plot import plot_scalability_results
from solver.LF_SCF import solve_lf_scf
from solver.LF_SE import solve_lf_se
from solver.OL_SE import solve_ol_se, solve_ol_se_cc

_ALGORITHMS = [
    ("LF-SCF",   solve_lf_scf),
    ("LF-SE",    solve_lf_se),
    ("OL-SE",    solve_ol_se),
    ("OL-SE+CC", solve_ol_se_cc),
]

_SEED = 26

_DENSITY_REF_N_SS   = np.array([50,   75,   100,  150],  dtype=float)
_DENSITY_REF_CENTER = np.array([3.00, 7.75, 7.63, 6.83], dtype=float)
_DENSITY_REF_SUBURBS= np.array([0.74, 3.92, 1.59, 1.50], dtype=float)

"""
Get centre and suburbs densities for a given number of SSs
"""
def _get_densities(n_ss: int):

    n = float(n_ss)
    dc = float(np.interp(n, _DENSITY_REF_N_SS, _DENSITY_REF_CENTER))
    ds = float(np.interp(n, _DENSITY_REF_N_SS, _DENSITY_REF_SUBURBS))
    if n_ss < 50:
        scale = n / 50.0
        dc = max(0.5, _DENSITY_REF_CENTER[0] * scale)
        ds = max(0.1, _DENSITY_REF_SUBURBS[0] * scale)

    if ds >= dc:
        ds = dc * 0.5
    return dc, ds

"""Build the growth schedule: list of (n_ps, n_ss) tuples."""
def _make_size_schedule(max_steps: int = 200):
    schedule = []
    for step in range(max_steps):
        n_ss = 3 + step
        if n_ss < 15:
            n_ps = 1
        elif n_ss < 35:
            n_ps = 2
        elif n_ss < 50:
            n_ps = 3
        else:
            n_ps = 4
        schedule.append((n_ps, n_ss))
    return schedule

"""Split total PSs between centre and suburbs (centre gets the extra)."""
def _split_ps(n_ps):
    n_ps_suburbs = n_ps // 2
    n_ps_center = n_ps - n_ps_suburbs
    return n_ps_center, n_ps_suburbs

"""Split total SSs between centre and suburbs (centre gets the extra)."""
def _split_ss(n_ss):
    n_ss_suburbs = max(1, n_ss // 2)
    n_ss_center = max(1, n_ss - n_ss_suburbs)
    return n_ss_center, n_ss_suburbs

"""Generate a synthetic network of the requested size."""
def _build_network(n_ps: int, n_ss: int, seed: int = _SEED):
    n_ps_c, n_ps_s = _split_ps(n_ps)
    n_ss_c, n_ss_s = _split_ss(n_ss)
    dc, ds = _get_densities(n_ss)
    gen = Generator(demand_std_suburbs=0.2)
    return gen.generate(
        n_ps_center=n_ps_c,
        n_ps_suburbs=n_ps_s,
        n_ss_center=n_ss_c,
        n_ss_suburbs=n_ss_s,
        density_center=dc,
        density_suburbs=ds,
        seed=seed,
    )

def test_scalability(time: float):
    max_time = float(time)
    schedule = _make_size_schedule()

    # Per-algorithm results: list of dicts {step, n_ps, n_ss, n_nodes, t}
    results = {name: [] for name, _ in _ALGORITHMS}
    active = {name: True for name, _ in _ALGORITHMS}
    # Count consecutive runs exceeding max_time per algorithm
    consecutive_over = {name: 0 for name, _ in _ALGORITHMS}

    print()
    print("=" * 78)
    print(f"  SCALABILITY TEST  -  per-run cap: {max_time:.1f} s")
    print("=" * 78)
    print(f"{'step':>4} {'PS':>3} {'SS':>4} {'N':>4}  " +
          "  ".join(f"{name:>10}" for name, _ in _ALGORITHMS))
    print("-" * 78)

    for step, (n_ps, n_ss) in enumerate(schedule):
        if not any(active.values()):
            break
        try:
            net = _build_network(n_ps, n_ss, seed=_SEED + step)
        except Exception as exc:
            print(f"[step {step}] network generation failed ({exc}); stopping.")
            break

        n_nodes = net.n_nodes
        row_times = {}

        for name, solver in _ALGORITHMS:
            if not active[name]:
                row_times[name] = None
                continue
            print(f"  >> step {step:>3} | {name:<10} | N={n_nodes:>4} | running...    ",
                  end="\r", flush=True)
            t0 = _time.time()
            try:
                solver(net, time_limit=max_time, verbose=False)
            except Exception as exc:
                print(f"[step {step}] {name} raised {exc!r} - dropping.")
                active[name] = False
                row_times[name] = None
                continue
            elapsed = _time.time() - t0

            if elapsed > max_time:
                extra_times = [elapsed]
                for k in (1, 2):
                    try:
                        net_k = _build_network(n_ps, n_ss,
                                               seed=_SEED + step + 10_000 * k)
                    except Exception as exc:
                        print(f"[step {step}] retry generation failed ({exc}).")
                        break
                    print(f"  >> step {step:>3} | {name:<10} | N={n_nodes:>4} | retry {k}/2...  ",
                          end="\r", flush=True)
                    t0k = _time.time()
                    try:
                        solver(net_k, time_limit=max_time, verbose=False)
                    except Exception as exc:
                        print(f"[step {step}] {name} retry raised {exc!r}.")
                        break
                    extra_times.append(_time.time() - t0k)
                extra_times.sort()
                elapsed = extra_times[len(extra_times) // 2]  # median

            row_times[name] = elapsed
            results[name].append({
                "step": step,
                "n_ps": n_ps,
                "n_ss": n_ss,
                "n_nodes": n_nodes,
                "t": elapsed,
            })
            if elapsed > max_time:
                consecutive_over[name] += 1
                if consecutive_over[name] >= 2:
                    active[name] = False  # 2 consecutive overruns -> drop
            else:
                consecutive_over[name] = 0  # reset streak on a fast run

        cells = []
        for name, _ in _ALGORITHMS:
            t = row_times.get(name)
            cells.append("    -     " if t is None else f"{t:>9.3f}s")
        print(f"{step:>4} {n_ps:>3} {n_ss:>4} {n_nodes:>4}  " + "  ".join(cells))
    print()
    print("=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)
    for name, _ in _ALGORITHMS:
        runs = results[name]
        if not runs:
            print(f"  {name:<10}: no successful runs.")
            continue
        last = runs[-1]
        print(f"  {name:<10}: {len(runs):>3} runs | "
              f"max nodes reached = {last['n_nodes']:>4} "
              f"(PS={last['n_ps']}, SS={last['n_ss']}) | "
              f"last time = {last['t']:.3f}s")
    print("=" * 78)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    json_path = os.path.join(out_dir, f"scalability_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            "max_time": max_time,
            "results": results,
        }, f, indent=2)
    print(f"Data saved to: {json_path}")
    return results

if __name__ == "__main__":
    try:
        secs = float(input("Per-run time cap (seconds) [30]: ").strip() or "30")
    except ValueError:
        secs = 30.0
    results = test_scalability(secs)
    plot_scalability_results(results, secs, _ALGORITHMS)


