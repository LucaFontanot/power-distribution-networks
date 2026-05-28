import time

from dataset.dataset import load_network
from generator import Generator
from plot import plot_network, plot_solution
from solver.LF_SE import solve_lf_se
from solver.LF_SCF import solve_lf_scf
from solver.OL_SE import solve_ol_se, solve_ol_se_cc


_CASE_STUDIES = {
    "Case54": dict(
        n_ps_center=2, n_ps_suburbs=2,
        n_ss_center=25, n_ss_suburbs=25,
        density_center=3.00, density_suburbs=0.74,
    ),
    "Case78": dict(
        n_ps_center=1, n_ps_suburbs=2,
        n_ss_center=30, n_ss_suburbs=45,
        density_center=7.75, density_suburbs=3.92,
    ),
    "Case104": dict(
        n_ps_center=1, n_ps_suburbs=3,
        n_ss_center=60, n_ss_suburbs=40,
        density_center=7.63, density_suburbs=1.59,
    ),
    "Case154": dict(
        n_ps_center=2, n_ps_suburbs=2,
        n_ss_center=80, n_ss_suburbs=70,
        density_center=6.83, density_suburbs=1.50,
    ),
}
_SEED = 57
_TIME_LIMIT = 60 * 60

def _fmt_gap(label, val):
    return f"{val:.2f}%" if val is not None else "N/A"

def _run_case(data, is_generated=False):
    print()
    print("=" * 50)
    print(f"  Case study : {case_choice}")
    print("=" * 50)
    print(f"  Nodes      : {data.n_nodes}  (PS: {data.n_ps}, SS: {len(data.ss_indices)})")
    print(f"  Arcs       : {data.n_arcs}")
    print(f"  PS roots   : {data.ps_indices}")
    print(f"  Total demand : {data.demands.sum():.4f} MW")
    print("=" * 50)
    print()
    plot_network(data, show_arc_costs=False, show_area=is_generated)
    plot_network(data, show_arc_costs=True, show_area=is_generated)
    time_ms = time.time()
    arcs, obj, gap, root_gap = solve_lf_scf(data, time_limit=_TIME_LIMIT, verbose=True)
    print(f"LF-SCF:  time={time.time() - time_ms:.2f}s | gap={_fmt_gap('', gap)} | root gap={_fmt_gap('', root_gap)}")
    plot_solution(arcs, obj, data, title="LF-SCF Solution", show_area=is_generated)
    time_ms = time.time()
    arcs, obj, gap, root_gap = solve_lf_se(data, time_limit=_TIME_LIMIT, verbose=True)
    print(f"LF-SE:   time={time.time() - time_ms:.2f}s | gap={_fmt_gap('', gap)} | root gap={_fmt_gap('', root_gap)}")
    plot_solution(arcs, obj, data, title="LF-SE Solution", show_area=is_generated)
    time_ms = time.time()
    arcs, obj, gap, root_gap = solve_ol_se(data, time_limit=_TIME_LIMIT, verbose=True)
    print(f"OL-SE:   time={time.time() - time_ms:.2f}s | gap={_fmt_gap('', gap)} | root gap={_fmt_gap('', root_gap)}")
    plot_solution(arcs, obj, data, title="OL-SE Solution", show_area=is_generated)
    time_ms = time.time()
    arcs, obj, gap, root_gap = solve_ol_se_cc(data, time_limit=_TIME_LIMIT, verbose=True)
    print(f"OL-SE+CC: time={time.time() - time_ms:.2f}s | gap={_fmt_gap('', gap)} | root gap={_fmt_gap('', root_gap)}")
    plot_solution(arcs, obj, data, title="OL-SE Solution", show_area=is_generated)

def _generate_data(case):
    gen = Generator()
    return gen.generate(
        case["n_ps_center"],
        case["n_ps_suburbs"],
        case["n_ss_center"],
        case["n_ss_suburbs"],
        case["density_center"],
        case["density_suburbs"],
        _SEED
    )

if __name__ == "__main__":
    print("Welcome to the Power Distribution Networks solver.")
    print("Select dataset options:")
    print("  existing | Provides the dataset from given CSVs containing nodes and arcs information.")
    print("  generate | Generates a new dataset based on specified parameters.")
    choice = input("Enter your choice (existing/generate): ").strip().lower()
    print("Available case studies:")
    for case_name in _CASE_STUDIES.keys():
        print(f"  {case_name}")
    print("  all")
    case_choice = input("Enter the case study you want to run: ").strip()
    if case_choice not in _CASE_STUDIES and case_choice != "all":
        print(f"Invalid case study choice: {case_choice}")
        exit(1)
    selected_cases = _CASE_STUDIES.keys() if case_choice == "all" else [case_choice]
    for case_choice in selected_cases:
        if choice == "existing":
            data = load_network(case_choice)
            data.dump(f"{case_choice}_dump.txt")
        elif choice == "generate":
            case_params = _CASE_STUDIES[case_choice]
            data = _generate_data(case_params)
        else:
            print(f"Invalid choice: {choice}")
            exit(1)
        _run_case(data, is_generated=(choice == "generate"))
