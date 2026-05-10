import time

from dataset.dataset import load_network
from generator import Generator
from plot import plot_network, plot_solution
from solver.LF_SE import solve_lf_se
from solver.LF_SCF import solve_lf_scf
from solver.OL_SE import solve_ol_se, solve_ol_se_cc


CASE_STUDIES = {
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
SEED = 57
TIME_LIMIT = 60

def run_case(data):
    plot_network(data, show_arc_costs=False)
    plot_network(data, show_arc_costs=True)
    time_ms = time.time()
    arcs, obj = solve_lf_scf(data, time_limit=TIME_LIMIT, verbose=False)
    print(f"LF-SCF: time={time.time() - time_ms:.2f} seconds")
    plot_solution(arcs, obj, data, title="LF-SCF Solution")
    time_ms = time.time()
    arcs, obj = solve_lf_se(data, time_limit=TIME_LIMIT, verbose=False)
    print(f"LF-SE: time={time.time() - time_ms:.2f} seconds")
    plot_solution(arcs, obj, data, title="LF-SE Solution")
    time_ms = time.time()
    arcs, obj = solve_ol_se(data, time_limit=TIME_LIMIT, verbose=False)
    print(f"OL-SCF: time={time.time() - time_ms:.2f} seconds")
    plot_solution(arcs, obj, data, title="OL-SCF Solution")
    time_ms = time.time()
    arcs, obj = solve_ol_se_cc(data, time_limit=TIME_LIMIT, verbose=False)
    print(f"OL-SCF: time={time.time() - time_ms:.2f} seconds")
    plot_solution(arcs, obj, data, title="OL-SCF Solution")

def generate_data(case):
    gen = Generator()
    return gen.generate(
        case["n_ps_center"],
        case["n_ps_suburbs"],
        case["n_ss_center"],
        case["n_ss_suburbs"],
        case["density_center"],
        case["density_suburbs"],
        SEED
    )

if __name__ == "__main__":
    print("Welcome to the Power Distribution Networks solver.")
    print("Select dataset options:")
    print("  existing | Provides the dataset from given CSVs containing nodes and arcs information.")
    print("  generate | Generates a new dataset based on specified parameters.")
    choice = input("Enter your choice (existing/generate): ").strip().lower()
    print("Available case studies:")
    for case_name in CASE_STUDIES.keys():
        print(f"  {case_name}")
    case_choice = input("Enter the case study you want to run: ").strip()
    if case_choice not in CASE_STUDIES:
        print(f"Invalid case study choice: {case_choice}")
        exit(1)
    if choice == "existing":
        data = load_network(case_choice)
        run_case(data)
    elif choice == "generate":
        case_params = CASE_STUDIES[case_choice]
        data = generate_data(case_params)
        run_case(data)
    else:
        print(f"Invalid choice: {choice}")
        exit(1)
