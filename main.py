import time

from generator import Generator
from plot import plot_network
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

def run_case(generator, case):
    data = generator.generate(
        case["n_ps_center"],
        case["n_ps_suburbs"],
        case["n_ss_center"],
        case["n_ss_suburbs"],
        case["density_center"],
        case["density_suburbs"],
        SEED
    )
    plot_network(data, show_arc_costs=False)
    plot_network(data, show_arc_costs=True)
    time_ms = time.time()
    arcs, obj = solve_lf_scf(data, time_limit=21600, verbose=False)
    print(f"LF-SCF: time={time.time() - time_ms:.2f} seconds")
    time_ms = time.time()
    arcs, obj = solve_lf_se(data, time_limit=21600, verbose=False)
    print(f"LF-SE: time={time.time() - time_ms:.2f} seconds")
    time_ms = time.time()
    arcs, obj = solve_ol_se(data, time_limit=21600, verbose=False)
    print(f"OL-SCF: time={time.time() - time_ms:.2f} seconds")
    time_ms = time.time()
    arcs, obj = solve_ol_se_cc(data, time_limit=21600, verbose=False)
    print(f"OL-SCF: time={time.time() - time_ms:.2f} seconds")




if __name__ == "__main__":
    gen = Generator()
    run_case(gen, CASE_STUDIES["Case54"])
    #run_case(gen, CASE_STUDIES["Case78"])
    #run_case(gen, CASE_STUDIES["Case104"])
    #run_case(gen, CASE_STUDIES["Case154"])
