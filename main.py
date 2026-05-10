from generator import Generator

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
SEED = 40

def run_case(generator, case):
    for i in range(100):
        data = generator.generate(
            case["n_ps_center"],
            case["n_ps_suburbs"],
            case["n_ss_center"],
            case["n_ss_suburbs"],
            case["density_center"],
            case["density_suburbs"],
            i,
        )
        generator.plot(data)


if __name__ == "__main__":
    gen = Generator()
    run_case(gen, CASE_STUDIES["Case54"])
