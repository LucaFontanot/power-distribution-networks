import gurobipy as gp
from gurobipy import GRB

from .utils import _unpack, _extract_solution, build_adjacency

"""
Loop-Feeder Single Commodity Flow (LF-SCF).
"""
def solve_lf_scf(net, time_limit: float = 21600, verbose: bool = True):
    R, D, A, c, d, p = _unpack(net)
    nD = len(D)

    fwd, bwd, inc = build_adjacency(A)

    m = gp.Model("LF-SCF")
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = int(verbose)

    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    f = m.addVars(A, vtype=GRB.CONTINUOUS, name="f", lb=-GRB.INFINITY)
    s = m.addVars(A, vtype=GRB.CONTINUOUS, name="s", lb=-GRB.INFINITY)

    m.setObjective(gp.quicksum(c[arc] * x[arc] for arc in A), GRB.MINIMIZE)

    for k in D:
        # (2) degree = 2
        m.addConstr(
            gp.quicksum(x[arc] for arc in inc[k]) == 2,
            name=f"deg_{k}",
        )
        # (3) real power flow conservation
        m.addConstr(
            gp.quicksum(f[arc] for arc in bwd[k])
            - gp.quicksum(f[arc] for arc in fwd[k])
            == d[k],
            name=f"flow_f_{k}",
        )
        # (4) fictitious flow conservation
        m.addConstr(
            gp.quicksum(s[arc] for arc in bwd[k])
            - gp.quicksum(s[arc] for arc in fwd[k])
            == 1,
            name=f"flow_s_{k}",
        )

    for arc in A:
        i, j = arc
        # (5) half-capacity rule for real flow
        m.addConstr(f[arc] >= -(x[arc] * p[arc]) / 2.0, name=f"cap_f_lo_{i}_{j}")
        m.addConstr(f[arc] <= (x[arc] * p[arc]) / 2.0, name=f"cap_f_hi_{i}_{j}")

        # (6) half-capacity rule for fictitious flow
        m.addConstr(s[arc] >= -(x[arc] * nD) / 2.0, name=f"cap_s_lo_{i}_{j}")
        m.addConstr(s[arc] <= (x[arc] * nD) / 2.0, name=f"cap_s_hi_{i}_{j}")

    m.write("lf_scf.lp")
    m.optimize()
    return _extract_solution(m, x, A)
