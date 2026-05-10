import gurobipy as gp
from gurobipy import GRB

from .utils import (
    _unpack,
    _extract_solution,
    star,
    forward_star,
    backward_star,
)

"""
Loop-Feeder Single Commodity Flow (LF-SCF).
"""
def solve_lf_scf(net, time_limit: float = 21600, verbose: bool = True):

    R, D, A, c, d, p = _unpack(net)
    nD = len(D)

    m = gp.Model("LF-SCF")
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = int(verbose)

    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    f = m.addVars(A, vtype=GRB.CONTINUOUS, name="f", lb=-GRB.INFINITY)
    s = m.addVars(A, vtype=GRB.CONTINUOUS, name="s", lb=-GRB.INFINITY)

    m.setObjective(gp.quicksum(c[arc] * x[arc] for arc in A), GRB.MINIMIZE)

    # (2) degree = 2 for every demand node
    for k in D:
        m.addConstr(
            gp.quicksum(x[arc] for arc in star(k, A)) == 2,
            name=f"deg_{k}",
        )

    for k in D:
        fwd = forward_star(k, A)
        bwd = backward_star(k, A)

        # (3) real power flow conservation
        m.addConstr(
            gp.quicksum(f[arc] for arc in bwd)
            - gp.quicksum(f[arc] for arc in fwd)
            == d[k],
            name=f"flow_f_{k}",
        )

        # (4) fictitious flow conservation
        m.addConstr(
            gp.quicksum(s[arc] for arc in bwd)
            - gp.quicksum(s[arc] for arc in fwd)
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

    m.optimize()
    return _extract_solution(m, x, A)
