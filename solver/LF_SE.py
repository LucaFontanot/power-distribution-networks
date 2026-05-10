import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from .utils import (
    _unpack,
    _extract_solution,
    star,
    forward_star,
    backward_star,
)

"""
Loop-Feeder Subtour Elimination (LF-SE).
"""
def solve_lf_se(net, time_limit: float = 21600, verbose: bool = True):

    R, D, A, c, d, p = _unpack(net)
    R_set = set(R)

    m = gp.Model("LF-SE")
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = int(verbose)
    m.Params.LazyConstraints = 1

    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    f = m.addVars(A, vtype=GRB.CONTINUOUS, name="f", lb=-GRB.INFINITY)

    m.setObjective(gp.quicksum(c[arc] * x[arc] for arc in A), GRB.MINIMIZE)

    # (8) degree = 2
    for k in D:
        m.addConstr(
            gp.quicksum(x[arc] for arc in star(k, A)) == 2,
            name=f"deg_{k}",
        )

    # (10) flow conservation
    for k in D:
        fwd = forward_star(k, A)
        bwd = backward_star(k, A)
        m.addConstr(
            gp.quicksum(f[arc] for arc in bwd)
            - gp.quicksum(f[arc] for arc in fwd)
            == d[k],
            name=f"flow_{k}",
        )

    # (11) half-capacity rule
    for arc in A:
        i, j = arc
        m.addConstr(f[arc] >= -(x[arc] * p[arc]) / 2.0, name=f"cap_lo_{i}_{j}")
        m.addConstr(f[arc] <= (x[arc] * p[arc]) / 2.0, name=f"cap_hi_{i}_{j}")

    n_nodes = net.n_nodes

    def subtour_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        x_val = model.cbGetSolution(x)
        H = nx.Graph()
        H.add_nodes_from(range(n_nodes))
        for arc in A:
            if x_val[arc] > 0.5:
                H.add_edge(*arc)
        for comp in nx.connected_components(H):
            if not (comp & R_set):
                S_arcs = [(i, j) for (i, j) in A if i in comp and j in comp]
                if S_arcs:
                    model.cbLazy(
                        gp.quicksum(x[arc] for arc in S_arcs)
                        <= len(comp) - 1
                    )

    m.optimize(subtour_callback)
    return _extract_solution(m, x, A)
