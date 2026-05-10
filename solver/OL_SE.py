import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from .utils import _unpack, _extract_solution, build_adjacency

"""
Open-Loop Subtour Elimination (OL-SE).

If `cut_constraints=True`, additionally adds the explicit cut-based
constraints upfront
"""
def solve_ol_se(net, time_limit: float = 21600, verbose: bool = True,
                cut_constraints: bool = False):
    R, D, A, c, d, p = _unpack(net)
    R_set = set(R)
    D_set = set(D)
    n_nodes = net.n_nodes
    n_star = n_nodes  # index of virtual node

    fwd, bwd, inc = build_adjacency(A)

    m = gp.Model("OL-SE+CC" if cut_constraints else "OL-SE")
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = int(verbose)
    m.Params.LazyConstraints = 1

    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    f = m.addVars(A, vtype=GRB.CONTINUOUS, name="f", lb=-GRB.INFINITY)

    m.setObjective(gp.quicksum(c[arc] * x[arc] for arc in A), GRB.MINIMIZE)

    for k in D:
        # (13) degree = 2
        m.addConstr(
            gp.quicksum(x[arc] for arc in inc[k]) == 2,
            name=f"deg_{k}",
        )
        # (15) flow conservation
        m.addConstr(
            gp.quicksum(f[arc] for arc in bwd[k])
            - gp.quicksum(f[arc] for arc in fwd[k])
            == d[k],
            name=f"flow_{k}",
        )

    # (16) half-capacity rule
    for arc in A:
        i, j = arc
        m.addConstr(f[arc] >= -(x[arc] * p[arc]) / 2.0, name=f"cap_lo_{i}_{j}")
        m.addConstr(f[arc] <= (x[arc] * p[arc]) / 2.0, name=f"cap_hi_{i}_{j}")

    # Optional cut-based constraints (S' = D, one per root r in R):
    # Σ x[D, other_roots]  >=  Σ x[D, r]
    if cut_constraints:
        for r in R:
            other_roots = R_set - {r}
            arcs_to_r = [
                (i, j) for (i, j) in A
                if (i in D_set and j == r) or (j in D_set and i == r)
            ]
            arcs_to_others = [
                (i, j) for (i, j) in A
                if (i in D_set and j in other_roots)
                or (j in D_set and i in other_roots)
            ]
            if arcs_to_r or arcs_to_others:
                m.addConstr(
                    gp.quicksum(x[arc] for arc in arcs_to_others)
                    >= gp.quicksum(x[arc] for arc in arcs_to_r),
                    name=f"cc_r{r}",
                )

    def open_loop_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        x_val = model.cbGetSolution(x)
        H = nx.Graph()
        H.add_nodes_from(range(n_nodes + 1))
        for arc in A:
            if x_val[arc] > 0.5:
                H.add_edge(*arc)
        for r in R:
            H.add_edge(n_star, r)
        for bcc in nx.biconnected_components(H):
            if n_star in bcc:
                continue
            if len(bcc & R_set) <= 1:
                S_arcs = [(i, j) for (i, j) in A if i in bcc and j in bcc]
                if S_arcs:
                    model.cbLazy(
                        gp.quicksum(x[arc] for arc in S_arcs)
                        <= len(bcc) - 1
                    )

    m.optimize(open_loop_callback)
    return _extract_solution(m, x, A)


def solve_ol_se_cc(net, time_limit: float = 21600, verbose: bool = True):
    """OL-SE + Cut Constraints — convenience wrapper around `solve_ol_se`."""
    return solve_ol_se(net, time_limit=time_limit, verbose=verbose,
                       cut_constraints=True)
