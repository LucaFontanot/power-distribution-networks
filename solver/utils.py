from typing import List, Tuple, Dict
from collections import defaultdict

"""Extract sets and parameters from a NetworkData object."""
def unpack(net):
    R = list(net.ps_indices)
    D = list(net.ss_indices)

    # Guard: R and D must be disjoint — PS (root) nodes must never appear in D.
    # If ss_indices accidentally contains PS indices every root gets a degree=2
    # constraint and nD inflates, corrupting the cap_s bound (Bug 2).
    overlap = set(R) & set(D)
    if overlap:
        raise ValueError(
            f"ss_indices contains PS node indices {sorted(overlap)}. "
            f"R and D must be disjoint (|R|={len(R)}, |D|={len(D)})."
        )

    A = list(net.costs.keys())
    c = net.costs
    d = {i: float(net.demands[i]) for i in D}
    p = {arc: float(net.capacities[arc]) for arc in A}
    return R, D, A, c, d, p

"""Precompute forward/backward/star adjacency."""
def build_adjacency(A: List[Tuple[int, int]]):
    fwd = defaultdict(list)   # fwd[i]  = arcs (i,j) leaving i
    bwd = defaultdict(list)   # bwd[j]  = arcs (i,j) entering j
    inc = defaultdict(list)   # inc[k]  = all arcs incident to k
    for arc in A:
        i, j = arc
        fwd[i].append(arc)
        bwd[j].append(arc)
        inc[i].append(arc)
        inc[j].append(arc)
    return fwd, bwd, inc

"""Return (active_arcs, obj, gap_pct, root_gap_pct) from a solved model, or (None, None, None, None)."""
def extract_solution(model, x, A, root_bound=None):
    if model.SolCount == 0:
        return None, None, None, None
    active_arcs = [arc for arc in A if x[arc].X > 0.5]
    obj = model.ObjVal
    gap_pct = model.MIPGap * 100
    if root_bound is not None and abs(obj) > 1e-10:
        root_gap_pct = abs(root_bound - obj) / abs(obj) * 100
    else:
        root_gap_pct = None
    return active_arcs, obj, gap_pct, root_gap_pct
