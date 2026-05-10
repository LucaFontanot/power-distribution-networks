from typing import List, Tuple

"""Extract sets and parameters from a NetworkData object."""
def _unpack(net):
    R = list(net.ps_indices)
    D = list(net.ss_indices)
    A = list(net.costs.keys())
    c = net.costs
    d = {i: float(net.demands[i]) for i in D}
    max_d = max(d.values()) if d else 1.0
    p = {arc: 2.0 * max_d for arc in A}
    return R, D, A, c, d, p

"""All arcs incident to node k."""
def star(k: int, A: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in A if i == k or j == k]

"""Arcs (k, j) leaving k (k is the smaller index)."""
def forward_star(k: int, A: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in A if i == k]

"""Arcs (i, k) entering k (k is the larger index)."""
def backward_star(k: int, A: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(i, j) for (i, j) in A if j == k]

"""Return (active_arcs, objective_value) from a solved model, or (None, None)."""
def _extract_solution(model, x, A):
    if model.SolCount == 0:
        return None, None
    active_arcs = [arc for arc in A if x[arc].X > 0.5]
    return active_arcs, model.ObjVal
