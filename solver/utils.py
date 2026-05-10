from typing import List, Tuple, Dict
from collections import defaultdict

"""Extract sets and parameters from a NetworkData object."""
def _unpack(net):
    R = list(net.ps_indices)
    D = list(net.ss_indices)
    A = list(net.costs.keys())
    c = net.costs
    d = {i: float(net.demands[i]) for i in D}
    p = {arc: float(net.cable_capacity) for arc in A}
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

"""Return (active_arcs, objective_value) from a solved model, or (None, None)."""
def _extract_solution(model, x, A):
    if model.SolCount == 0:
        return None, None
    active_arcs = [arc for arc in A if x[arc].X > 0.5]
    return active_arcs, model.ObjVal
