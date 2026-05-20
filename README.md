# Power Distribution Networks

> **Optimal planning of power distribution networks with fault-tolerant configuration**

A Python framework for modelling, generating, loading and solving **Mixed-Integer Programming (MIP)** problems related to the optimal design of medium-voltage power distribution networks. The network topology must satisfy fault-tolerance requirements (loop-feeder or open-loop structure) and cable capacity constraints, while minimising total installation cost.

This project is based on the research work:

> Renato B., Alberto G., Marco M., Ludovico N.
> **Optimal planning of power distribution networks with fault-tolerant configuration**
> *Computers & Operations Research*, Volume 185, 2026, 107248, ISSN 0305-0548
> [https://doi.org/10.1016/j.cor.2025.107248](https://doi.org/10.1016/j.cor.2025.107248)

---

## Problem Description

The project addresses the **optimal design of electrical power distribution networks**, where:

- **Primary Substations (PS)** act as supply roots (sources of power).
- **Secondary Substations (SS)** are demand nodes that must be served.
- Each SS must be connected to exactly **two cables** (degree-2 constraint), ensuring fault tolerance: if one cable fails, power can still be delivered via the other.
- Cables have a **rated capacity** (MW) that must not be exceeded.
- The objective is to **minimise the total cost** of installed cables.

Two topological configurations are supported:
- **Loop-Feeder (LF)**: Each SS forms a loop that must be rooted at at least one PS.
- **Open-Loop (OL)**: Loops are allowed between PSs, modelling open-loop normally-open switches.

---

## Project Structure

```
power-distribution-networks/
│
├── network_data.py       # NetworkData dataclass — holds all network data
├── generator.py          # Synthetic network generator
├── plot.py               # Visualization utilities (matplotlib)
├── scalability.py        # Scalability benchmark runner
├── test.py               # Interactive test runner (existing or generated data)
│
├── solver/
│   ├── utils.py          # Shared helpers (unpack, adjacency, extract_solution)
│   ├── LF_SCF.py         # Loop-Feeder Single Commodity Flow solver
│   ├── LF_SE.py          # Loop-Feeder Subtour Elimination solver
│   └── OL_SE.py          # Open-Loop Subtour Elimination solver (+ cut constraints variant)
│
└──dataset/
    ├── dataset.py        # CSV loader for real-world case studies
    ├── Case54/           # 54-node case (4 PS, 50 SS)
    ├── Case78/           # 78-node case (3 PS, 75 SS)
    ├── Case104/          # 104-node case (4 PS, 100 SS)
    ├── Case154/          # 154-node case (4 PS, 150 SS)
    └── Case205/          # 205-node case (4 PS, 200 SS)
```

---

## Network Model

All network data is stored in a `NetworkData` dataclass (`network_data.py`) with the following fields:

| Field | Type | Description |
|---|---|---|
| `nodes` | `np.ndarray (N, 2)` | Node coordinates in km |
| `types` | `List[str]` | Node type: `PS_center`, `PS_suburbs`, `SS_center`, `SS_suburbs` |
| `demands` | `np.ndarray (N,)` | Load demand in MW (0 for PSs) |
| `costs` | `Dict[(i,j), float]` | Arc installation cost in EUR |
| `capacities` | `Dict[(i,j), float]` | Arc rated capacity in MW |
| `ps_indices` | `List[int]` | Indices of Primary Substations |
| `ss_indices` | `List[int]` | Indices of Secondary Substations |
| `obstacle_polyline` | `List[(x,y)]` | Snake-like obstacle crossing the map |
| `side_center` | `float` | City-centre bounding box side (km) |
| `side_suburbs` | `float` | Suburban bounding box side (km) |

The network is partitioned into two spatial zones:
- **City centre** — higher node density, higher average load demand.
- **Suburbs** — lower density, forming a ring around the centre.

---

## Solvers

All solvers are implemented using **[Gurobi](https://www.gurobi.com/)** (`gurobipy`) and are located in the `solver/` package. Each solver returns a tuple `(active_arcs, obj, gap_pct, root_gap_pct)`.

---

## Dataset

Real-world case studies are stored under `dataset/` as CSV files and can be loaded via `dataset.load_network(name)`.

Each case is available in the **base configuration** and three **load-reduced** variants (`LESS_1`, `LESS_3`, `LESS_10`):

| Case | Nodes | PS | SS |
|---|---|---|---|
| Case54 | 54 | 4 | 50 |
| Case78 | 78 | 3 | 75 |
| Case104 | 104 | 4 | 100 |
| Case154 | 154 | 4 | 150 |
| Case205 | 205 | 5 | 200 |

Each folder contains:
- `<Case>_Nodes.CSV` — node coordinates, typology and load demand.
- `<Case>_Branches.CSV` — arc endpoints, capacity and cost.

---

## Synthetic Network Generator

The `Generator` class (`generator.py`) creates synthetic networks with configurable spatial and electrical parameters.

The generator:
1. Places nodes respecting the given spatial densities (suburban nodes are placed in the ring outside the centre box via rejection sampling).
2. Generates Gaussian load demands (absolute value to ensure positivity).
3. Creates a snake-like **obstacle polyline** crossing the map; arcs that cross it receive an extra penalty cost.
4. Computes arc costs as `base_cost_per_km × (1 + U[0, 0.1]) × euclidean_distance`.
5. Replaces direct costs with **all-pairs shortest-path costs** (Dijkstra via NetworkX).
6. Intentionally violates the triangle inequality for a random fraction of arcs.

---

## Scalability Testing

`scalability.py` benchmarks all four solvers across a growing sequence of synthetic networks (from ~3 SSs up to 200 SSs). Each solver is dropped after **two consecutive runs** exceeding the time cap.

Results are saved as a JSON file and a PNG chart in `plots/`.

---

## Usage

```bash
python test.py
```

The script will prompt you to:
1. Choose between **existing** (CSV dataset) or **generated** (synthetic) data.
2. Select a case study: `Case54`, `Case78`, `Case104`, `Case154`, or `all`.

All four solvers are then run sequentially and their solutions are plotted.