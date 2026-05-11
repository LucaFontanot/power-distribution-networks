from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class NetworkData:
    """Holds all data for a generated distribution network."""

    # Node information
    nodes: np.ndarray  # shape (N, 2) — x,y coordinates in km
    types: List[str]  # 'PS', 'SS_center', or 'SS_suburbs'
    demands: np.ndarray  # shape (N,) — MW demand per node (0 for PSs)

    # Arc information
    costs: Dict[Tuple[int, int], float]       # arc (i,j) with i<j → cost in EUR
    capacities: Dict[Tuple[int, int], float]  # arc (i,j) with i<j → capacity in MW

    # Index helpers
    ps_indices: List[int]
    ss_indices: List[int]

    # Obstacle polyline: sequence of (x, y) waypoints defining the snake
    obstacle_polyline: List[Tuple[float, float]]

    # Zone geometry (km)
    side_center: float
    side_suburbs: float

    # Generation metadata
    n_ps_center: int
    n_ps_suburbs: int
    n_ss_center: int
    n_ss_suburbs: int
    density_center: float
    density_suburbs: float
    seed: int

    # Derived counts
    @property
    def n_ps(self) -> int:
        return self.n_ps_center + self.n_ps_suburbs

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_arcs(self) -> int:
        return len(self.costs)

    def dump(self, path: "str | Path" = "network_dump.txt") -> Path:
        out = Path(path)
        lines: List[str] = []

        lines.append("=" * 70)
        lines.append("NETWORK DATA DUMP")
        lines.append("=" * 70)
        lines.append(self.summary())
        lines.append("")

        lines.append("-" * 70)
        lines.append(f"NODES  ({self.n_nodes} total)")
        lines.append("-" * 70)
        lines.append(f"  {'Idx':>4}  {'Type':<12}  {'x (km)':>10}  {'y (km)':>10}  {'demand (MW)':>12}")
        for i, (xy, t, dem) in enumerate(zip(self.nodes, self.types, self.demands)):
            lines.append(f"  {i:>4}  {t:<12}  {xy[0]:>10.4f}  {xy[1]:>10.4f}  {dem:>12.4f}")
        lines.append("")

        lines.append("-" * 70)
        lines.append("INDEX SETS")
        lines.append("-" * 70)
        lines.append(f"  PS (roots) [{len(self.ps_indices)}]: {self.ps_indices}")
        lines.append(f"  SS (demand) [{len(self.ss_indices)}]: {self.ss_indices}")
        lines.append("")

        lines.append("-" * 70)
        lines.append(f"ARCS  ({self.n_arcs} total)")
        lines.append("-" * 70)
        lines.append(f"  {'(i,j)':<12}  {'cost (EUR)':>14}  {'capacity (MW)':>14}")
        for arc, cost in sorted(self.costs.items()):
            cap = self.capacities[arc]
            lines.append(f"  {str(arc):<12}  {cost:>14.2f}  {cap:>14.4f}")
        lines.append("")

        if self.obstacle_polyline:
            lines.append("-" * 70)
            lines.append(f"OBSTACLE POLYLINE  ({len(self.obstacle_polyline)} waypoints)")
            lines.append("-" * 70)
            for k, (x, y) in enumerate(self.obstacle_polyline):
                lines.append(f"  [{k:>3}]  x={x:>10.4f}  y={y:>10.4f}")
            lines.append("")

        lines.append("-" * 70)
        lines.append("ZONE GEOMETRY")
        lines.append("-" * 70)
        lines.append(f"  Centre box   : {self.side_center:.4f} x {self.side_center:.4f} km")
        lines.append(f"  Suburbs box  : {self.side_suburbs:.4f} x {self.side_suburbs:.4f} km")
        lines.append(f"  Density centre   : {self.density_center} nodes/km²")
        lines.append(f"  Density suburbs  : {self.density_suburbs} nodes/km²")
        lines.append(f"  Seed         : {self.seed}")
        lines.append("=" * 70)

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def summary(self) -> str:
        return (
            f"Network summary\n"
            f"  PSs centre   : {self.n_ps_center}\n"
            f"  PSs suburbs  : {self.n_ps_suburbs}\n"
            f"  SSs centre   : {self.n_ss_center}\n"
            f"  SSs suburbs  : {self.n_ss_suburbs}\n"
            f"  Total nodes  : {self.n_nodes}\n"
            f"  Total arcs   : {self.n_arcs}\n"
            f"  Centre box   : {self.side_center:.2f} x {self.side_center:.2f} km  "
            f"({self.density_center} nodes/km²)\n"
            f"  Suburbs box  : {self.side_suburbs:.2f} x {self.side_suburbs:.2f} km  "
            f"({self.density_suburbs} nodes/km²)\n"
            f"  Obstacle     : {len(self.obstacle_polyline) - 1} segments, "
            f"{len(self.obstacle_polyline)} waypoints\n"
            f"  Seed         : {self.seed}"
        )


