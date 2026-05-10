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
    costs: Dict[Tuple[int, int], float]  # arc (i,j) with i<j → cost in EUR
    cable_capacity: float                # standardised rated capacity per cable (MW)

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


