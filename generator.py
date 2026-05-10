import numpy as np
from network_data import NetworkData
from typing import Dict, List, Tuple
import networkx as nx

class Generator:

    """
    base_cost_per_km : float
        Base arc cost coefficient (EUR/km). Default: 30,000.
    obstacle_cost : float
        Extra cost added when an arc crosses the obstacle polyline (EUR).
        Default: 1,000,000.
    triangle_vio_fraction : float
        Fraction of arcs for which the triangle inequality is intentionally
        violated. Default: 0.1.
    demand_mean_center : float
        Mean load demand for city-centre SSs (MW). Default: 0.4.
    demand_std_center : float
        Std of load demand for city-centre SSs (MW). Default: 0.2.
    demand_mean_suburbs : float
        Mean load demand for suburban SSs (MW). Default: 0.2.
    demand_std_suburbs : float
        Std of load demand for suburban SSs (MW). Default: 0.6.
    """
    def __init__(
            self,
            base_cost_per_km: float = 30_000,
            obstacle_cost: float = 1_000_000,
            triangle_vio_fraction: float = 0.1,
            demand_mean_center: float = 0.4,
            demand_std_center: float = 0.2,
            demand_mean_suburbs: float = 0.2,
            demand_std_suburbs: float = 0.6,
    ):
        self.base_cost_per_km = base_cost_per_km
        self.obstacle_cost = obstacle_cost
        self.triangle_vio_fraction = triangle_vio_fraction
        self.demand_mean_center = demand_mean_center
        self.demand_std_center = demand_std_center
        self.demand_mean_suburbs = demand_mean_suburbs
        self.demand_std_suburbs = demand_std_suburbs

    """
    Generate a single synthetic network.

    n_ps_center    : number of Primary Substations in city centre
    n_ps_suburbs   : number of Primary Substations in suburbs
    n_ss_center    : number of Secondary Substations in city centre
    n_ss_suburbs   : number of Secondary Substations in suburbs
    density_center : spatial density of centre nodes (nodes/km²)
    density_suburbs: spatial density of suburban nodes (nodes/km²)
    seed           : random seed for reproducibility
    """
    def generate(
            self,
            n_ps_center: int,
            n_ps_suburbs: int,
            n_ss_center: int,
            n_ss_suburbs: int,
            density_center: float,
            density_suburbs: float,
            seed: int = 42,
    ) -> NetworkData:
        rng = np.random.default_rng(seed)

        n_ps = n_ps_center + n_ps_suburbs

        # Place nodes
        nodes, types, side_c, side_s = self._place_nodes(
            n_ps_center, n_ps_suburbs,
            n_ss_center, n_ss_suburbs,
            density_center, density_suburbs, rng
        )

        # Generate demands
        demands = self._generate_demands(n_ps, n_ss_center, n_ss_suburbs, rng)

        # Generate random obstacle polyline (snake across the map)
        obstacle = self._generate_obstacle_polyline(side_s / 2, rng)

        # Compute raw arc costs
        raw_costs = self._compute_raw_costs(nodes, obstacle, rng)

        # Derive shortest-path costs
        sp_costs = self._shortest_path_costs(nodes, raw_costs)

        # Intentionally violate triangle inequality for some arcs
        sp_costs = self._violate_triangle(sp_costs, rng)

        return NetworkData(
            nodes=nodes,
            types=types,
            demands=demands,
            costs=sp_costs,
            ps_indices=list(range(n_ps)),
            ss_indices=list(range(n_ps, n_ps + n_ss_center + n_ss_suburbs)),
            obstacle_polyline=obstacle,
            side_center=side_c,
            side_suburbs=side_s,
            n_ps_center=n_ps_center,
            n_ps_suburbs=n_ps_suburbs,
            n_ss_center=n_ss_center,
            n_ss_suburbs=n_ss_suburbs,
            density_center=density_center,
            density_suburbs=density_suburbs,
            seed=seed,
        )

    """Compute bounding box side (km) from node count and density."""
    def _box_side(self, n_nodes: int, density: float) -> float:
        return np.sqrt(n_nodes / density)

    """
    Place all nodes respecting the target spatial densities.
    Node order: PS_center | PS_suburbs | SS_center | SS_suburbs
    """
    def _place_nodes(
            self,
            n_ps_center: int,
            n_ps_suburbs: int,
            n_ss_center: int,
            n_ss_suburbs: int,
            density_center: float,
            density_suburbs: float,
            rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[str], float, float]:

        side_c = self._box_side(n_ss_center, density_center)
        side_s = self._box_side(n_ss_suburbs, density_suburbs)
        half_c, half_s = side_c / 2, side_s / 2

        if half_s <= half_c:
            raise ValueError(
                f"Suburban box (side={side_s:.2f} km) must be larger than "
                f"centre box (side={side_c:.2f} km). "
                f"Ensure density_suburbs < density_center."
            )

        # Centre nodes — strictly inside the centre box
        ps_center = rng.uniform(-half_c, half_c, (n_ps_center, 2))
        ss_center = rng.uniform(-half_c, half_c, (n_ss_center, 2))

        # Suburban nodes — in the ring outside the centre box
        ps_suburbs = self._sample_in_ring(n_ps_suburbs, half_s, half_c, rng)
        ss_suburbs = self._sample_in_ring(n_ss_suburbs, half_s, half_c, rng)

        nodes = np.vstack([ps_center, ps_suburbs, ss_center, ss_suburbs])
        types = (
                ['PS_center'] * n_ps_center +
                ['PS_suburbs'] * n_ps_suburbs +
                ['SS_center'] * n_ss_center +
                ['SS_suburbs'] * n_ss_suburbs
        )

        return nodes, types, side_c, side_s

    """
    Sample n points uniformly inside the outer box but strictly outside
    the inner (centre) box — i.e. in the ring/frame area only.
    Uses rejection sampling; converges quickly when the ring is large.
    """
    def _sample_in_ring(
            self,
            n: int,
            half_outer: float,
            half_inner: float,
            rng: np.random.Generator,
    ) -> np.ndarray:
        points = []
        while len(points) < n:
            batch = rng.uniform(-half_outer, half_outer, (max(n * 4, 64), 2))
            outside = batch[
                (np.abs(batch[:, 0]) > half_inner) |
                (np.abs(batch[:, 1]) > half_inner)
                ]
            points.extend(outside.tolist())
        return np.array(points[:n])

    """
    Generate a random snake-like obstacle polyline that crosses the map
    from the left edge to the right edge.  The number of segments scales
    randomly with the map size (side = 2 * half_outer).
    """
    def _generate_obstacle_polyline(
            self,
            half_s: float,
            rng: np.random.Generator,
    ) -> List[Tuple[float, float]]:
        side = half_s * 2
        # At least 2 segments; upper bound grows with map size
        n_segs = int(rng.integers(2, max(4, int(side) + 1)))
        n_pts = n_segs + 1

        xs = np.linspace(-half_s, half_s, n_pts)
        ys = np.empty(n_pts)
        ys[0] = float(rng.uniform(-half_s, half_s))

        # Random walk: each step can deviate up to ~40 % of the map height
        max_step = side * 0.4
        for k in range(1, n_pts):
            ys[k] = float(np.clip(
                ys[k - 1] + rng.uniform(-max_step, max_step),
                -half_s, half_s,
            ))

        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    """Return True if segments p1-p2 and p3-p4 properly intersect."""
    def _segments_intersect(
            self,
            p1: Tuple[float, float],
            p2: Tuple[float, float],
            p3: Tuple[float, float],
            p4: Tuple[float, float],
    ) -> bool:
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        return (
            ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
        )

    """
    Generate load demands (MW).
    PSs have zero demand. SS demands follow Gaussian distributions
    (absolute value to ensure positivity).
    """
    def _generate_demands(
            self,
            n_ps: int,
            n_ss_center: int,
            n_ss_suburbs: int,
            rng: np.random.Generator,
    ) -> np.ndarray:
        d_center = np.abs(rng.normal(self.demand_mean_center,
                                     self.demand_std_center, n_ss_center))
        d_suburbs = np.abs(rng.normal(self.demand_mean_suburbs,
                                      self.demand_std_suburbs, n_ss_suburbs))
        return np.concatenate([
            np.zeros(n_ps),
            d_center,
            d_suburbs,
        ])

    """
    Compute raw arc costs:
        cost = base_cost_per_km * (1 + c) * euclidean_distance
    with an additional obstacle_cost if the arc crosses x = obstacle_x.
    c ~ Uniform(0, 0.1).
    """
    def _compute_raw_costs(
            self,
            nodes: np.ndarray,
            obstacle: List[Tuple[float, float]],
            rng: np.random.Generator,
    ) -> Dict[Tuple[int, int], float]:
        n = len(nodes)
        costs = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(nodes[i] - nodes[j]))
                c = float(rng.uniform(0.0, 0.1))
                cost = self.base_cost_per_km * (1.0 + c) * dist

                # Obstacle crossing: add penalty if arc crosses any polyline segment
                pi = (float(nodes[i][0]), float(nodes[i][1]))
                pj = (float(nodes[j][0]), float(nodes[j][1]))
                for k in range(len(obstacle) - 1):
                    if self._segments_intersect(pi, pj, obstacle[k], obstacle[k + 1]):
                        cost += self.obstacle_cost
                        break

                costs[(i, j)] = cost
        return costs

    """
    Replace direct arc costs with all-pairs shortest path costs,
    as done in the paper to derive the final weighted graph.
    """
    def _shortest_path_costs(
            self,
            nodes: np.ndarray,
            raw_costs: Dict[Tuple[int, int], float],
    ) -> Dict[Tuple[int, int], float]:
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        for (i, j), cost in raw_costs.items():
            G.add_edge(i, j, weight=cost)

        sp = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

        return {
            (i, j): sp[i][j]
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
        }

    """
    Intentionally violate the triangle inequality for a random fraction
    of arcs by reducing their cost to below the sum of two alternatives.
    """
    def _violate_triangle(
            self,
            costs: Dict[Tuple[int, int], float],
            rng: np.random.Generator,
    ) -> Dict[Tuple[int, int], float]:
        keys = list(costs.keys())
        n_vio = int(len(keys) * self.triangle_vio_fraction)
        indices = rng.choice(len(keys), size=n_vio, replace=False)
        for idx in indices:
            i, j = keys[idx]
            costs[(i, j)] *= float(rng.uniform(0.3, 0.7))
        return costs