import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from data.network_data import NetworkData
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
    Plot the network nodes and optionally arc costs.
    """
    def plot(
            self,
            net: NetworkData,
            title: str = "Network",
            show_arc_costs: bool = False,
            figsize: Tuple[int, int] = (8, 8),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)

        if show_arc_costs:
            costs_vals = list(net.costs.values())
            vmin, vmax = min(costs_vals), max(costs_vals)
            cmap = plt.cm.RdYlBu_r
            for (i, j), cost in net.costs.items():
                xi, yi = net.nodes[i]
                xj, yj = net.nodes[j]
                color = cmap((cost - vmin) / (vmax - vmin + 1e-9))
                ax.plot([xi, xj], [yi, yj], color=color, linewidth=0.3,
                        alpha=0.4, zorder=1)
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Arc cost (EUR)', shrink=0.6)

        # Draw nodes
        for idx, (xy, t) in enumerate(zip(net.nodes, net.types)):
            if t in ('PS_center', 'PS_suburbs'):
                color = '#D85A30' if t == 'PS_center' else '#A0391A'
                ax.plot(*xy, 's', color=color, markersize=12,
                        zorder=3, markeredgecolor='#993C1D', markeredgewidth=1)
                ax.text(xy[0], xy[1] + 0.18, f'PS{idx}',
                        ha='center', fontsize=8, color='#993C1D', fontweight='bold')
            elif t == 'SS_center':
                ax.plot(*xy, 'o', color='#378ADD', markersize=7,
                        zorder=3, markeredgecolor='#185FA5', markeredgewidth=0.8)
            else:
                ax.plot(*xy, 'o', color='#85B8E8', markersize=5,
                        zorder=3, markeredgecolor='#378ADD', markeredgewidth=0.6)

        # Draw obstacle polyline (snake)
        poly_x = [p[0] for p in net.obstacle_polyline]
        poly_y = [p[1] for p in net.obstacle_polyline]
        ax.plot(poly_x, poly_y,
                color='black', linewidth=2, linestyle='--', zorder=2, label='obstacle')
        ax.text(poly_x[-1] + 0.05, poly_y[-1], 'obstacle',
                fontsize=8, color='black')

        # Draw zone boxes
        hc = net.side_center / 2
        hs = net.side_suburbs / 2
        rect_c = mpatches.Rectangle((-hc, -hc), net.side_center, net.side_center,
                                    linewidth=1.2, edgecolor='#378ADD',
                                    facecolor='none', linestyle='--', zorder=0)
        rect_s = mpatches.Rectangle((-hs, -hs), net.side_suburbs, net.side_suburbs,
                                    linewidth=1.2, edgecolor='#85B8E8',
                                    facecolor='none', linestyle=':', zorder=0)
        ax.add_patch(rect_c)
        ax.add_patch(rect_s)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#D85A30', label='PS — city centre'),
            mpatches.Patch(color='#A0391A', label='PS — suburbs'),
            mpatches.Patch(color='#378ADD', label='SS — city centre'),
            mpatches.Patch(color='#85B8E8', label='SS — suburbs'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

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