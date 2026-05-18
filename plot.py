import datetime
import os

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from network_data import NetworkData
from typing import List, Optional, Tuple


def plot_network(
        net: NetworkData,
        title: str = "Network",
        show_arc_costs: bool = False,
        show_area: bool = True,
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
            ax.text(xy[0], xy[1] + 0.14, f'SS{idx}',
                    ha='center', fontsize=6, color='#185FA5', fontweight='bold')
            ax.text(xy[0], xy[1] - 0.18, f'{net.demands[idx]:.1f} MW',
                    ha='center', fontsize=5.5, color='#185FA5')
        else:
            ax.plot(*xy, 'o', color='#85B8E8', markersize=5,
                    zorder=3, markeredgecolor='#378ADD', markeredgewidth=0.6)
            ax.text(xy[0], xy[1] + 0.14, f'SS{idx}',
                    ha='center', fontsize=6, color='#378ADD', fontweight='bold')
            ax.text(xy[0], xy[1] - 0.18, f'{net.demands[idx]:.1f} MW',
                    ha='center', fontsize=5.5, color='#378ADD')

    # Draw obstacle polyline (snake)
    if net.obstacle_polyline:
        poly_x = [p[0] for p in net.obstacle_polyline]
        poly_y = [p[1] for p in net.obstacle_polyline]
        ax.plot(poly_x, poly_y,
                color='black', linewidth=2, linestyle='--', zorder=2, label='obstacle')
        ax.text(poly_x[-1] + 0.05, poly_y[-1], 'obstacle',
                fontsize=8, color='black')

    if show_area:
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


def plot_solution(
        arcs: Optional[List[Tuple[int, int]]],
        obj: Optional[float],
        net: NetworkData,
        title: str = "Solution",
        show_area: bool = False,
        figsize: Tuple[int, int] = (8, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    # Draw all candidate arcs as faint background edges
    for (i, j) in net.costs:
        xi, yi = net.nodes[i]
        xj, yj = net.nodes[j]
        ax.plot([xi, xj], [yi, yj], color='#CCCCCC', linewidth=0.4,
                alpha=0.5, zorder=1)

    # Draw selected arcs
    if arcs:
        for (i, j) in arcs:
            xi, yi = net.nodes[i]
            xj, yj = net.nodes[j]
            ax.plot([xi, xj], [yi, yj], color='#2CA02C', linewidth=1.8,
                    alpha=0.9, zorder=2)

    # Draw nodes
    for idx, (xy, t) in enumerate(zip(net.nodes, net.types)):
        if t in ('PS_center', 'PS_suburbs'):
            color = '#D85A30' if t == 'PS_center' else '#A0391A'
            ax.plot(*xy, 's', color=color, markersize=12,
                    zorder=4, markeredgecolor='#993C1D', markeredgewidth=1)
            ax.text(xy[0], xy[1] + 0.18, f'PS{idx}',
                    ha='center', fontsize=8, color='#993C1D', fontweight='bold')
        elif t == 'SS_center':
            ax.plot(*xy, 'o', color='#378ADD', markersize=7,
                    zorder=4, markeredgecolor='#185FA5', markeredgewidth=0.8)
            ax.text(xy[0], xy[1] + 0.14, f'SS{idx}',
                    ha='center', fontsize=6, color='#185FA5', fontweight='bold')
            ax.text(xy[0], xy[1] - 0.18, f'{net.demands[idx]:.1f} MW',
                    ha='center', fontsize=5.5, color='#185FA5')
        else:
            ax.plot(*xy, 'o', color='#85B8E8', markersize=5,
                    zorder=4, markeredgecolor='#378ADD', markeredgewidth=0.6)
            ax.text(xy[0], xy[1] + 0.14, f'SS{idx}',
                    ha='center', fontsize=6, color='#378ADD', fontweight='bold')
            ax.text(xy[0], xy[1] - 0.18, f'{net.demands[idx]:.1f} MW',
                    ha='center', fontsize=5.5, color='#378ADD')

    # Draw obstacle polyline
    if net.obstacle_polyline:
        poly_x = [p[0] for p in net.obstacle_polyline]
        poly_y = [p[1] for p in net.obstacle_polyline]
        ax.plot(poly_x, poly_y,
                color='black', linewidth=2, linestyle='--', zorder=3, label='obstacle')
        ax.text(poly_x[-1] + 0.05, poly_y[-1], 'obstacle',
                fontsize=8, color='black')

    if show_area:
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
        mpatches.Patch(color='#2CA02C', label='Selected arc'),
        mpatches.Patch(color='#CCCCCC', label='Candidate arc'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    cost_str = f"  —  obj = {obj:,.0f} EUR" if obj is not None else "  —  no feasible solution"
    ax.set_title(f"{title}{cost_str}", fontsize=13, fontweight='bold')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_scalability_results(results, max_time, algorithms):
    if not any(results.values()):
        print("Nothing to plot.")
        return

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]

    for idx, (name, _) in enumerate(algorithms):
        runs = results[name]
        if not runs:
            continue
        color = colors[idx % len(colors)]
        xs = [r["n_nodes"] for r in runs]
        ys = [r["t"] for r in runs]
        plt.plot(xs, ys, marker="o", linewidth=2, color=color, label=name)


    plt.axhline(max_time, color="red", linestyle=":", alpha=0.6,
                label=f"time cap = {max_time:.1f}s")
    plt.xlabel("Network size (number of nodes)")
    plt.ylabel("Execution time [s]")
    plt.title("Scalability of the power-distribution-network solvers")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    out_path = os.path.join(out_dir, f"scalability_{stamp}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Chart saved to: {out_path}")
    plt.show()