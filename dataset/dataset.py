import csv
from pathlib import Path

import numpy as np

from network_data import NetworkData

DATASET_DIR = Path(__file__).parent

# Known metadata for base cases (from main.py CASE_STUDIES)
_CASE_METADATA = {
    "Case54":  dict(n_ps_center=2, n_ps_suburbs=2,  n_ss_center=25, n_ss_suburbs=25, density_center=3.00, density_suburbs=0.74),
    "Case78":  dict(n_ps_center=1, n_ps_suburbs=2,  n_ss_center=30, n_ss_suburbs=45, density_center=7.75, density_suburbs=3.92),
    "Case104": dict(n_ps_center=1, n_ps_suburbs=3,  n_ss_center=60, n_ss_suburbs=40, density_center=7.63, density_suburbs=1.59),
    "Case154": dict(n_ps_center=2, n_ps_suburbs=2,  n_ss_center=80, n_ss_suburbs=70, density_center=6.83, density_suburbs=1.50),
}


def load_network(name: str) -> NetworkData:
    base_name = name

    folder = DATASET_DIR / base_name
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    nodes_csv    = folder / f"{name}_Nodes.CSV"
    branches_csv = folder / f"{name}_Branches.CSV"

    for path in (nodes_csv, branches_csv):
        if not path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {path}")

    # ------------------------------------------------------------------ nodes
    coordinates, demands, raw_types = _load_nodes(nodes_csv)

    ps_indices = [i for i, t in enumerate(raw_types) if t == "PS"]
    ss_indices = [i for i, t in enumerate(raw_types) if t == "SS"]

    meta  = _CASE_METADATA.get(base_name)
    types = _classify_types(coordinates, raw_types, ps_indices, ss_indices, meta)

    # Count the actual split after classification
    n_ps_center  = sum(1 for t in types if t == "PS_center")
    n_ps_suburbs = sum(1 for t in types if t == "PS_suburbs")
    n_ss_center  = sum(1 for t in types if t == "SS_center")
    n_ss_suburbs = sum(1 for t in types if t == "SS_suburbs")

    # Use known densities when available, fall back to 0
    density_center  = meta["density_center"]  if meta else 0.0
    density_suburbs = meta["density_suburbs"] if meta else 0.0

    side_center  = np.sqrt(n_ss_center  / density_center)  if density_center  else 0.0
    side_suburbs = np.sqrt(n_ss_suburbs / density_suburbs) if density_suburbs else 0.0

    # --------------------------------------------------------------- branches
    costs, capacities = _load_branches(branches_csv)

    return NetworkData(
        nodes=coordinates,
        types=types,
        demands=demands,
        costs=costs,
        capacities=capacities,
        ps_indices=ps_indices,
        ss_indices=ss_indices,
        obstacle_polyline=[],
        side_center=side_center,
        side_suburbs=side_suburbs,
        n_ps_center=n_ps_center,
        n_ps_suburbs=n_ps_suburbs,
        n_ss_center=n_ss_center,
        n_ss_suburbs=n_ss_suburbs,
        density_center=density_center,
        density_suburbs=density_suburbs,
        seed=0,
    )


def _load_nodes(path: Path):
    coordinates = []
    demands     = []
    raw_types   = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coordinates.append([float(row["x (km)"]), float(row["y (km)"])])
            demands.append(float(row["load (MW)"]))
            raw_types.append(row["typology"].strip())

    return np.array(coordinates), np.array(demands), raw_types


def _load_branches(path: Path):
    costs = {}
    capacities = {}

    with open(path, newline="", encoding="cp1252") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            i        = int(row[0]) - 1   # convert to 0-based index
            j        = int(row[1]) - 1
            capacity = float(row[2])
            cost     = float(row[3])
            key      = (min(i, j), max(i, j))
            costs[key]      = cost
            capacities[key] = capacity

    return costs, capacities


def _classify_types(coordinates, raw_types, ps_indices, ss_indices, meta):
    types = [""] * len(raw_types)

    if meta is None:
        for i, t in enumerate(raw_types):
            types[i] = "PS_center" if t == "PS" else "SS_center"
        return types

    # PSs: n_ps_center closest to origin → PS_center
    ps_by_dist = sorted(ps_indices, key=lambda i: np.linalg.norm(coordinates[i]))
    n_ps_center = meta["n_ps_center"]
    ps_center_set = set(ps_by_dist[:n_ps_center])
    for i in ps_indices:
        types[i] = "PS_center" if i in ps_center_set else "PS_suburbs"

    # SSs: n_ss_center closest to origin → SS_center
    ss_by_dist = sorted(ss_indices, key=lambda i: np.linalg.norm(coordinates[i]))
    n_ss_center = meta["n_ss_center"]
    ss_center_set = set(ss_by_dist[:n_ss_center])
    for i in ss_indices:
        types[i] = "SS_center" if i in ss_center_set else "SS_suburbs"

    return types
