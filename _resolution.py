"""
Testing which resolution value for Leiden Clustering is best for the data.
"""

#TODO: document this script

import numpy as np
import scanpy as sc
import igraph as ig
import leidenalg
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _build_graph_from_h5ad(
    adata,
    n_neighbors=15,
    n_pcs=30,
):
    # If PCA not present, compute it
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=n_pcs)

    # Build kNN graph (reuses PCA automatically)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    adj = adata.obsp["connectivities"]

    sources, targets = adj.nonzero()
    weights = adj[sources, targets].A1

    graph = ig.Graph(
        n=adj.shape[0],
        edges=list(zip(sources, targets)),
        edge_attrs={"weight": weights},
        directed=False
    )
    return graph, adata


def _compute_stability_fast(partitions):
    base = partitions[0]
    scores = [
        adjusted_rand_score(base, p)
        for p in partitions[1:]
    ]
    return np.mean(scores) if scores else 1.0


def _leiden_auto_resolution(
    graph,
    n_cells,
    resolution_range=(0.2, 2.0),
    resolution_step=0.2,
    n_runs=3,
    random_state=42,
    fast_mode=False,
):
    rng = np.random.default_rng(random_state)
    resolutions = np.arange(
        resolution_range[0],
        resolution_range[1] + resolution_step,
        resolution_step
    )

    results = []
    for res in resolutions:
        partitions = []
        modularities = []
        runs = 1 if fast_mode else n_runs
        for _ in range(runs):
            seed = int(rng.integers(0, 1_000_000))

            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=res,
                weights="weight",
                seed=seed,
            )

            partitions.append(partition.membership)
            modularities.append(partition.modularity)

        n_clusters = len(set(partitions[0]))
        stability = 1.0 if fast_mode else _compute_stability_fast(partitions)
        results.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "stability": stability,
            "modularity": np.mean(modularities),
        })
    df = pd.DataFrame(results)

    # Avoid over-fragmentation
    df = df[df["n_clusters"] < np.sqrt(n_cells)]
    df["score"] = 0.7 * df["stability"] + 0.3 * df["modularity"]
    best_row = df.sort_values("score", ascending=False).iloc[0]
    return best_row["resolution"], df


def compute_leiden_resolution(
    adata,
    resolution="auto",
    fast_mode=False,
    n_neighbors=15,
    n_pcs=30,
    n_runs=3,
):
    graph, adata = _build_graph_from_h5ad(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
    )
    if resolution == "auto":
        best_res, table = _leiden_auto_resolution(
            graph,
            n_cells=adata.n_obs,
            n_runs=n_runs,
            fast_mode=fast_mode,
        )
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=best_res,
            weights="weight",
        )
        adata.obs["leiden"] = partition.membership
        return adata, best_res, table
    else:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="weight",
        )
        adata.obs["leiden"] = partition.membership
        return adata, resolution, None