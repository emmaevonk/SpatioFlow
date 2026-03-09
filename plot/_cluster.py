import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import pandas as pd


def cluster(
    adata: AnnData,
    colors: list = ["transcript_counts", "nucleus_count", "cell_area", "segmentation_method"],
    n_comps: int = 50,
    n_neighbors: int = 15,
    n_pcs: int = 30,
    color_map: str = "viridis",
    palette: str | None = None
):
    """
    Compute UMAP clustering for specified columns. 

    This function generates a UMAP clustering, computing PCA, neighbors and 
    Leiden clustering if necessary using Scanpy. Optionally it generates
    a UMAP containing specified columns to color by the user.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates.
    colors : list
        List of columns present in adata.obs to color the UMAP with.
    n_comps : int, default = 50
        Number of principal components to compute for the PCA.
    n_neighbors : int, default = 15
        Size of the local neighborhood used for manifold approximation.
    n_pcs : int, default = 30
        Amount of principal components to use when computing nearest
        neigbor.
    color_map : str, default = "viridis"
        Color map to use for continuous variables.
    palette : str
        Palette used for the UMAP.

    Returns
    -------
    AnnData
        AnnData object containing Leiden clustering, UMAP clustering and 
        neighbors.

    Notes
    -----
    This function adds columns to adata.obs (leiden, pca, and umap).

    Examples
    --------
    >>> cluster(adata, colors=["n_counts"], n_neighbors=10)

    """
    # PCA
    if "pca" not in adata.uns:
        sc.tl.pca(adata, n_comps=n_comps)

    # Neighbor graph
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs
        )

    # UMAP
    sc.tl.umap(adata)
    if palette is None:
        sc.pl.umap(
            adata,
            color=colors,
            color_map=color_map,
            ncols=3,
        )
    else:
        sc.pl.umap(
            adata,
            color=colors,
            color_map=color_map,
            palette=palette,
            ncols=3,
        )
    return adata
    