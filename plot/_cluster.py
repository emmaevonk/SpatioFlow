import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt


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
    

# TODO add more parameters here
def cells_by_clustering(adata):
    if 'leiden' not in adata:
        print("Leiden algorithm is not applied to adata. Please do this first before calling this function.")
        return
    adata.obs["cluster"] = "Cluster " + (adata.obs["leiden"].astype(int) + 1).astype(str)
    adata.obsm["X_spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    sc.pl.umap(
        adata,
        color="cluster",
        title="UMAP Projection of Cells by Clustering",
        frameon=True,
        legend_loc="right margin",
        ax=axes[0],
        show=False
    )

    sc.pl.embedding(
        adata,
        basis="spatial",
        color="cluster",
        title="Cell Coordinates by Cluster",
        frameon=True,
        legend_loc="right margin",
        ax=axes[1],
        show=False
    )       

    # Flip Y axis so tissue orientation matches Xenium viewer (origin top-left)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("X coordinate (µm)")
    axes[1].set_ylabel("Y coordinate (µm)")

    plt.tight_layout()
    plt.savefig("xenium_umap_and_spatial.png", dpi=150, bbox_inches="tight")
    plt.show()
