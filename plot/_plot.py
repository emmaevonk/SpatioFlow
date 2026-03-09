from matplotlib import pyplot as plt
import squidpy as sq
import scanpy as sc
import spatialdata as sd
import spatialdata_plot
from spatialdata import SpatialData
from anndata import AnnData

"""
Visualization and exploratory analysis utilities for spatial transcriptomics.

This module provides plotting functions for:
- Spatial visualization
- Quality control distributions
- Dimensionality reduction
- Marker gene ranking
- Image exploration and cropping

Designed for AnnData and SpatialData objects
"""

def plot_image(
        adata: AnnData,
        colors: list = ["leiden"]
):
    """
    Plot spatial scatter colored by gene expression or metadata
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates in 
        adata.obsm["spatial"]
    colors : list of str
        Features or observation columns used for coloring. 
        Examples: gene names, "leiden"

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the spatial scatter plot.

    Notes
    -----
    Wrapper around sq.pl.spatial_scatter
    """
    sq.pl.spatial_scatter(
        adata,
        library_id="spatial",
        color=colors,
        shape=None,
        size=2,
        img=False,
    )

def rank_genes_group(
        adata: AnnData,
        res: float = 1
) -> AnnData:
    """
    Identify marker genes per Leiden cluster.

    Parameters
    ----------
    adata : AnnData
        Must contain adata.obs["leiden"].
    res : float, default = 0.3
        Resolution used for Leiden clustering

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Results stored in adata.uns["rank_genes_groups"].
    Uses Wilcoxon rank-sum test.
    """
    if "leiden" not in adata.obs:
        print("Leiden clustering has not been performed yet. Adding 'Leiden' column...")
        sc.tl.leiden(adata, resolution=res, random_state=42)

    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden",
        method="wilcoxon"
    )

    # Plot ranking of genes. Plot top 10 genes per cluster
    sc.pl.rank_genes_groups(adata, n_genes=10)
    return adata