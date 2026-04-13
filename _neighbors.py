import squidpy as sq
import matplotlib.pyplot as plt
from anndata import AnnData

def nhood_enrichment(
        adata: AnnData,
        cluster_key: str = "leiden",
        show: bool = True
) -> plt.Figure:
    """
    Compute and visualize neighborhood enrichment between clusters.

    Neighborhood enrichment quantifies whether cells from specific clusters
    are found in spatial proximity more or less often than expected by chance.

    This function computes enrichment statistics using Squidpy and generates
    a heatmap visualization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial connectivity information.
        Spatial neighbors must be computed beforehand using 
        ``sq.gr.spatial_neighbors``
    cluster_key : str, default = "leiden"
        Column in adata.obs containing cluster labels.
    show : bool, default = True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the neighborhood enrichment heatmap.

    Raises
    ------
    KeyError
        If ``cluster_key`` is not found in adata.obs.
    ValueError
        If spatial neighbors are missing.

    Notes
    -----
    Results are stored in adata.uns["nhood_enrichment"]

    Positive values indicate clusters that are spatially enriched together,
    while negative values indicate spatial avoidance.

    Examples
    --------
    >>> import squidpy as sq
    >>> sq.gr.spatial_neighbors(adata)
    >>> fig = nhood_enrichment(adata, cluster_key="leiden")
    """
    # Compute enrichment
    sq.gr.spatial_neighbors(adata) # if error when running this function, change this line (add if statement)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        figsize=(8, 8),
        title="Neighborhood enrichment adata",
    )
    fig.show()
    return fig


def compute_spatial_neighbors(
        adata: AnnData,
        cluster_key: str = "leiden"
) -> AnnData:
    # Compute enrichment
    sq.gr.spatial_neighbors(adata)
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key)

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        figsize=(8, 8),
        title="Neighborhood enrichment adata",
    )
    fig.show()
    return fig