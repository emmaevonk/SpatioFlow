# import packages
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import scvi
import cellcharter as cc
from spatialdata import read_zarr
from lightning.pytorch import seed_everything 
from pathlib import Path
from anndata import AnnData

# TODO: maybe add ARI graph?

def _dim_red(
        adata: AnnData,
        epoch: int = 20
):
    """
    Train an scVI model for dimensionality reduction on spatial transcriptomics data.

    Sets up and trains a Variational Inference model (scVI) on the provided AnnData
    object. The model learns a low-dimensional latent representation of gene expression
    that accounts for technical noise and batch effects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing gene expression counts. Must be compatible
        with scVI's data requirements (raw counts recommended).
    epoch: int, default = 20
        Maximum number of training epochs. Early stopping may halt training before
        this limit is reached. Default is 20.

    Returns
    -------
    scvi.model.SCVI
        A trained scVI model. The latent representation can be extracted via
        ``model.get_latent_representation()``.

    Notes
    -----
    The model is configured with:
    - 1 hidden layer (``n_layers=1``)
    - 10-dimensional latent space (``n_latent=10``)
    - Layer normalization enabled on both encoder and decoder
    - Batch normalization disabled
    """
    scvi.model.SCVI.setup_anndata(adata)
    LOAD_MODEL = True

    model = scvi.model.SCVI(
        adata,
        n_layers=1,
        n_latent=10,
        use_layer_norm="both",
        use_batch_norm="none",
    )

    print(f"Start training model. This can take a while. Do not stop the code while running...")
    model.train(
        max_epochs=epoch,
        early_stopping=True,
        enable_progress_bar=True
    )
    return model


def _plot_epoch(model):
    """
    Plot training and validation reconstruction loss curves across epochs.

    Visualises the reconstruction loss for both the training and validation
    sets over the course of model training. This diagnostic plot helps select 
    an appropriate epoch count. The ideal value sits just before the validation
    loss begins to plateau or diverge from the training loss.

    Parameters
    ----------
    model : scvi.model.scVI
        A trained scVI model
    
    Returns
    -------
    None
        Displays the plots inline (in a Jupyter Notebook). 

    Notes
    -----
    - Training loss is shown in dark green, validation loss is shown in red.
    - A large gap between the two curves may indicate overfitting. Consider 
      reducing the number of epochs or enabling stronger regularisation.
    """
    plt.figure(figsize=(5, 5))
    plt.plot(
        model.history[f"reconstruction_loss_train"],
        label="train",
        color="darkgreen",
        linewidth=1.25
    )

    plt.plot(
        model.history[f"reconstruction_loss_validation"],
        label="validation",
        color="firebrick",
        linewidth=1.25
    )

    plt.legend()
    plt.title("reconstruction_loss")
    plt.tight_layout()


def _neigh_aggr(
        adata: AnnData,
        library_key: str = "sample"
        ):
    """
    Build a spatial neighborhood graph and aggregate neighbor embeddings.

    Constructs a spatial connectivity graph using Squidpy, visualises the
    spatial layout of cells/spots, and aggregates the scVI latent
    representations of each cell's neighbors using CellCharter. The resulting
    neighborhood-aware embedding is stored in ``adata.obsm["X_cellcharter"]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates in ``adata.obsm``.
    library_key : str, default = "sample"
        Column in ``adata.obs`` used to distinguish tissue sections or 
        samples. 

    Returns
    -------
    AnnData
        The input ``adata`` object updated in-place with:
        - ``adata.obsp["spatial_connectivities"]``: sparse connectivity matrix
        from the spatial neighborhood graph.
        - ``adata.obsm["X_cellcharter"]``: aggregated neighbor embeddings

    Notes
    -----
    - The spatial graph uses generic coordinate type with a 99th-percentile
      distance threshold to filter spurious long-range connections.
    - Neighbor aggregation is performed over 3 hops (``n_layers=3``)
    - A spatial scatter plot of the first sample in ``library_key`` is 
      rendered automatically, overlaying the connectivity graph
    """
    # spatial neighborhood graph
    sq.gr.spatial_neighbors(adata, library_key=library_key, coord_type='generic', percentile=99)

    sq.pl.spatial_scatter(
        adata, 
        shape=None, 
        library_key=library_key,
        library_id=adata.obs[library_key].cat.categories[0],
        color=library_key, 
        size=1, 
        figsize=(10,10),
        connectivity_key="spatial_connectivities",
        ncols=1
    )

    cc.gr.aggregate_neighbors(adata, n_layers=3, use_rep="X_scVI", out_key="X_cellcharter", sample_key=library_key)
    return adata


def _cluster(
        adata: AnnData,
        n_cluster: int = 18
):
    """
    Assign spatial domain labels using Guassian Mixture Model clustering.

    Fits a GMM to the CellCharter neighborhood embeddings and assigns each
    cell or spot to one ot ``n_cluster`` spatial domains. Cluster labels are
    stored in ``adata.obs["spatial_domain"]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with CellCharter embeddings in
        ``adata.obsm["X_cellcharter"]``.
    n_cluster : int, default = 18
        Number of spatial domains (GMM components) to identify. 
        This value should be informed by prior knowledge of the tissue 
        architecture of tuned using stability matrices (ARI across runs).
    
    Returns
    -------
    AnnData
        The input ``adata`` object updated in-place with cluster assignments
        in ``adata.obs["spatial_domain"]`` (categorical integer labels).
    
    Notes
    -----
    - The random state is fixed at ``12345`` for reproducibility
    - Cluster labels are integer indices and do not carry inherent biological
    meaning until validated against known marker genes or tissue annotations.
    """
    gmm = cc.tl.Cluster(n_clusters=n_cluster, random_state=12345)
    gmm.fit(adata, use_rep="X_cellcharter")
    adata.obs["spatial_domain"] = gmm.predict(adata, use_rep="X_cellcharter")
    return adata


def stability_cellcharter(
    adata: AnnData,
    n_clusters: tuple[int] = (2, 10),
    max_runs: int = 10,
    convergence_tol: float = 0.01,
    output_dir : Path = ""
):
    """
    Computes the stability graph of CellCharter, showing the clustering stability.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing CellCharter results
    n_clusters : tuple[int], default = (2, 10)
        The amount of clusters checked (min, max).
    max_runs : int, default = 10
        Maximum number of repititions for each value of number of clusters.
    convergence_tol : float, default = 0.01
        Convergence tolerance for the clustering stability. If the Mean Absolute Percentage
        Error between consecutive iterations is below `convergence_tol` the algorithm stops
        at `max_runs`.

    Returns
    -------
    None

    Notes
    -----
    The clustering stability plot is saved as 'clustering_stability_cellcharter.png'.
    """
    autok = cc.tl.ClusterAutoK(
        n_clusters=n_clusters, 
        max_runs=max_runs,
        convergence_tol=convergence_tol
    )
    autok.fit(adata, use_rep="X_cellcharter")
    cc.pl.autok_stability(autok, save=output_dir / "clustering_stability_cellcharter.png")


def run_cellcharter(
        adata: AnnData,
        output_dir: Path | None = None,
        plot: bool = True,
        epoch: int = 20,
        library_key: str = "sample" 
    ):
    """
    End-to-end CellCharter pipeline for spatial domain identification.

    Orchestrate the full analysis pipeline:
    1. Dimensionality reduction via scVI
    2. Spatial neighborhood graph construction and neighbor embedding aggregation
    3. Gaussian Mixture Model clustering into spatial domains
    4. Saving the annotated data matrix (AnnData) to disk

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw gene expression counts and
        spatial coordinates. Must contain a ``sample_id`` column in 
        ``adata.obs`` to distinguish tissue sections.
    output_dir : Path or None, optional
        Directory where outputs are saved. If ``None`` the ``.h5ad`` file
        is written to the current working directory and figures are not 
        saved to disk.
    plot : bool, optional
        If ``True``, display the scVI training loss curve after model fitting.
        Default is ``True``.
    epoch: int, default = 20
        Maximum number of training epochs. Early stopping may halt training before
        this limit is reached. Default is 20.
    library_key : str, default = "sample"
        Column in ``adata.obs`` used to distinguish tissue sections or 
        samples.

    Returns
    -------
    None
        Results are saved to disk as ``adata_with_spatial_domains.h5ad``
        (gzip-compressed). The ``adata`` object is modified in-place with
        the following additions:
        - ``adata.obsm["X_scI"]``: scVI latent representation (float32)
        - ``adata.obsp["spatial_connectivities"]``: spatial connectivity matrix.
        - ``adata.obs["spatial_domain"]``: predicted spatial domain labels.

    Notes
    -----
    - Global random seeds are fixed to ``12345`` via ``seed_everything`` for
    reproducibility across runs.
    - Scanpy's figure output directory is set to ``output_dir`` when provided.

    Examples
    --------
    >>> run_cellcharter(adata, output_dir=Path("results/"), plot=True)
    """
    # set params
    seed_everything(12345)
    scvi.settings.seed = 12345
    sc.settings.figdir = str(output_dir)

    # perform dimensionality reduction
    model = _dim_red(adata, epoch=epoch) 
    if plot:
        _plot_epoch(model)

    adata.obsm["X_scVI"] = model.get_latent_representation(adata).astype(np.float32)

    adata = _neigh_aggr(adata, library_key=library_key) 
    adata = _cluster(adata)
    if output_dir is None:
        adata.write_h5ad("adata_with_spatial_domains.h5ad", compression="gzip")
    else:
        adata.write_h5ad(f"{output_dir}/adata_with_spatial_domains.h5ad", compression="gzip")
