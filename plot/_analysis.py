import os
from matplotlib import pyplot as plt
import seaborn as sns
import squidpy as sq
import scanpy as sc
import spatialdata as sd
import spatialdata_plot
from spatialdata import SpatialData
from anndata import AnnData


def dotplot(
    adata: AnnData,
    markers: list | None = None,
    cluster: str = 'leiden'
):  
    """
    Generating dotplots showing markers and their expression in the cells.

    This function generates a dotplot by utilizing the Scanpy module.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. It must contain the ranked genes.
    markers : list
        List of the markers shown in the dotplot.
    cluster : str, default = 'leiden'
        Clusters shown in the dotplot.

    Notes
    ------
    sc.tl.rank_genes needs to be run first before calling this function.
    """
    if markers is None:
        sc.settings.set_figure_params(dpi=500)
        # sc.pl.rank_genes_groups_dotplot(
        #     adata,
        #     n_genes=5,
        #     save="/exports/archive/hg-funcgenom-research/evonk/assigning_labels/_dotplotv2.png"
        # )
        sc.pl.rank_genes_groups_dotplot(adata, n_genes=5)
    else:
        sc.pl.dotplot(adata, var_names=markers, groupby=cluster, show=True, use_raw=False)

def celltype_composition(
        adata: AnnData,
        condition: str = "condition",
        celltype: str = 'cell_type',
        output_path: str | None = None,
        show: bool = True,
        save: bool = False
):
    composition_counts = adata.obs.groupby([condition, celltype]).size().reset_index(name='count')
    composition_counts['fraction'] = composition_counts.groupby(condition)['count'].transform(lambda x: x / x.sum())

    # Plot stacked bar chart
    plot_df = composition_counts.pivot(index=condition, columns=celltype, values='fraction').fillna(0)

    plot_df.plot(kind='bar', stacked=True, figsize=(8,5))
    plt.ylabel('Fraction of cells')
    plt.title('Cell type composition per assigned treatment')
    if show:
        plt.legend(title='Cell type', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()
    if save:
        plt.tight_layout()
        if output_path is None:
            output_path = os.path.join(os.getcwd(), "celltype_composition.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"The celltype composition figure is saved in the current running directory: {output_path}")