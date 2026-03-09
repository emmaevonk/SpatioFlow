import os
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from anndata import AnnData
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

"""
This module performs pseudobulk DE analysis per condition
"""

def _make_pseudobulk(
    adata,
    sample_col="assigned_sample",
    celltype_col="cell_type",
    treatment_col="assigned_treatment",
    layer="raw_counts" 
):
    """
    Create pseudobulk AnnData object for each cell type.

    For each cell type, raw counts are summed across cells belonging to the same
    biological sample, producing a sample-level count matrix suitable for bulk DE analysis.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw counts and metadata.
    sample_col : str, default="assigned_sample"
        Column in adata.obs identifying biological samples.
    celltype_col : str, default = "cell_type"
        column in adata.obs specifying cell type annotations.
    treatment_col : str, default = "assigned_treatment"
        Column in adata.obs specifying condition or treatment.
    layer : str, default = "raw_counts"
        Name of layer containing raw integer count data.

    Returns
    -------
    dict
        Dictionary mapping each cell type to a pseudobulk AnnData object where:
            - rows = samples
            - columns = genes
            - .obs contains sample-level treatment metadata
    
    Notes
    -----
    - Counts are summed per sample within each cell type
    - Assumes the specified layer contains raw (unnormalized) counts
    """
    pseudobulk_dict = {}
    genes = adata.var_names

    for ct in adata.obs[celltype_col].unique():
        print(f"Processing {ct}...")

        ad_ct = adata[adata.obs[celltype_col] == ct].copy()

        # Use raw counts from layer
        X = ad_ct.layers[layer]

        df = pd.DataFrame.sparse.from_spmatrix(
            X,
            index=ad_ct.obs[sample_col],
            columns=genes
        )

        # Sum counts per sample
        pb_df = df.groupby(level=0).sum()

        # Create AnnData
        pb = sc.AnnData(pb_df)

        # Add metadata
        pb.obs[sample_col] = pb_df.index

        pb.obs["assigned_treatment"] = (
            ad_ct.obs[[sample_col, treatment_col]]
            .drop_duplicates()
            .set_index(sample_col)
            .loc[pb_df.index, treatment_col]
        )

        pseudobulk_dict[ct] = pb
    return pseudobulk_dict

# TODO: change the defaults
def pseudobulk(
    adata: AnnData,
    celltype: str = "Macrophages",
    cond: list = ["WT_PBS", "MDX52_ASO_GIVI"],
    save: bool = True,
    output_path: str | None = None
):
    """
    Perform pseudobulk diferential expression analysis for a specific cell type.

    This function:
    1. Generates pseudobulk counts per sample
    2. Selects a specified cell type
    3. Run DESeq2 using treatments as the design factor
    4. Identifies significantly differentially expressed genes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw counts and metadata.
    celltype : str, default = "Macrophages"
        Cell type to analyze.
    cond : list[str], default = ["WT_PBS", "MDX52_ASO_GIVI"].
        Two treatment groups to compare. [reference_condition, comparison_condition]
    save : bool, default = True
        Whether to save the significant genes to a CSV file.
    output_path : str or None, default = None
        Path to save the CSV file. If None, saves to the curent working directory
        with an automatically generated filename.

    Returns
    -------
    pd.DataFrame
        Dataframe containing significant genes with:
        - padj < 0.05
        - |log2FoldChange| > 0.25

    Notes
    -----
    - Counts are converted to integers (required for DESeq2)
    - The design formula is: ~ assigned_treatment
    - PyDESeq2 is used as a Python implementation of DESeq2
    - If save = True, results are written to disk
    """
    pb_dict = _make_pseudobulk(adata)
    pb = pb_dict[celltype].copy()

    # prepare data
    counts = pd.DataFrame(
        pb.X.toarray(),
        index=pb.obs_names,
        columns=pb.var_names
    )

    metadata = pb.obs[["assigned_treatment"]]

    # Ensure counts are integers
    counts = counts.astype(int) 

    # Run DESeq2
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors="assigned_treatment",
        refit_cooks=True
    )

    dds.deseq2()

    contrast_list = ["assigned_treatment", cond[0], cond[1]]
    stat_res = DeseqStats(dds, contrast=contrast_list)
    stat_res.summary()

    de_df = stat_res.results_df

    # Get significant genes
    sig = de_df.query("padj < 0.05 & abs(log2FoldChange) > 0.25")

    if save:
        if output_path is None:
            output_path = os.path.join(os.getcwd(), f"sig_genes_{contrast_list[1]}_{contrast_list[2]}.csv")
        sig.to_csv(output_path)
        print(f"The significant genes are written to a CSV file in the current running directory: {output_path}")

