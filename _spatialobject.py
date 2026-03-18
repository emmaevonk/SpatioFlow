import spatialdata as sd
import os
import scanpy as sc
import numpy as np
from pathlib import Path
from spatialdata_io import xenium
from spatialdata import SpatialData, read_zarr
from anndata import AnnData

"""
Input/output utilities for spatial transcriptomics data.

This module provides functions to:
- Read Xenium data into SpatialData format
- Convert SpatialData tables to AnnData objects for downstream analysis

Currently supports Xenium outputs and SpatialData Zarr stores.
"""

def read_data(
    xenium_path: str,
    output_path: str | None = None
):
    """
    Read Xenium data into a SpatialData object.

    This function loads spatial transcriptomics data from either:
    - A raw Xenium output directory
    - An existing SpatialData Zarr store

    Parameters
    ----------
    xenium_path : str or Path
        Path to:
        - Xenium output directory, or
        - Existing ``.zarr`` SpatialData store.
    output_path : str or Path or None, default=None
        If provided, the loaded SpatialData object is written to this
        location as a Zarr store.

    Returns
    -------
    SpatialData
        Loaded spatial dataset.

    Notes
    -----
    - Xenium data are read using ``spatialdata_io.xenium``.
    - If ``xenium_path`` ends with ``.zarr``, the dataset is loaded using
      ``read_zarr``.
    - Writing to ``output_path`` overwrites existing data at that location.

    Examples
    --------
    >>> sdata = read_data("xenium_run/")
    >>> sdata = read_data("data.zarr")
    >>> read_data("xenium_run/", output_path="processed.zarr")
    """
    try:
        if str(xenium_path).endswith(".zarr"):
            sdata = read_zarr(xenium_path)
        else:
            sdata = xenium(xenium_path)
            
        # Write to zarr file in the output directory (path + .zarr) if output path is given
        if output_path is not None:
            if not output_path.endswith(".zarr"):
                output_path = output_path + "sdata.zarr"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                sdata.write(output_path)
            elif os.path.exists(output_path):
                print(f"The path to the output ({output_path}) already exists. Not overwriting an existing file. The sdata file is returned through this function.")
            else:
                sdata.write(output_path)
        return sdata
    except FileNotFoundError as e: 
        print(e)


def convert_sdata_adata(
    sdata: SpatialData | Path,
    table_key: str = "table",
    leiden_resolution: float = 1.0,
) -> AnnData:
    """
    Extract an AnnData table from a SpatialData object and compute Leiden clustering.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing a table layer.
    table_key : str, default="table"
        Key in ``sdata.tables`` containing the expression matrix.
    leiden_resolution : float, default=0.3
        Resolution parameter for Leiden clustering.

    Returns
    -------
    AnnData
        Copy of the selected table with Leiden cluster labels stored in
        ``adata.obs["leiden"]``.

    Raises
    ------
    KeyError
        If ``table_key`` is not present in ``sdata.tables``.

    Notes
    -----
    This function performs the following steps:
    1. Extract table from SpatialData
    2. Compute PCA
    3. Build neighbor graph
    4. Run Leiden clustering

    The original SpatialData object is not modified.

    Examples
    --------
    >>> adata = convert_sdata_adata(sdata)
    >>> adata.obs["leiden"].value_counts()
    """
    adata= sdata.tables[table_key].copy()

    # add leiden column to it for later visualizations
    sc.tl.leiden(adata, resolution=leiden_resolution)
    return adata
