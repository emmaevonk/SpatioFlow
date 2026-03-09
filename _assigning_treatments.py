import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from spatialdata import SpatialData
from pathlib import Path
from matplotlib.patches import Rectangle

"""
This module assigns treatment groups/conditions to the AnnData object
using an Excel with coordinates of the treatment groups/conditions.
"""

# TODO: change the default values!
def assign_condition(
    adata: AnnData,
    sdata: SpatialData,
    path_to_excel: Path,
    path_to_xenium: Path = "/exports/archive/hg-groep-spitali/Emma/xenium-resegmentation/experiment_AFM/nuclei_slide1_AFM_ranger_2601",
    sep: str = ';',
    slideid: int | None = None,
    slidecolumn: str = "slideid",
    visual: bool = True
) -> AnnData:
    """
    Assign conditions or treatment groups using coordinates.

    The centroids of the cells are used as coordinates to assign
    conditions to the cells. Based on these coordinates, cells are
    assigned to a treatment group or are getting the class 'unassigned'.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with morphology information. 
    sdata : SpatialData
        Must contain the scaling factor.
    path_to_excel : str
        Path to the excel file containing coordinates and sample IDs.
    path_to_xenium : str
        Path to the Xenium output. Must contain cells.parquet file.
    sep : str, default = ';'
        Seperator in the Excel file.
    slideid : int, default = 91
        Slideid of the Xenium output as mentioned in the Excel file.
    slidecolumn : str, default = 'slideid'
        The name of the column in the Excel file mentioneing the slide ID.
    visual : bool, default = True
        Whether or not to show the visualizatinos for the assignment per
        sample.

    Returns
    -------
    AnnData
        Annotated data matrix where centroids of the cells and the conditions
        have been added to the object.

    Notes
    -----
    Annotated data matrix is changed by adding the treatment groups or 
    conditions in the 'assigned_treatments' column.
    """
    treatments = pd.read_csv(path_to_excel, sep=sep)

    if slidecolumn in treatments:
        if slideid is not None:
            treatments = treatments[treatments[slidecolumn] == slideid]
    print(treatments)

    # Transform the dataframe a bit so everything has its own colums
    if "coordinates" in treatments.columns:
        coords = treatments["coordinates"].str.split(",", expand=True).astype(int)
    else:
        print(f"Coordinates column is not present in the Excel file. Terminating the process...")
        return adata
    coords.columns = ["xmin", "xmax", "ymin", "ymax"]

    treatments = pd.concat([treatments.drop(columns="coordinates"), coords], axis=1)
    if 'outs' in path_to_xenium:
        cells_parquet = pd.read_parquet(f"{path_to_xenium}/cells.parquet")
    else:
        cells_parquet = pd.read_parquet(f"{path_to_xenium}/outs/cells.parquet")

    # merge data of centroids with anndata object
    adata.obs = adata.obs.merge(
        cells_parquet[['cell_id', 'x_centroid', 'y_centroid']],
        on='cell_id',
        how='left'
    )

    # Check transformations for shapes
    if 'cell_boundaries' in sdata.shapes:
        cb = sdata.shapes['cell_boundaries']
        scale_transform = cb.attrs["transform"]["global"]
        scale_x, scale_y = scale_transform.scale

    # Scale the coordinates
    if "x_centroid" in adata.obs:
        adata.obs['x_morphology_focus'] = adata.obs['x_centroid'] * scale_x
        adata.obs['y_morphology_focus'] = adata.obs['y_centroid'] * scale_y
    else:
        adata.obs['x_morphology_focus'] = adata.obs['x_centroid_y'] * scale_x
        adata.obs['y_morphology_focus'] = adata.obs['y_centroid_y'] * scale_y

    # Update obsm
    adata.obsm['spatial_morphology'] = adata.obs[['x_morphology_focus', 'y_morphology_focus']].values
    
    # Run the assignment
    adata = _assign_treatment_by_boundaries(adata, treatments)
    if visual:
        _visualize_conditions(adata, treatments)
    return adata


def _assign_treatment_by_boundaries(
        adata : AnnData, 
        treatment_df : pd.DataFrame
        ) -> AnnData:
    """
    Using cell boundaries to assign treatments to the AnnData object.

    This function uses the morphology focus present in the AnnData
    object to assign treatments based on the maximum and minimum 
    coordinates.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain morphology focus columns.
    treatment_df : pd.DataFrame
        Dataframe containing information about the coordinates of the 
        conditions or treatment groups.
    
    Returns
    -------
    AnnData
        Annotated data matrix where 3 columns are added to adata.obs:
            1. assigned_sample
            2. assigned_treatment
            3. assigned_slideid
    """
    adata.obs['assigned_sample'] = 'unassigned'
    adata.obs['assigned_treatment'] = 'unassigned'
    adata.obs['assigned_slideid'] = 0
    
    for idx, row in treatment_df.iterrows():
        mask = (
            (adata.obs['x_morphology_focus'] >= row['xmin']) &
            (adata.obs['x_morphology_focus'] <= row['xmax']) &
            (adata.obs['y_morphology_focus'] >= row['ymin']) &
            (adata.obs['y_morphology_focus'] <= row['ymax'])
        )
        
        n_cells = mask.sum()

        # Check the assigned treatments for the samples
        print(f"Sample {row['sample ']:>3} ({row['treatment']:15s}): {n_cells:6d} cells assigned")
        
        adata.obs.loc[mask, 'assigned_sample'] = row['sample ']
        adata.obs.loc[mask, 'assigned_treatment'] = row['treatment']
        adata.obs.loc[mask, 'assigned_slideid'] = row['slideid']    
    return adata

def _visualize_conditions(
        adata: AnnData,
        treatments: pd.DataFrame
):
    """
    Visualize the assignment of treatment groups/conditions to validate coordinates.

    This function allows for validation by the user by showing a visualization of the
    Xenium slide and which of the cells in the samples belong to which condition.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must contain the assigned treatments/conditions.
    treatments : pd.DataFrame
        The assigned treatments/conditions in dataframe format.

    Notes
    -----
    This function shows a visualization of the assignment of the conditions/treatment groups
    with different colors. It does not change the AnnData object, hence, nothing is returned.
    """
    plt.figure(figsize=(8, 10))

    # Convert treatment labels to categorical
    treatment_cat = adata.obs["assigned_treatment"].astype("category")
    treatment_codes = treatment_cat.cat.codes

    # Scatter cells
    scatter = plt.scatter(
        adata.obs["x_morphology_focus"],
        adata.obs["y_morphology_focus"],
        c=treatment_codes,
        s=1
    )

    # Draw treatment boundary rectangles
    for _, row in treatments.iterrows():
        width = row["xmax"] - row["xmin"]
        height = row["ymax"] - row["ymin"]

        rect = Rectangle(
            (row["xmin"], row["ymin"]),
            width,
            height,
            fill=False
        )
        plt.gca().add_patch(rect)

    # Automatically generate legend from scatter object
    handles, _ = scatter.legend_elements()
    labels = treatment_cat.cat.categories
    plt.legend(handles, labels, title="Treatment", loc="lower right")

    plt.gca().invert_yaxis()  # Important for Xenium coordinate system
    plt.xlabel("X (morphology space)")
    plt.ylabel("Y (morphology space)")
    plt.title("Treatment assignment across slide")
    plt.savefig("treatments_across_slides.png")
    plt.show()

