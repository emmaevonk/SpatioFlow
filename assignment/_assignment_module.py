# importing the packages
from __future__ import annotations
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.morphology import closing, disk
from skimage.measure import label

from skimage.segmentation import watershed
from skimage.morphology import erosion, disk as morph_disk
import scipy.ndimage as ndi

# import ipywidgets as widgets
# from IPython.display import display, clear_output


def _load_xenium(
        path: str,
        output_dir: str 
) -> pd.DataFrame:
    """
    Read data from input.

    This function reads the given Xenium path and checks if the input file exists. Additionally, it
    checks the presence of x/y columns and returns a dataframe with this information.

    Parameters
    ----------
    path : str
        Path to the Xenium output directory.
    
    Returns
    -------
    df : pd.DataFrame
        Dataframe containing x/y information of the cells.

    Notes
    -----
    This function does not use an AnnData object, it looks at the cells from the Xenium output to
    obtain their coordinates for further processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ext = path.lower()
    sep = "\t" if ".tsv" in ext else ","
    df  = pd.read_csv(path+"cells.csv.gz", sep=sep, compression="gzip", low_memory=False)

    print(f"Loaded {len(df):,} rows.")

    x_candidates = ["x_centroid", "x_location", "X", "x", "X_centroid"]
    y_candidates = ["y_centroid", "y_location", "Y", "y", "Y_centroid"]
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        print("Could not auto-detect x/y columns. Available:", list(df.columns))

    df = df.rename(columns={x_col: "x", y_col: "y"})
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    return df


def plot_samples(
        df : pd.DataFrame, 
        cluster_ids, 
        highlight: int = None,
         conditions: dict = None, 
         ax: matplotlib.axes = None
):
    """
    Visualize clustered cells in a 2D scatter plot of the slide.

    This function plots cells from a dataframe containing spatial coordinates
    and cluster assignments. Each cluster is rendered with a unique color, 
    optionally highlighting one cluster and annotating clusters with labels.

    Cluster centroids are labeled on the plot, and a legend describing
    cluster size and condition metadata is displayed.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing x/y coordinates of each points, including the 
        cluster/sample identifier for each point.

    cluster_ids : iterable of int
        List or iterable of cluster(s) to plot. each ID corresponds to a 
        group of points.
    
    highlight : int, optional
        A cluster ID to visually emphasize. Highlighted clusters are plotted
        with larger markers, higher opacity, and a different background color.
        Default = None

    conditions : dict[int, str], optional
        If provided, the condition text is appended to the cluster label and 
        shown in both the annotation and legend.

    ax : Axes, optional
        Existing matplotlib axes to plot on.

    Notes
    -----
    - Each cluster in cluster_ids is plotted with a color from the global CMAP colormap
    - Clustered centroids are computed from the mean x and y coordinates and annotated 
      directly on the plot.
      - A legend is provided showing different colors per cluster and which cluster
        is
    """
    standalone = ax is None
    CMAP = plt.get_cmap("tab20")
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#13131f")
    noise = df[df["sample_id"] == -1]
    if len(noise):
        ax.scatter(noise["x"], noise["y"], s=0.3, c="#444455",
                   alpha=0.3, linewidths=0, rasterized=True)
    legend_handles = []
    for cid in cluster_ids:
        pts = df[df["sample_id"] == cid]
        col = CMAP(cid % 20)
        is_sel = (cid == highlight)
        ax.scatter(pts["x"], pts["y"],
                   s=1.5 if is_sel else 0.6,
                   c=[col], alpha=0.9 if is_sel else 0.45,
                   linewidths=0, rasterized=True)
        cx, cy = pts["x"].mean(), pts["y"].mean()
        cond_str = conditions.get(cid, "") if conditions else ""
        lbl = f"S{cid}" + (f"\n{cond_str}" if cond_str else "")
        fc = "#3a7bd5" if is_sel else "#2a2a3e"
        ax.annotate(lbl, (cx, cy), color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, alpha=0.85))
        legend_handles.append(
            mpatches.Patch(color=col,
                           label=f"S{cid}: {cond_str or 'unlabelled'}  ({len(pts):,} cells)"))
    ax.legend(handles=legend_handles,
              loc="upper left",
              bbox_to_anchor=(1.01, 1),
              borderaxespad=0,
              framealpha=0.4,
              fontsize=8,
              facecolor="#1e1e2e",
              labelcolor="white")
    ax.invert_yaxis()
    ax.tick_params(colors="grey")
    for sp in ax.spines.values(): sp.set_edgecolor("#555577")
    ax.set_xlabel("x (um)", color="grey")
    ax.set_ylabel("y (um)", color="grey")
    ax.set_title(f"{len(cluster_ids)} sample(s) detected", color="white", fontsize=11)
    if standalone:
        plt.tight_layout()
        plt.show(fig)

def detect_samples_watershed(
        df: pd.DataFrame, 
        pixel_size_um: int = 20, 
        blur_sigma: int = 3,
        closing_radius: int = 5, 
        erosion_radius: int = 30, 
        min_cells: int = 500
):
    """ 
    Detect spatial samples using grid-based watershed segmentation.

    This function converts spatial cell coordinates into a 2D density grid,
    performing smoothing and morphological preprocessing, and then applies 
    watershed segmentation to identify spatially separated samples.
    Cells are assigned to segmented regions and small regions are filtered
    based on a minimum cell count threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing cell coordinates (x/y)
    
    pixel_size_um : int, default=20
        Pixel size of the intermediate spatial grid in micrometers.
        Larger values produce a coarser grid and faster processing but
        may reduce segmentation resolution.
    
    blur_sigma : int, default=3
        Standard deviation of the Gaussian filter applied to the density
        grid. Controls smoothing of the spatial density map.

    closing_radius : int, default=5
        Radius of the morphological closing operation used to fill small
        gaps in the binary mask.

    erosion_radius : int, default=30
        Radius of the morphological erosion used to generate seed regions
        for watershed segmentation. Larger values enforce stronger separation 
        between nearby samples.

    min_cells : int, default=500
        Minimum number of cells required for a segmented region to be considered
        a valid sample. Regions with fewer cells are laballed as noise.

    Returns
    -------
    final_labels : numpy.ndarray
        Array of length ``len(df)`` assigning a sample ID to each cell.
        Valid samples are labeled with consecutive integers starting
        from 0. Cells not assigned to a valid region are labeled ``-1``.

    labeled_image : numpy.ndarray
        2D watershed segmentation mask where each pixel contains the
        watershed component label.

    x_min : float
        Minimum x-coordinate of the spatial domain. Useful for mapping
        between grid coordinates and original spatial coordinates.

    y_min : float
        Minimum y-coordinate of the spatial domain.

    pixel_size_um : int
        Pixel size used for the spatial grid. Required for converting
        between grid indices and real-world coordinates.

    Notes
    -----
    - The algorithm assumes spatially separated samples with clear
      density gaps between them.
    - The erosion step is critical for separating adjacent samples
      and may require tuning depending on sample spacing.
    - Grid resolution (``pixel_size_um``) and smoothing (``blur_sigma``)
      strongly influence segmentation quality.

    The workflow follows these steps:
    1. Convert x/y cell coordinates into a coarse pixel grid
    2. Smooth the grid using a Gaussian filter to create a density map
    3. Threshold the density map to obtain a binary mask
    4. Apply morphological closing and hole filling to clean the mask.
    5. Erode the mask to create seed regions separating nearby samples
    6. Use distance transform + watershed to segment individual samples.
    7. Assign each cell to a segmented region.
    8. Remove regions containing fewer than the minimum amount of cells defined.
    """
    coords = df[["x", "y"]].values
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    nx = int(np.ceil((x_max - x_min) / pixel_size_um)) + 1
    ny = int(np.ceil((y_max - y_min) / pixel_size_um)) + 1
    print(f"Grid: {nx} x {ny} px  ({pixel_size_um} µm/px)")

    xi = ((coords[:, 0] - x_min) / pixel_size_um).astype(int)
    yi = ((coords[:, 1] - y_min) / pixel_size_um).astype(int)
    grid = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(grid, (yi, xi), 1)

    blurred = gaussian_filter(grid, sigma=blur_sigma)
    threshold = blurred.max() * 0.05
    binary = blurred > threshold
    binary = closing(binary, disk(closing_radius))
    binary = binary_fill_holes(binary)

    # perform erosion to separate close samples
    eroded = erosion(binary, disk(erosion_radius))
    seeds  = label(eroded)
    print(f"Seeds after erosion: {seeds.max()}")

    # create watershed surface by distance transforming
    distance = ndi.distance_transform_edt(binary)
    labeled_image = watershed(-distance, seeds, mask=binary)
    print(f"Components after watershed: {labeled_image.max()}")

    # map cells to labels
    cell_labels = labeled_image[yi, xi] - 1
    unique, counts = np.unique(cell_labels[cell_labels >= 0], return_counts=True)
    keep  = unique[counts >= min_cells]
    mask  = np.isin(cell_labels, keep)
    cell_labels[~mask] = -1
    remap = {old: new for new, old in enumerate(sorted(keep))}
    final_labels = np.array([remap.get(l, -1) for l in cell_labels])

    print(f"\n✓ {len(remap)} sample(s)  |  {(final_labels==-1).sum():,} unassigned cells")
    return final_labels, labeled_image, x_min, y_min, pixel_size_um


SplitRecord = dict  # {"type", "sample_id", "single_value"?  "x_points"? "y_points"?}
SplitHistory = list[SplitRecord]


# ---------------------------------------------------------------------------
# Core split helpers
# ---------------------------------------------------------------------------

def do_one_split(
    df: pd.DataFrame,
    sample_id: int,
    split_type: str,
    single_value: float | None = None,
    x_points: list[float] | None = None,
    y_points: list[float] | None = None,
) -> pd.DataFrame:
    """
    Split a single sample into two new samples along a user-defined cut.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``x``, ``y``, ``sample_id``.
    sample_id : int
        The raw sample ID to split.
    split_type : {"vertical", "horizontal", "diagonal"}
        Type of geometric split.
    single_value : float, optional
        Cut position for vertical (x = value) or horizontal (y = value) splits.
    x_points : list[float], optional
        Two x-coordinates defining a diagonal split line.
    y_points : list[float], optional
        Two y-coordinates defining a diagonal split line.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the target sample replaced by two new samples.
        New sample IDs are ``max(sample_id) + 1`` and ``max(sample_id) + 2``.
    """
    df = df.copy()
    is_target = df["sample_id"] == sample_id

    if is_target.sum() == 0:
        print(f"  WARNING: sample_id {sample_id} not found — skipping.")
        return df

    cx = df.loc[is_target, "x"]
    cy = df.loc[is_target, "y"]

    if split_type == "vertical":
        cross = cx - single_value
    elif split_type == "horizontal":
        cross = cy - single_value
    elif split_type == "diagonal":
        x1, x2 = x_points
        y1, y2 = y_points
        cross = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}. "
                         "Expected 'vertical', 'horizontal', or 'diagonal'.")

    max_id = int(df["sample_id"].max())
    side_a = is_target & pd.Series(cross >= 0, index=df.index).fillna(False)
    side_b = is_target & pd.Series(cross <  0, index=df.index).fillna(False)

    df.loc[side_a, "sample_id"] = max_id + 1
    df.loc[side_b, "sample_id"] = max_id + 2
    return df


def renumber(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Renumber sample IDs sequentially, ordered by y-centroid (bottom → top).

    Cells labelled ``-1`` (noise/unassigned) are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``x``, ``y``, ``sample_id``.

    Returns
    -------
    df : pd.DataFrame
        Copy with remapped ``sample_id`` values starting from 0.
    new_ids : list[int]
        Sorted list of valid sample IDs after renumbering.
    """
    valid_ids = [c for c in df["sample_id"].unique() if c != -1]
    if not valid_ids:
        return df.copy(), []

    centroids  = df[df["sample_id"] != -1].groupby("sample_id")["y"].mean()
    sorted_ids = centroids.sort_values().index.tolist()
    remap      = {old: new for new, old in enumerate(sorted_ids)}

    df = df.copy()
    df["sample_id"] = df["sample_id"].apply(lambda x: remap.get(x, -1))
    new_ids = sorted(c for c in df["sample_id"].unique() if c != -1)
    return df, new_ids


# ---------------------------------------------------------------------------
# History replay
# ---------------------------------------------------------------------------

def replay_all_splits(
    base_df: pd.DataFrame,
    history: SplitHistory,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Reconstruct the current segmentation by replaying a split history.

    Each entry in ``history`` stores a *display ID* (stable across
    renumbering). At every step the display ID is resolved to the current
    raw ``sample_id`` by sorting samples by their x-centroid.

    Parameters
    ----------
    base_df : pd.DataFrame
        The original dataframe before any splits.
    history : list[dict]
        Ordered list of split records as produced by the UI.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with every recorded split applied and IDs renumbered.
    ids : list[int]
        Sorted list of current sample IDs.
    """
    df = base_df.copy()
    for entry in history:
        centroids  = df[df["sample_id"] != -1].groupby("sample_id")["y"].mean()  # ← x to y
        sorted_ids = centroids.sort_values().index.tolist()
        raw_id     = sorted_ids[entry["sample_id"]]

        df = do_one_split(
            df,
            sample_id    = raw_id,
            split_type   = entry["type"],
            single_value = entry.get("single_value"),
            x_points     = entry.get("x_points"),
            y_points     = entry.get("y_points"),
        )
    return renumber(df)


# ---------------------------------------------------------------------------
# Split-record constructors (keep record creation consistent)
# ---------------------------------------------------------------------------

def make_vertical_record(sample_id: int, x: float) -> SplitRecord:
    """Return a split record for a vertical cut at ``x``."""
    return {"type": "vertical", "sample_id": sample_id, "single_value": x}


def make_horizontal_record(sample_id: int, y: float) -> SplitRecord:
    """Return a split record for a horizontal cut at ``y``."""
    return {"type": "horizontal", "sample_id": sample_id, "single_value": y}


def make_diagonal_record(
    sample_id: int,
    x1: float, y1: float,
    x2: float, y2: float,
) -> SplitRecord:
    """Return a split record for a diagonal cut through two points."""
    return {
        "type": "diagonal",
        "sample_id": sample_id,
        "x_points": [x1, x2],
        "y_points": [y1, y2],
    }


# TODO: Check what you want the user to do (which parameters, default values, etc.)
def run_watershed(path_xenium, output_dir):
    # This performs the first step of the assignment of samples (watershed segmentation)
    df_raw = _load_xenium(path_xenium, output_dir)

    test_labels, test_img, *_ = detect_samples_watershed(df_raw)

    df_test = df_raw.copy()
    df_test["sample_id"] = test_labels
    test_ids = sorted([c for c in np.unique(test_labels) if c != -1])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#1e1e2e")
    axes[0].imshow(test_img, origin="upper", cmap="tab20", aspect="auto")
    axes[0].set_title("Watershed segmentation mask", color="white")
    axes[0].tick_params(colors="grey")
    plot_samples(df_test, test_ids, ax=axes[1])
    plt.tight_layout()
    plt.show(fig)
    return df_test


def manual_split_samples(df_test):
    RESET = True

    if RESET:
        split_history = []
        df_split_base = df_test.copy()
        for var in ["df_test", "test_ids"]:
            if var in globals():
                del globals()[var]


# df = run_watershed()
# manual_split_samples(df)