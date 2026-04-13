from .plot import (
    dotplot,
    celltype_composition,
    cluster,
    plot_image,
    rank_genes_group,
    multimodal_segmentation,
    multimodal_segmentation_slice,
    plot_labels,
    # banksy
)

# from .plot.spatial_clustering import (
#     banksy
# )

from .assignment import (
    SplitSession,
    LabelSession,
    plot_samples,
    detect_samples_watershed,
    # do_one_split, # discard for now, it is not used I think
    # _renumber,
    replay_all_splits,
    # make_vertical_record,
    # make_horizontal_record,
    # make_diagonal_record,
    run_watershed,
    manual_split_samples
)

from ._spatialobject import (
    read_data,
    convert_sdata_adata
    )
from ._metrics import (
    frac_transcripts,
    staining_positive,
    metrics
    )
from ._qc import (
    plot_outliers,
    perform_quality_control,
    control_probes_codew,
    plot_qc_metrics,
    recommend_threshold,
    is_outlier,
    detect_outlier
)

from ._cellcom import (
    morans_score
)
from ._neighbors import (
    nhood_enrichment
)

from ._pseudobulk import (
    pseudobulk,
    _make_pseudobulk
)

from ._resolution import (
    compute_leiden_resolution
)

from ._preprocess import (
    size_normalization
)