from ._analysis import (
    dotplot,
    celltype_composition
)

from ._cluster import (
    cluster
)

from ._plot import (
    plot_image,
    rank_genes_group,
    plot_labels,
    plot_count_distr
)

from ._segmentation import (
    multimodal_segmentation,
    multimodal_segmentation_slice
)

from .spatial_clustering import (
    run_banksy,
    run_cellcharter,
    stability_cellcharter
)