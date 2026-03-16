# 💫 SpatialAPI
> *SpatialAPI is a functional API for spatial transcriptomics. Use it wisely.*

## 🌟 Highlights

- End-to-end analyses in one pipeline!
- Recognizing samples from the same Xenium slide
- Assigning labels to samples
- Performing QC, clustering, multimodal segmentation, assigning conditions and pseudobulk DE analysis
- Errorhandler
- Using well known tools in spatial transcriptomics, e.g. Squidpy and Scanpy

## ℹ️ Overview

SpatialAPI is a functional pipeline for analyzing spatial transcriptomics data. It provides a Command Line Interface (CLI) to perform quality control, spatial statistics, neighborhood analysis, and visualization for datasets generated with the Xenium platform from 10X Genomics.

The API is designed around a clean, functional workflow:
```py
spatialapi.method()
```

This makes it easy to build reproducible spatial analysis pipelines with minimal boilerplate.


## ✨ Features
SpatialAPI supports a wide range of spatial transcriptomics analyses.

### Quality control
- Automatic QC metric calculation
- Data-driven filtering recommendations
- Cell- and gene-level diagnostics

### Spatial Analysis
- Neighbor enrichment / cell-cell interaction analysis
- Spatial neighborhood constructuren
- Spatial statistics and summary metrics

### Visualization
- High-resolution spatial plotting
- Overlay gene expression or cell annotations
- Quick visualization with:
```py
spatialapi.plot_image()
```

### Metrics & Diagnostics
- Dataset-level summary statistics
- Spatial structure evaluation
- Analysis-ready feature generation


## 📄 License
MIT license is applied.


