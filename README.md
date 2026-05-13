# 📦 SpatioFlow

> *SpatioFlow is a functional API for spatial transcriptomics. Use it wisely.*


[Documentation](https://emmaevonk.github.io/SpatioFlow/) can be found [here](https://emmaevonk.github.io/SpatioFlow/).

## 🌟 Highlights

- End-to-end analyses in one pipeline!
- Recognizing samples from the same Xenium slide
- Assigning labels to samples
- Performing QC, clustering, multimodal segmentation, assigning conditions and pseudobulk DE analysis
- Errorhandler
- Using well known tools in spatial transcriptomics, e.g. Squidpy and Scanpy

## ℹ️ Overview

SpatioFlow is a functional pipeline for analyzing spatial transcriptomics data. It provides a Command Line Interface (CLI) to perform quality control, spatial statistics, neighborhood analysis, and visualization for datasets generated with the Xenium platform from 10X Genomics.

The API is designed around a clean, functional workflow:
```py
SpatioFlow.method()
```

This makes it easy to build reproducible spatial analysis pipelines with minimal boilerplate.


## ✨ Features
SpatioFlow supports a wide range of spatial transcriptomics analyses.

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
SpatioFlow.plot_image()
```

### Metrics & Diagnostics
- Dataset-level summary statistics
- Spatial structure evaluation
- Analysis-ready feature generation

## ⬇️ Installation

Currently, SpatioFlow is not yet available on Pypi. 

Installing from the source:
```bash
git clone https://github.com/emmaevonk/SpatioFlow.git
cd spatialapi
pip install -e .
```

Minimum requirements: look at requirements.txt file 

## 🚀 Quick Start 
A small example of steps that can be taken with SpatioFlow. 

1. Load Xenium data
```py
import SpatioFlow
sdata = SpatioFlow.read_data(path_xenium)
adata = sdata.tables["table"].copy()
``` 
2. Obtain metrics (percentage transcripts and staining positive pixels allocated)
```py
perc_transcripts, s_pos = SpatioFlow.metrics(sdata)
``` 
3. QC
```py
SpatioFlow.plot_qc_metrics(adata, save=True)
thr = SpatioFlow.recommend_threshold(adata)
adata_filtered = SpatioFlow.perform_quality_control(adata, thr)
```
4. Log normalize adata
```py
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)
```
5. Assigning conditions
```py
adata = SpatioFlow.run_watershed(adata, sdata)
df_split_base = df.copy()
session = SpatioFlow.SplitSession(df_split_base)
session.show()
```
6. Perform annotation
7. Analysis
```py
SpatioFlow.dotplot(adata)
SpatioFlow.nhood_enrichment(adata) 
SpatioFlow.celltype_composition(adata)
SpatioFlow.pseudobulk(adata, celltype="Endothelial cells")
```

## 📄 License
MIT license is applied.