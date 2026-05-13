
## 🚀 Quick Start 
A small example of steps that can be taken with SpatioFlow. 

1. Load Xenium data
```py
import SpatioFlow
sdata = SpatioFlow.read_data(path_xenium)
adata = SpatioFlow.tables["table"].copy()
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