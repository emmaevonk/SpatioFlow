
## 🚀 Quick Start 
A small example of steps that can be taken with SpatialAPI. 

1. Load Xenium data
```py
import spatialapi
sdata = spatialapi.read_data(path_xenium)
adata = sdata.tables["table"].copy()
``` 
2. Obtain metrics (percentage transcripts and staining positive pixels allocated)
```py
perc_transcripts, s_pos = spatialapi.metrics(sdata)
``` 
3. QC
```py
spatialapi.plot_qc_metrics(adata, save=True)
thr = spatialapi.recommend_threshold(adata)
adata_filtered = spatialapi.perform_quality_control(adata, thr)
```
4. Log normalize adata
```py
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)
```
5. Assigning conditions
```py
adata = spatialapi.run_watershed(adata, sdata)
df_split_base = df.copy()
session = SpatialAPI.SplitSession(df_split_base)
session.show()
```
6. Perform annotation
7. Analysis
```py
spatialapi.dotplot(adata)
spatialapi.nhood_enrichment(adata) 
spatialapi.celltype_composition(adata)
spatialapi.pseudobulk(adata, celltype="Endothelial cells")
```