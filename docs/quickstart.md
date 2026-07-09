
## 🚀 Quick Start 
A small example of steps that can be taken with STAIA. 

1. Load Xenium data
```py
import STAIA
sdata = STAIA.read_data(path_xenium)
adata = STAIA.tables["table"].copy()
``` 
2. Obtain metrics (percentage transcripts and staining positive pixels allocated)
```py
perc_transcripts, s_pos = STAIA.metrics(sdata)
``` 
3. QC
```py
STAIA.plot_qc_metrics(adata, save=True)
thr = STAIA.recommend_threshold(adata)
adata_filtered = STAIA.perform_quality_control(adata, thr)
```
4. Log normalize adata
```py
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)
```
5. Assigning conditions
```py
adata = STAIA.run_watershed(adata, sdata)
df_split_base = df.copy()
session = STAIA.SplitSession(df_split_base)
session.show()
```
6. Perform annotation
7. Analysis
```py
STAIA.dotplot(adata)
STAIA.nhood_enrichment(adata) 
STAIA.celltype_composition(adata)
STAIA.pseudobulk(adata, celltype="Endothelial cells")
```