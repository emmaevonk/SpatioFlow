# Assignment
To assign labels to samples correctly, the functions need to be run in this order:

```py
output_dir = "assigning_labels"
df = STAIA.run_watershed(adata=adata, output_dir=output_dir)

# perform manual split
df_split_base = df.copy()
session = SpatialAPI.SplitSession(df_split_base)
session.show()

# get results from splitting
df_results, ids = session.result()

# pass to the labelling session (LabelSession)
label_session = SpatialAPI.LabelSession(df_results, ids, adata=adata, output_dir=output_dir)
label_session.show()

# perform label assignment
conditions = label_session.result()
```

::: STAIA.assignment
    options:
      members:
        - detect_samples_watershed
        - plot_samples
        - run_watershed
      show_root_heading: false
      show_root_toc_entry: false
      filters:
        - "!^_"