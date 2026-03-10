"""
Interactive Jupyter widget UI for manually splitting spatial samples.

This module is intentionally kept separate from the core logic in
the assignment module.  It handles all ipywidgets / matplotlib concerns and
manages the mutable split-session state (history, pending preview) in a
local object rather than in global variables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Import pure logic from the functional API
from ._assignment_module import (
    replay_all_splits,
    do_one_split,
    renumber,
    make_vertical_record,
    make_horizontal_record,
    make_diagonal_record,
    plot_samples
)

def _draw_cut(ax: plt.Axes, record: dict) -> None:
    """Draw the cut made by the user"""
    stype = record["type"]
    if stype == "vertical":
        ax.axvline(record["single_value"], color="white", lw=2, ls="--", alpha=0.85)
    elif stype == "horizontal":
        ax.axhline(record["single_value"], color="white", lw=2, ls="--", alpha=0.85)
    else:
        x1, x2 = record["x_points"]
        y1, y2 = record["y_points"]
        ax.plot([x1, x2], [y1, y2], color="white", lw=2, ls="--", alpha=0.85)
        ax.scatter([x1, x2], [y1, y2], color="white", s=50, zorder=6)


def _render(
    w_out: widgets.Output,
    df,
    ids: list[int],
    record: dict | None = None,
) -> None:
    """Clear w_out and draw the current segmentation (with optional cut preview)."""
    w_out.clear_output(wait=True)
    with w_out:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1e1e2e")
        plot_samples(df, ids, ax=ax)
        if record is not None:
            _draw_cut(ax, record)
        plt.tight_layout()
        plt.show()
        plt.close(fig)


# Session class: splits samples interactively.
class SplitSession:
    """
    Manages a single interactive splitting session.

    Parameters
    ----------
    base_df : pd.DataFrame
        The original (pre-split) dataframe. Never mutated.

    Notes
    -----
    All mutable state (history, pending preview) lives on this object,
    not in module-level globals. Create a new ``SplitSession`` for each
    dataset you want to work with.
    """
    def __init__(self, base_df):
        self._base_df      = base_df.copy()
        self._history: list[dict] = []
        self._pending      = {"record": None, "df": None, "ids": None}

        # Build initial state
        self._df, self._ids = replay_all_splits(self._base_df, self._history)

    def result(self):
        """
        Return the current segmentation.

        Returns
        -------
        df : pd.DataFrame
            Dataframe with all confirmed splits applied.
        ids : list[int]
            Sorted list of current sample IDs.
        """
        return self._df.copy(), list(self._ids)

    @property
    def history(self) -> list[dict]:
        """Read-only view of the confirmed split history."""
        return list(self._history)

    def show(self) -> None:
        """Build and display the interactive widget in the current Jupyter cell."""
        clear_output(wait=True)
        self._build_ui()

    def _build_ui(self) -> None:
        # input widgets
        w_type = widgets.RadioButtons(
            options=["vertical (x)", "horizontal (y)", "diagonal"],
            value="vertical (x)",
            description="Split type:",
            style={"description_width": "100px"},
        )
        w_sample = widgets.BoundedIntText(
            value=0, min=0, max=500,
            description="Sample ID:",
            style={"description_width": "100px"},
            layout=widgets.Layout(width="220px"),
        )

        w_single_label = widgets.Label("Cut at x =")
        w_single_value = widgets.FloatText(
            value=5000, layout=widgets.Layout(width="150px")
        )
        w_single_row = widgets.HBox([w_single_label, w_single_value])

        w_x1 = widgets.FloatText(value=0,    description="x1 (µm):",
                                  style={"description_width": "80px"},
                                  layout=widgets.Layout(width="190px"))
        w_y1 = widgets.FloatText(value=0,    description="y1 (µm):",
                                  style={"description_width": "80px"},
                                  layout=widgets.Layout(width="190px"))
        w_x2 = widgets.FloatText(value=1000, description="x2 (µm):",
                                  style={"description_width": "80px"},
                                  layout=widgets.Layout(width="190px"))
        w_y2 = widgets.FloatText(value=1000, description="y2 (µm):",
                                  style={"description_width": "80px"},
                                  layout=widgets.Layout(width="190px"))
        w_diag_rows = widgets.VBox([
            widgets.HBox([w_x1, w_y1]),
            widgets.HBox([w_x2, w_y2]),
        ])
        w_diag_rows.layout.display = "none"

        # action buttons
        w_btn_preview = widgets.Button(description="Preview",
                                        button_style="primary",
                                        layout=widgets.Layout(width="120px"))
        w_btn_confirm = widgets.Button(description="✅ Confirm",
                                        button_style="success",
                                        layout=widgets.Layout(width="120px"),
                                        disabled=True)
        w_btn_discard = widgets.Button(description="❌ Discard",
                                        button_style="danger",
                                        layout=widgets.Layout(width="120px"),
                                        disabled=True)
        w_btn_undo    = widgets.Button(description="↩ Undo last",
                                        button_style="warning",
                                        layout=widgets.Layout(width="120px"))

        w_status = widgets.HTML("")
        w_log    = widgets.HTML("")
        w_out    = widgets.Output()

        # helpers (showing history)
        def _update_log():
            if not self._history:
                w_log.value = "<i style='color:grey'>No splits confirmed yet.</i>"
                return
            lines = ["<b>Confirmed splits:</b><br>"]
            for i, s in enumerate(self._history):
                if s["type"] == "vertical":
                    desc = f"S{s['sample_id']} — vertical at x = {s['single_value']:.1f}"
                elif s["type"] == "horizontal":
                    desc = f"S{s['sample_id']} — horizontal at y = {s['single_value']:.1f}"
                else:
                    desc = (f"S{s['sample_id']} — diagonal "
                            f"({s['x_points'][0]:.0f}, {s['y_points'][0]:.0f}) → "
                            f"({s['x_points'][1]:.0f}, {s['y_points'][1]:.0f})")
                lines.append(f"&nbsp;&nbsp;{i+1}. {desc}<br>")
            w_log.value = "".join(lines)

        def _reset_pending():
            for k in self._pending:
                self._pending[k] = None

        # callbacks
        def on_type_change(change):
            if change["name"] != "value":
                return
            is_diag = change["new"] == "diagonal"
            w_diag_rows.layout.display  = "" if is_diag else "none"
            w_single_row.layout.display = "none" if is_diag else ""
            w_single_label.value = (
                "Cut at x =" if change["new"] == "vertical (x)" else "Cut at y ="
            )

        def on_preview(_):
            _reset_pending()
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            w_status.value = ""

            split_type_raw    = w_type.value
            sample_id_display = w_sample.value

            df_current, ids_current = replay_all_splits(self._base_df, self._history)

            if sample_id_display not in ids_current:
                w_status.value = (
                    f"<span style='color:#e74c3c'>"
                    f"Sample {sample_id_display} not found. "
                    f"Available: {ids_current}</span>"
                )
                return

            # build the record (stores display ID for stable history replay)
            if split_type_raw == "vertical (x)":
                record = make_vertical_record(sample_id_display, w_single_value.value)
            elif split_type_raw == "horizontal (y)":
                record = make_horizontal_record(sample_id_display, w_single_value.value)
            else:
                record = make_diagonal_record(
                    sample_id_display,
                    w_x1.value, w_y1.value,
                    w_x2.value, w_y2.value,
                )

            # resolve display ID 
            centroids  = (df_current[df_current["sample_id"] != -1]
                          .groupby("sample_id")["x"].mean())
            sorted_ids = centroids.sort_values().index.tolist()
            raw_id     = sorted_ids[sample_id_display]

            df_after = do_one_split(
                df_current, raw_id, record["type"],
                single_value = record.get("single_value"),
                x_points     = record.get("x_points"),
                y_points     = record.get("y_points"),
            )
            df_after, new_ids = renumber(df_after)

            self._pending["record"] = record
            self._pending["df"]     = df_after
            self._pending["ids"]    = new_ids

            _render(w_out, df_after, new_ids, record=record)
            w_status.value = (
                f"<span style='color:#f39c12'>"
                f"Preview: {len(new_ids)} samples — confirm or discard?</span>"
            )
            w_btn_confirm.disabled = False
            w_btn_discard.disabled = False

        def on_confirm(_):
            if self._pending["record"] is None:
                return

            confirmed_rec = self._pending["record"]
            print("Appending record:", confirmed_rec)       
            self._history.append(confirmed_rec)

            self._df, self._ids = replay_all_splits(self._base_df, self._history)
            print("IDs after replay:", self._ids)           
            print("Unique sample_ids in df:", self._df["sample_id"].unique())

            w_status.value = (
                f"<span style='color:#2ecc71'>"
                f"✅ Confirmed! {len(self._history)} split(s) total, "
                f"{len(self._ids)} samples.</span>"
            )
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            _update_log()
            _render(w_out, self._df, self._ids, record=None)
            _reset_pending()

        def on_discard(_):
            _reset_pending()
            w_btn_confirm.disabled = True
            w_btn_discard.disabled = True
            w_status.value = (
                "<span style='color:#e74c3c'>"
                "❌ Discarded. Adjust values and preview again.</span>"
            )
            df_now, ids_now = replay_all_splits(self._base_df, self._history)
            _render(w_out, df_now, ids_now, record=None)

        def on_undo(_):
            if not self._history:
                w_status.value = "<span style='color:#e74c3c'>Nothing to undo.</span>"
                return
            removed = self._history.pop()
            self._df, self._ids = replay_all_splits(self._base_df, self._history)
            w_status.value = (
                f"<span style='color:#f39c12'>"
                f"↩ Undid: {removed['type']} on S{removed['sample_id']}. "
                f"{len(self._history)} split(s) remaining.</span>"
            )
            _update_log()
            _render(w_out, self._df, self._ids, record=None)

        w_type.observe(on_type_change)
        w_btn_preview.on_click(on_preview)
        w_btn_confirm.on_click(on_confirm)
        w_btn_discard.on_click(on_discard)
        w_btn_undo.on_click(on_undo)

        _update_log()

        display(widgets.VBox([
            widgets.HTML(
                "<b>Manual split tool</b><br>"
                "<small style='color:grey'>Sample IDs in the plot = "
                "what you type in 'Sample ID'.</small>"
            ),
            w_type,
            w_sample,
            w_single_row,
            w_diag_rows,
            widgets.HBox([w_btn_preview, w_btn_confirm, w_btn_discard, w_btn_undo]),
            w_status,
            w_log,
            w_out,
        ]))

        # Initial render
        _render(w_out, self._df, self._ids, record=None)


# Label session: let user assign labels to the samples interactively
class LabelSession:
    """
    Interactive UI for assigning experimental labels to spatial samples.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns ``x``, ``y``, ``sample_id``.
        Typically the result of ``SplitSession.result()``.
    ids : list[int]
        Sample IDs to label. Typically the second return value of
        ``SplitSession.result()``.
    adata : anndata.AnnData, optional
        If provided, ``adata.obs`` will be updated with ``sample_id``
        and ``condition`` columns on export.
    output_dir : str, optional
        Directory to write CSV exports to. Defaults to current directory.
    """

    def __init__(self, df, ids, adata=None, output_dir="."):
        self._df         = df.copy()
        self._ids        = list(ids)
        self._adata      = adata
        self._output_dir = output_dir
        self._conditions = {cid: f"Sample_{cid}" for cid in self._ids}

    def result(self) -> dict:
        """
        Return the current label assignments.

        Returns
        -------
        dict[int, str]
            Mapping of sample ID → condition label.
        """
        return dict(self._conditions)

    def show(self) -> None:
        """Build and display the interactive labelling widget."""
        clear_output(wait=True)
        self._build_ui()

    # Widget construction
    def _build_ui(self) -> None:
        import pandas as pd
        import os

        w_dropdown = widgets.Dropdown(
            options=[
                (f"S{cid}  ({(self._df['sample_id'] == cid).sum():,} cells)", cid)
                for cid in self._ids
            ],
            description="Sample:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="340px"),
        )
        w_condition = widgets.Text(
            value=self._conditions[self._ids[0]],
            placeholder="e.g. treated, control, WT",
            description="Label:",
            style={"description_width": "80px"},
            layout=widgets.Layout(width="320px"),
        )
        w_btn_assign = widgets.Button(
            description="Assign", button_style="primary",
            icon="check", layout=widgets.Layout(width="110px"),
        )
        w_btn_export = widgets.Button(
            description="Save & Export", button_style="success",
            layout=widgets.Layout(width="160px"),
        )
        w_status   = widgets.HTML(value="")
        w_plot_out = widgets.Output()
        w_table_out = widgets.Output()

        # helper functions
        def _refresh_plot():
            w_plot_out.clear_output(wait=True)
            with w_plot_out:
                fig, ax = plt.subplots(figsize=(10, 7))
                fig.patch.set_facecolor("#1e1e2e")
                plot_samples(
                    self._df, self._ids,
                    highlight=w_dropdown.value,
                    conditions=self._conditions,
                    ax=ax,
                )
                plt.tight_layout()
                plt.show()
                plt.close(fig)

        def _refresh_table():
            with w_table_out:
                clear_output(wait=True)
                rows = [
                    (f"S{cid}", self._conditions[cid],
                     f"{(self._df['sample_id'] == cid).sum():,}")
                    for cid in self._ids
                ]
                tbl = pd.DataFrame(rows, columns=["Sample", "Label", "N cells"])
                display(tbl.style.set_properties(**{"text-align": "left"}))

        # callbacks
        def on_dropdown_change(change):
            if change["name"] == "value":
                w_condition.value = self._conditions.get(change["new"], "")
                _refresh_plot()

        def on_assign(_):
            cid = w_dropdown.value
            val = w_condition.value.strip()
            if val:
                self._conditions[cid] = val
                w_status.value = (
                    f"<span style='color:#2ecc71'>S{cid} assigned to '{val}'</span>"
                )
            else:
                w_status.value = (
                    "<span style='color:#e74c3c'>Please enter a label name.</span>"
                )
            _refresh_plot()
            _refresh_table()

        def on_export(_):
            import pandas as pd
            df_out = self._df.copy()
            df_out["condition"] = df_out["sample_id"].map(
                self._conditions).fillna("unassigned")

            # CSV export
            cells_path = os.path.join(self._output_dir, "cells_annotated.csv")
            df_out.to_csv(cells_path, index=False)

            cond_df = pd.DataFrame([
                {"sample_id": k, "condition": v,
                 "n_cells": int((df_out["sample_id"] == k).sum())}
                for k, v in self._conditions.items()
            ])
            cond_path = os.path.join(self._output_dir, "sample_conditions.csv")
            cond_df.to_csv(cond_path, index=False)

            # adata update
            adata_msg = ""
            if self._adata is not None:
                meta_df = df_out[["cell_id", "sample_id", "condition"]]
                self._adata.obs = self._adata.obs.merge(
                    meta_df, on="cell_id", how="left"
                )
                adata_msg = "<br>- adata.obs updated: 'sample_id' and 'condition' columns added"

            w_status.value = (
                f"<span style='color:#2ecc71'>Saved!<br>"
                f"- {cells_path}<br>"
                f"- {cond_path}"
                f"{adata_msg}</span>"
            )
            with w_table_out:
                clear_output(wait=True)
                display(cond_df)

        w_dropdown.observe(on_dropdown_change)
        w_btn_assign.on_click(on_assign)
        w_btn_export.on_click(on_export)

        controls = widgets.VBox([
            widgets.HTML("<b>Label each sample with an experimental label:</b>"),
            widgets.HBox([w_dropdown, w_condition, w_btn_assign]),
            widgets.HBox([w_btn_export, w_status]),
        ])
        display(controls)
        display(widgets.HBox([w_plot_out, w_table_out]))
        _refresh_plot()
        _refresh_table()