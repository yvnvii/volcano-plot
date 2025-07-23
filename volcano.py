"""
Streamlit Proteomics Volcano Plot Web App
----------------------------------------
Workflow:
 1. Upload one or more Excel files
 2. Select/rename columns for Healthy vs PE groups
 3. Merge on Protein.Names (not Genes!)
 4. Clean data (drop 0 / NA rows, optionally drop 1 column/group with most zeros)
 5. Normalize by total intensity per sample
 6. Compute log2 Fold Change and p-values (t-test or Wilcoxon if variance==0)
 7. Draw an interactive volcano plot
 8. Download up/down regulated tables & the cleaned dataframe

Run locally:
    pip install streamlit pandas numpy scipy plotly openpyxl kaleido statsmodels

Author: Yuki Ogawa
Date: July 22 2025
"""

# -------------------------------
# Imports
# -------------------------------
import io
import numpy as np
import pandas as pd
import base64
import streamlit as st
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import plotly.express as px

# -------------------------------
# Utility functions
# -------------------------------

def find_most_zero_na_col(df: pd.DataFrame, cols: list[str]) -> str:
    """Return the column name among `cols` with the highest count of 0 or NA."""
    counts = {c: ((df[c] == 0) | (df[c].isna())).sum() for c in cols}
    return max(counts, key=counts.get)


def normalize_by_total(target_df: pd.DataFrame, cols: list[str], ref_df: pd.DataFrame):
    """Divide target_df[cols] by column totals computed from ref_df[cols]."""
    totals = (
        ref_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=0)
    )
    target_df.loc[:, cols] = target_df[cols].div(totals, axis=1)
    return target_df, totals


def _row_pvalue(row, healthy_cols, pe_cols):
    control = row[healthy_cols].astype(float)
    pe = row[pe_cols].astype(float)

    # Handle constant groups
    if control.std(ddof=1) == 0 and pe.std(ddof=1) == 0:
        return 1.0
    if control.std(ddof=1) == 0 or pe.std(ddof=1) == 0:
        # SciPy >=1.11 supports method=; remove if version lower
        return mannwhitneyu(control, pe, alternative="two-sided").pvalue
    return ttest_ind(control, pe, equal_var=False, nan_policy="omit").pvalue


def compute_stats(df: pd.DataFrame, healthy_cols: list[str], pe_cols: list[str]) -> pd.DataFrame:
    """Add avg_healthy, avg_pe, log2FC, p_value, neg_log10_pval columns to df."""
    df = df.copy()
    df["avg_healthy"] = df[healthy_cols].mean(axis=1)
    df["avg_pe"] = df[pe_cols].mean(axis=1)
    df["log2FC"] = np.log2((df["avg_pe"] + 1e-12) / (df["avg_healthy"] + 1e-12))

    df["p_value"] = df.apply(_row_pvalue, axis=1, args=(healthy_cols, pe_cols))
    df["neg_log10_pval"] = -np.log10(df["p_value"].replace(0, np.nextafter(0, 1)))
    return df


def label_significance(df: pd.DataFrame, fc_threshold: float, pval_threshold: float, use_q=False) -> pd.DataFrame:
    pcol = "q_value" if use_q else "p_value"
    return df.assign(
        Significance=np.select(
            [
                (df["log2FC"] > fc_threshold) & (df[pcol] < pval_threshold),
                (df["log2FC"] < -fc_threshold) & (df[pcol] < pval_threshold),
            ],
            ["Upregulated", "Downregulated"],
            default="Not Significant",
        )
    )


def to_csv_download(df: pd.DataFrame, filename: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

def make_unique(names):
    """['A','B','A','A'] -> ['A','B','A_2','A_3']"""
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)  # 最初はそのまま
    return out


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Proteomics Volcano Plot", layout="wide")
st.title("Proteomics Volcano Plot")

st.markdown(
    """
**Upload your Excel files**, map columns for Healthy and PE groups, and generate an interactive volcano plot.

- All merges are done on **Protein.Names**.
- You can optionally use **Genes** for labeling points.
    """
)

# --- File upload
uploaded = st.file_uploader(
    "Upload one or more Excel files (.xlsx)", type=["xlsx"], accept_multiple_files=True
)

if not uploaded:
    st.info("Please upload at least one Excel file.")
    st.stop()

# Read selected sheet from each file
all_dfs = []
for file in uploaded:
    xls = pd.ExcelFile(file)
    sheet = st.selectbox(
        f"Select sheet for {file.name}", xls.sheet_names, key=f"sheet_{file.name}"
    )
    df_tmp = pd.read_excel(file, sheet_name=sheet)
    df_tmp["__source_file__"] = file.name
    all_dfs.append(df_tmp)

st.success(f"Loaded {len(all_dfs)} dataframes.")

with st.expander("Preview first few rows of each dataframe"):
    for i, d in enumerate(all_dfs, 1):
        st.write(f"**{uploaded[i-1].name}**")
        st.dataframe(d.head())

# --- Column mapping
st.subheader("Column Mapping")
protein_col = st.text_input("Protein column name (for merge)", value="Protein.Names")
gene_col = st.text_input("Gene/Genes column name (optional, used for labels)", value="Genes")

if any(protein_col not in df.columns for df in all_dfs):
    st.error("Protein column not found in at least one uploaded file. Please correct the name.")
    st.stop()

# --- Merge
merged = all_dfs[0]
for nxt in all_dfs[1:]:
    merged = pd.merge(merged, nxt, on=protein_col, how="outer", suffixes=("", "_dup"))

raw_df = merged.copy()  # before normalization

# Resolve duplicates
merged.columns = make_unique(list(merged.columns))

# Cast numeric
for c in merged.columns:
    try:
        merged[c] = pd.to_numeric(merged[c])
    except (ValueError, TypeError):
        pass


st.dataframe(merged, use_container_width=True)

# --- Select sample columns
st.markdown("### Select Sample Columns")
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
healthy_cols = st.multiselect("Healthy sample columns", numeric_cols, default=[])
pe_cols = st.multiselect("PE sample columns", numeric_cols, default=[])

if not healthy_cols or not pe_cols:
    st.warning("Select at least one column for each group.")
    st.stop()

# Optional: drop worst columns
with st.expander("Optional: Drop one column per group with most 0/NA"):
    do_drop_h = st.checkbox("Drop worst Healthy column", value=False)
    do_drop_p = st.checkbox("Drop worst PE column", value=False)

if do_drop_h:
    col_h_rm = find_most_zero_na_col(merged, healthy_cols)
    healthy_cols = [c for c in healthy_cols if c != col_h_rm]
    st.info(f"Removed {col_h_rm} from Healthy group.")

if do_drop_p:
    col_p_rm = find_most_zero_na_col(merged, pe_cols)
    pe_cols = [c for c in pe_cols if c != col_p_rm]
    st.info(f"Removed {col_p_rm} from PE group.")

all_sample_cols = healthy_cols + pe_cols

# --- Data cleaning
st.subheader("Data Cleaning")
remove_zero_na = st.checkbox("Remove rows having any 0 or NA in selected sample columns", value=True)

clean_df = merged.copy()
for c in all_sample_cols:
    clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

if remove_zero_na:
    mask = clean_df[all_sample_cols].apply(lambda r: ((r != 0) & (~r.isna())).all(), axis=1)
    clean_df = clean_df.loc[mask].reset_index(drop=True)
    st.write(f"Rows after removing 0/NA: {clean_df.shape[0]} (from {merged.shape[0]})")

st.dataframe(clean_df, use_container_width=True)

# --- Normalization
st.subheader("Normalization")
normalize = st.checkbox("Normalize to TOTAL intensity (column sums)", value=True)

if normalize:
    clean_df, totals = normalize_by_total(clean_df, all_sample_cols, raw_df)
    with st.expander("Totals used for normalization"):
        st.dataframe(totals.to_frame(name="Total_Intensity"))

# --- Statistics
st.subheader("Statistics")
use_fdr = st.checkbox("Apply BH/FDR correction", value=False)
fc_threshold = st.number_input("log2FC threshold", value=1.0, step=0.1)
pval_threshold = st.number_input("P-value / Q-value threshold", value=0.05, step=0.01, format="%.3f")

stats_df = compute_stats(clean_df, healthy_cols, pe_cols)

if use_fdr:
    stats_df["q_value"] = multipletests(stats_df["p_value"], method="fdr_bh")[1]
    stats_df["neg_log10_qval"] = -np.log10(stats_df["q_value"].replace(0, np.nextafter(0, 1)))

stats_df = label_significance(stats_df, fc_threshold, pval_threshold, use_q=use_fdr)

# --- Volcano plot
st.subheader("Volcano Plot")
label_col = gene_col if gene_col and gene_col in stats_df.columns else protein_col
y_axis = "neg_log10_qval" if use_fdr else "neg_log10_pval"

color_map = {
    "Upregulated": "red",
    "Downregulated": "blue",
    "Not Significant": "gray",
}

fig = px.scatter(
    stats_df,
    x="log2FC",
    y=y_axis,
    color="Significance",
    hover_data={protein_col: True, label_col: True, "p_value": ':.3e', "log2FC": ':.3f'},
    color_discrete_map=color_map,
)

# Threshold lines
fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="black")
fig.add_vline(x=fc_threshold, line_dash="dash", line_color="black")
fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="black")
fig.update_layout(height=650)

st.plotly_chart(fig, use_container_width=True)

# --- Tables & downloads
st.subheader("Results Tables")
cols_to_show = [protein_col, label_col, "avg_pe", "avg_healthy", "log2FC", "p_value", y_axis]
if use_fdr:
    cols_to_show.insert(cols_to_show.index("p_value") + 1, "q_value")

upregulated = stats_df[stats_df["Significance"] == "Upregulated"][cols_to_show]
downregulated = stats_df[stats_df["Significance"] == "Downregulated"][cols_to_show]

st.write("**Upregulated proteins**", upregulated.shape)
st.dataframe(upregulated, use_container_width=True)
to_csv_download(upregulated, "upregulated_proteins.csv")

st.write("**Downregulated proteins**", downregulated.shape)
st.dataframe(downregulated, use_container_width=True)
to_csv_download(downregulated, "downregulated_proteins.csv")

st.write("**Cleaned + stats dataframe**", stats_df.shape)
st.dataframe(stats_df, use_container_width=True)
to_csv_download(stats_df, "all_proteins_with_stats.csv")

# Save figure as PNG
html_bytes = fig.to_html(full_html=False).encode("utf-8")
st.download_button(
    label="Download interactive plot (HTML)",
    data=html_bytes,
    file_name="volcano_plot.html",
    mime="text/html"
)

# --- Sanity check vs R
st.subheader("Sanity Check")
prot = st.text_input("Protein to inspect (exact match)")
if prot:
    row_py = stats_df[stats_df[protein_col] == prot][[
        protein_col, "avg_healthy", "avg_pe", "log2FC", "p_value"
    ] + (["q_value"] if use_fdr else [])]
    st.write("Python:", row_py)

# --- Optional debug block
with st.expander("DEBUG: raw / normalized / totals for one protein"):
    debug_on = st.checkbox("Show debug table", value=False)
    if debug_on and prot:
        final_cols = healthy_cols + pe_cols
        raw_numeric = clean_df[final_cols].apply(pd.to_numeric, errors="coerce")
        totals_dbg = raw_numeric.fillna(0).sum(axis=0)

        row_norm = clean_df.loc[clean_df[protein_col] == prot, final_cols]
        if not row_norm.empty:
            debug_tbl = pd.DataFrame({
                "Sample": final_cols,
                "Normalized": row_norm.T.iloc[:, 0],
                "Total_Intensity": totals_dbg[final_cols].values,
                "Group": ["Healthy" if c in healthy_cols else "PE" for c in final_cols]
            })
            st.dataframe(debug_tbl)
            st.write("Python avg_healthy:", row_norm[healthy_cols].mean(axis=1).iloc[0])
            st.write("Python avg_pe:", row_norm[pe_cols].mean(axis=1).iloc[0])

st.success("Done! Adjust thresholds or selections above to update.")
