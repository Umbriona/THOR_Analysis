import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
from Bio import SeqIO
from glob import glob
from scipy.optimize import curve_fit
import pandas as pd

import sklearn
from scipy.stats import pearsonr, spearmanr

####################################################################################
###################    Summary Thermal Shift Assay helpers    ######################  
####################################################################################


def clean_tm_dataframe(df):
    return (
        df.copy()
        .dropna(subset=["#Annotation", "#Tm"])
        .assign(
            **{
                "#Annotation": lambda x: x["#Annotation"].astype(str).str.strip(),
                "#Tm": lambda x: pd.to_numeric(x["#Tm"], errors="coerce")
            }
        )
        .dropna(subset=["#Tm"])
    )


def summarize_tm_shifts(df):
    df_clean = clean_tm_dataframe(df)

    tm_summary = (
        df_clean.groupby("#Annotation", as_index=False)
        .agg(
            mean_Tm=("#Tm", "mean"),
            std_Tm=("#Tm", "std"),
            n_replicates=("#Tm", "size")
        )
        .sort_values("#Annotation")
        .reset_index(drop=True)
    )

    tm_summary["wildtype_id"] = tm_summary["#Annotation"].str.split("_", n=1).str[0]
    tm_summary["variant_id"] = tm_summary["#Annotation"].str.split("_", n=1).str[1]
    tm_summary["is_wildtype"] = tm_summary["variant_id"].isna()

    wildtype_tm = (
        tm_summary.loc[tm_summary["is_wildtype"], ["#Annotation", "mean_Tm"]]
        .rename(
            columns={
                "#Annotation": "wildtype_id",
                "mean_Tm": "wildtype_mean_Tm"
            }
        )
    )

    tm_summary = tm_summary.merge(wildtype_tm, on="wildtype_id", how="left")
    tm_summary["Tm_difference_vs_wildtype"] = tm_summary["mean_Tm"] - tm_summary["wildtype_mean_Tm"]

    variant_summary = (
        tm_summary.loc[~tm_summary["is_wildtype"]].copy()
        [[
            "wildtype_id",
            "#Annotation",
            "variant_id",
            "mean_Tm",
            "wildtype_mean_Tm",
            "Tm_difference_vs_wildtype",
            "n_replicates"
        ]]
        .rename(columns={
            "#Annotation": "variant_name",
            "mean_Tm": "variant_mean_Tm"
        })
        .sort_values(["wildtype_id", "variant_name"])
        .reset_index(drop=True)
    )

    tm_summary = tm_summary[[
        "#Annotation",
        "wildtype_id",
        "variant_id",
        "is_wildtype",
        "mean_Tm",
        "std_Tm",
        "n_replicates",
        "wildtype_mean_Tm",
        "Tm_difference_vs_wildtype"
    ]]

    return tm_summary, variant_summary


def prepare_variant_shift_plot_df(df, batch_name):
    blank_labels = {"blank", "blanks", "empty", "control"}
    _, variant_summary = summarize_tm_shifts(df)

    return (
        variant_summary.loc[
            ~variant_summary["wildtype_id"].str.lower().isin(blank_labels)
        ]
        .copy()
        .assign(batch=batch_name)
        [[
            "batch",
            "wildtype_id",
            "variant_name",
            "variant_id",
            "variant_mean_Tm",
            "wildtype_mean_Tm",
            "Tm_difference_vs_wildtype",
            "n_replicates"
        ]]
        .sort_values(["wildtype_id", "batch", "variant_name"])
        .reset_index(drop=True)
    )


def mm_to_inches(mm):
    return mm / 25.4


def plot_variant_shift_boxplot(shift_df, split_hues_by_batch=True, size_mm=(60, 60), font_size=5.5, order=None):
    shift_df = shift_df.copy()

    if order is None:
        order = sorted(shift_df["wildtype_id"].dropna().unique())

    rc_params = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size
    }

    wildtype_palette = {
        "MGYP001421927114": "#6ACC64",
        "A0A2S1LEZ1": "#EE854A",
        "A0A372IUB3": "#4878D0"
    }
    batch_palette = {"Batch 3": "#4c72b0", "Batch 4": "#dd8452"}
    box_palette = [wildtype_palette.get(wildtype, "#B0B0B0") for wildtype in order]
    strip_palette = {wildtype: wildtype_palette.get(wildtype, "#4D4D4D") for wildtype in order}

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))

        if split_hues_by_batch:
            sns.boxplot(
                data=shift_df,
                x="wildtype_id",
                y="Tm_difference_vs_wildtype",
                hue="batch",
                order=order,
                showfliers=False,
                palette=batch_palette,
                linewidth=0.6,
                ax=ax
            )
            sns.stripplot(
                data=shift_df,
                x="wildtype_id",
                y="Tm_difference_vs_wildtype",
                hue="batch",
                order=order,
                dodge=True,
                jitter=0.15,
                alpha=0.85,
                size=2.2,
                linewidth=0.3,
                edgecolor="black",
                palette=batch_palette,
                ax=ax
            )

            handles, labels = ax.get_legend_handles_labels()
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_handles.append(handle)
                    unique_labels.append(label)

            ax.legend(
                unique_handles,
                unique_labels,
                title="Batch",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0
            )
        else:
            sns.boxplot(
                data=shift_df,
                x="wildtype_id",
                y="Tm_difference_vs_wildtype",
                order=order,
                showfliers=False,
                palette=box_palette,
                linewidth=0.6,
                ax=ax
            )
            sns.stripplot(
                data=shift_df,
                x="wildtype_id",
                y="Tm_difference_vs_wildtype",
                hue="wildtype_id",
                order=order,
                hue_order=order,
                dodge=False,
                jitter=0.15,
                alpha=0.85,
                size=2.2,
                linewidth=0.3,
                edgecolor="black",
                palette=strip_palette,
                ax=ax
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

        ax.axhline(0, linestyle="--", color="grey", linewidth=0.5)
        ax.set_xlabel("Wildtype")
        ax.set_yticks([-15, -10, -5, 0,5,10,15,20,25,30,35])
        ax.set_ylabel("Mean Tm shift vs wildtype (°C)")
        ax.set_title("Variant thermal shift distributions")
        ax.tick_params(axis="x", rotation=0, length=2, width=0.5)
        ax.tick_params(axis="y", length=2, width=0.5)
        plt.tight_layout(pad=0.4)

    return ax

####################################################################################
#################    Best Variants Thermal Shift Assay helpers    ################## 
####################################################################################

def _normalize_tsa_curve_df(curve_df):
    curve_df = curve_df.copy()
    curve_df["temperature_c"] = pd.to_numeric(curve_df["temperature_c"], errors="coerce")
    curve_df["absorbance"] = pd.to_numeric(curve_df["absorbance"], errors="coerce")
    curve_df = curve_df.dropna(how="any").sort_values("temperature_c").reset_index(drop=True)

    if curve_df.empty:
        curve_df["normalized_absorbance"] = pd.Series(dtype=float)
        return curve_df

    peak_index = curve_df["absorbance"].idxmax()
    pre_peak_df = curve_df.loc[:peak_index].copy()
    norm_min = pre_peak_df["absorbance"].min()
    norm_max = curve_df.loc[peak_index, "absorbance"]

    if pd.isna(norm_min) or pd.isna(norm_max) or norm_max == norm_min:
        curve_df["normalized_absorbance"] = 0.0
    else:
        curve_df["normalized_absorbance"] = (
            (curve_df["absorbance"] - norm_min) / (norm_max - norm_min)
        ).clip(0, 1)

    return curve_df.reset_index(drop=True)


def _read_tsa_curve_file(curve_path):
    curve_path = Path(curve_path)
    header_values = {}
    data_rows = []

    for line in curve_path.read_text().splitlines():
        if not line:
            continue
        if line.startswith("#"):
            if ":" in line:
                key, value = line[1:].split(":", 1)
                header_values[key.strip()] = value.strip()
            continue
        data_rows.append(line.split("\t"))

    curve_df = pd.DataFrame(data_rows, columns=["temperature_c", "absorbance"])
    curve_df = _normalize_tsa_curve_df(curve_df)

    return curve_df, header_values


def load_tsa_curve(curve_dir, well_index):
    curve_path = Path(curve_dir) / f"{well_index}.txt"
    curve_df, _ = _read_tsa_curve_file(curve_path)
    return curve_df.reset_index(drop=True)


def load_tsa_curve_file(curve_path):
    curve_df, header_values = _read_tsa_curve_file(curve_path)
    curve_df["Tm"] = pd.to_numeric(header_values.get("Fitted Tm"), errors="coerce")
    curve_df["R_square"] = pd.to_numeric(header_values.get("R square"), errors="coerce")

    return curve_df.reset_index(drop=True)


def prepare_tsa_curve_data_from_directory(curve_dir, tm_summary_df=None):
    curve_dir = Path(curve_dir)
    curve_paths = sorted(
        path for path in curve_dir.glob("*.txt")
        if "_rep" in path.stem and path.stem.rsplit("_rep", 1)[-1].isdigit()
    )

    if not curve_paths:
        raise ValueError(
            f"No replicate TSA curve files matching '*_repN.txt' were found in {curve_dir}"
        )

    curve_frames = []
    replicate_rows = []

    for curve_path in curve_paths:
        sample_name, replicate_label = curve_path.stem.rsplit("_rep", 1)
        replicate_id = int(replicate_label)
        wildtype_id = sample_name.split("_", 1)[0]
        sample_type = "Wildtype" if sample_name == wildtype_id else "Best variant"

        curve_df = load_tsa_curve_file(curve_path)
        curve_tm = pd.to_numeric(curve_df["Tm"], errors="coerce").dropna()
        curve_r_square = pd.to_numeric(curve_df["R_square"], errors="coerce").dropna()

        curve_df["wildtype_id"] = wildtype_id
        curve_df["variant_name"] = sample_name if sample_type == "Best variant" else np.nan
        curve_df["sample_name"] = sample_name
        curve_df["sample_type"] = sample_type
        curve_df["replicate_id"] = replicate_id
        curve_df["curve_file"] = curve_path.name
        curve_frames.append(curve_df)

        replicate_rows.append(
            {
                "wildtype_id": wildtype_id,
                "variant_name": sample_name if sample_type == "Best variant" else np.nan,
                "sample_name": sample_name,
                "sample_type": sample_type,
                "replicate_id": replicate_id,
                "curve_file": curve_path.name,
                "Tm": float(curve_tm.iloc[0]) if not curve_tm.empty else np.nan,
                "R_square": float(curve_r_square.iloc[0]) if not curve_r_square.empty else np.nan
            }
        )

    curve_data = pd.concat(curve_frames, ignore_index=True)
    replicate_summary = pd.DataFrame(replicate_rows)

    sample_type_order = {"Wildtype": 0, "Best variant": 1}
    replicate_summary["sample_type_order"] = (
        replicate_summary["sample_type"].map(sample_type_order).fillna(99).astype(int)
    )
    replicate_summary = (
        replicate_summary.sort_values(
            ["wildtype_id", "sample_type_order", "sample_name", "replicate_id"]
        )
        .drop(columns=["sample_type_order"])
        .reset_index(drop=True)
    )

    sample_summary = (
        replicate_summary.groupby(["wildtype_id", "sample_name", "sample_type"], as_index=False)
        .agg(
            curve_replicates=("replicate_id", "nunique"),
            curve_mean_Tm=("Tm", "mean"),
            curve_median_Tm=("Tm", "median"),
            curve_min_Tm=("Tm", "min"),
            curve_max_Tm=("Tm", "max")
        )
    )

    wildtype_summary = sample_summary.loc[sample_summary["sample_type"] == "Wildtype"].copy()
    variant_summary = sample_summary.loc[sample_summary["sample_type"] == "Best variant"].copy()

    if wildtype_summary.empty or variant_summary.empty:
        raise ValueError(
            "Expected both wildtype and variant replicate files in the curve directory."
        )

    wildtype_summary = wildtype_summary.rename(
        columns={
            "sample_name": "wildtype_name",
            "curve_replicates": "wildtype_curve_replicates",
            "curve_mean_Tm": "wildtype_curve_mean_Tm",
            "curve_median_Tm": "wildtype_curve_median_Tm",
            "curve_min_Tm": "wildtype_curve_min_Tm",
            "curve_max_Tm": "wildtype_curve_max_Tm"
        }
    )
    variant_summary = variant_summary.rename(
        columns={
            "sample_name": "variant_name",
            "curve_replicates": "variant_curve_replicates",
            "curve_mean_Tm": "variant_curve_mean_Tm",
            "curve_median_Tm": "variant_curve_median_Tm",
            "curve_min_Tm": "variant_curve_min_Tm",
            "curve_max_Tm": "variant_curve_max_Tm"
        }
    )

    curve_pairs = variant_summary.merge(
        wildtype_summary[
            [
                "wildtype_id",
                "wildtype_name",
                "wildtype_curve_replicates",
                "wildtype_curve_mean_Tm",
                "wildtype_curve_median_Tm",
                "wildtype_curve_min_Tm",
                "wildtype_curve_max_Tm"
            ]
        ],
        on="wildtype_id",
        how="left"
    )

    missing_wildtypes = curve_pairs.loc[curve_pairs["wildtype_name"].isna(), "wildtype_id"].tolist()
    if missing_wildtypes:
        raise ValueError(
            "Missing matching wildtype replicate files for: "
            + ", ".join(sorted(set(missing_wildtypes)))
        )

    if tm_summary_df is not None:
        _, tm_variant_summary = summarize_tm_shifts(tm_summary_df[["#Annotation", "#Tm"]])
        curve_pairs = curve_pairs.merge(
            tm_variant_summary[
                [
                    "wildtype_id",
                    "variant_name",
                    "variant_mean_Tm",
                    "wildtype_mean_Tm",
                    "Tm_difference_vs_wildtype",
                    "n_replicates"
                ]
            ],
            on=["wildtype_id", "variant_name"],
            how="left"
        )
    else:
        curve_pairs["variant_mean_Tm"] = curve_pairs["variant_curve_mean_Tm"]
        curve_pairs["wildtype_mean_Tm"] = curve_pairs["wildtype_curve_mean_Tm"]
        curve_pairs["Tm_difference_vs_wildtype"] = (
            curve_pairs["variant_mean_Tm"] - curve_pairs["wildtype_mean_Tm"]
        )
        curve_pairs["n_replicates"] = curve_pairs["variant_curve_replicates"]

    curve_pairs = curve_pairs.sort_values(["wildtype_id", "variant_name"]).reset_index(drop=True)

    return curve_pairs, replicate_summary, curve_data


def plot_tsa_replicate_bands(
    curve_data,
    curve_pairs=None,
    size_mm=(120, 60),
    font_size=5.5,
    order=None,
    title=None,
    show_tm_markers=True
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if order is None:
        if curve_pairs is not None and not curve_pairs.empty:
            order = curve_pairs["wildtype_id"].drop_duplicates().tolist()
        else:
            order = curve_data["wildtype_id"].dropna().drop_duplicates().tolist()

    rc_params = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size
    }
    wildtype_palette = {
        "MGYP001421927114": "#6ACC64",
        "A0A2S1LEZ1": "#EE854A",
        "A0A372IUB3": "#4878D0"
    }
    line_styles = {"Wildtype": "-", "Best variant": "--"}
    band_alphas = {"Wildtype": 0.12, "Best variant": 0.22}
    wildtype_order = {wildtype_id: index for index, wildtype_id in enumerate(order)}

    sample_summary = (
        curve_data[["wildtype_id", "sample_name", "sample_type"]]
        .drop_duplicates()
        .assign(
            wildtype_order=lambda x: x["wildtype_id"].map(wildtype_order).fillna(len(order)),
            sample_type_order=lambda x: x["sample_type"].map({"Wildtype": 0, "Best variant": 1}).fillna(99)
        )
        .sort_values(["wildtype_order", "sample_type_order", "sample_name"])
        .reset_index(drop=True)
    )
    sample_tm_summary = (
        curve_data[["wildtype_id", "sample_name", "sample_type", "replicate_id", "Tm"]]
        .dropna(subset=["Tm"])
        .drop_duplicates(["wildtype_id", "sample_name", "sample_type", "replicate_id"])
        .groupby(["wildtype_id", "sample_name", "sample_type"], as_index=False)
        .agg(median_Tm=("Tm", "median"))
    )
    sample_summary = sample_summary.merge(
        sample_tm_summary,
        on=["wildtype_id", "sample_name", "sample_type"],
        how="left"
    )

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))

        for _, sample_row in sample_summary.iterrows():
            subset = curve_data.loc[
                (curve_data["wildtype_id"] == sample_row["wildtype_id"])
                & (curve_data["sample_name"] == sample_row["sample_name"])
            ]
            if subset.empty:
                continue

            curve_band = (
                subset.groupby("temperature_c", as_index=False)
                .agg(
                    min_absorbance=("normalized_absorbance", "min"),
                    median_absorbance=("normalized_absorbance", "median"),
                    max_absorbance=("normalized_absorbance", "max")
                )
                .sort_values("temperature_c")
            )

            color = wildtype_palette.get(sample_row["wildtype_id"], "#666666")
            sample_type = sample_row["sample_type"]

            ax.fill_between(
                curve_band["temperature_c"],
                curve_band["min_absorbance"],
                curve_band["max_absorbance"],
                color=color,
                alpha=band_alphas.get(sample_type, 0.18),
                linewidth=0
            )
            ax.plot(
                curve_band["temperature_c"],
                curve_band["median_absorbance"],
                color=color,
                linestyle=line_styles.get(sample_type, "-"),
                linewidth=1.1,
                alpha=0.98
            )

            if show_tm_markers and pd.notna(sample_row["median_Tm"]):
                tm_curve_y = float(
                    np.interp(
                        float(sample_row["median_Tm"]),
                        curve_band["temperature_c"].to_numpy(dtype=float),
                        curve_band["median_absorbance"].to_numpy(dtype=float)
                    )
                )
                ax.scatter(
                    [sample_row["median_Tm"]],
                    [tm_curve_y],
                    color=color,
                    s=18,
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=1.0,
                    zorder=4
                )

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Normalized absorbance")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([30, 40, 50, 60, 70, 80, 90, 100])
        ax.tick_params(axis="x", length=2, width=0.5)
        ax.tick_params(axis="y", length=2, width=0.5)
        if title is not None:
            ax.set_title(title)

        legend_handles = []
        for wildtype_id in order:
            wildtype_subset = sample_summary.loc[
                (sample_summary["wildtype_id"] == wildtype_id)
                & (sample_summary["sample_type"] == "Wildtype")
            ]
            variant_subset = sample_summary.loc[
                (sample_summary["wildtype_id"] == wildtype_id)
                & (sample_summary["sample_type"] == "Best variant")
            ]
            color = wildtype_palette.get(wildtype_id, "#666666")

            if not wildtype_subset.empty:
                legend_handles.append(
                    Line2D([0], [0], color=color, linestyle="-", linewidth=1.1, label=f"{wildtype_id} WT")
                )
            for _, variant_row in variant_subset.iterrows():
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="--",
                        linewidth=1.1,
                        label=variant_row["sample_name"]
                    )
                )

        legend_handles.append(
            Patch(facecolor="#666666", edgecolor="none", alpha=0.18, label="Replicate range")
        )
        if show_tm_markers:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#666666",
                    marker="o",
                    markerfacecolor="#666666",
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    markersize=4,
                    linewidth=0,
                    label="Median Tm"
                )
            )
        ax.legend(
            handles=legend_handles,
            frameon=False,
            ncol=1,
            loc="upper left",
            #bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )
        plt.tight_layout(pad=0.4)

    return ax


####################################################################################
########################    Activity resistance analysis    ######################## 
####################################################################################

def activity_sample_sort_key(sample_name):
    sample_name = str(sample_name).strip()
    if sample_name == "NC":
        return ("zzzz", 1, sample_name)

    wildtype_id = sample_name.split("_", 1)[0]
    is_variant = "_" in sample_name

    return (wildtype_id, 1 if is_variant else 0, sample_name)


def compute_activity_rate(row, time_columns, fit_until_s=540):
    fit_time_columns = [column for column in time_columns if int(column) <= fit_until_s]
    x_values = np.array([int(column) for column in fit_time_columns], dtype=float)
    y_values = row[fit_time_columns].astype(float).to_numpy()

    if len(x_values) < 2 or np.isnan(y_values).any():
        return np.nan

    slope = np.polyfit(x_values, y_values, 1)[0]

    return -slope


def compute_minimum_signal(row, time_columns):
    signal_values = pd.to_numeric(row[time_columns], errors="coerce").to_numpy(dtype=float)

    if len(signal_values) == 0 or np.isnan(signal_values).all():
        return np.nan

    return float(np.nanmin(signal_values))


def load_activity_temperature_file(csv_path, fit_until_s=540, remove_inactive=True):
    df = pd.read_csv(csv_path)
    time_columns = sorted([column for column in df.columns if str(column).isdigit()], key=int)

    df["sample_name"] = df["Sample name"].astype(str).str.strip()
    df["wildtype"] = df.apply(lambda x: x["sample_name"].split("_")[0], axis=1)
    if remove_inactive:
        df = df.loc[~df["wildtype"].isin(["MGYP000212420431", "MGYP001376212606"])]
    df["temperature_c"] = pd.to_numeric(df["Temperature"], errors="coerce")
    df["activity_rate"] = df.apply(
        compute_activity_rate,
        axis=1,
        time_columns=time_columns,
        fit_until_s=fit_until_s
    )
    df["minimum_signal"] = df.apply(compute_minimum_signal, axis=1, time_columns=time_columns)

    nc_rate = df.loc[df["sample_name"].eq("NC"), "activity_rate"].mean()
    if pd.isna(nc_rate):
        nc_rate = 0.0

    nc_signal_values = (
        df.loc[df["sample_name"].eq("NC"), time_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if nc_signal_values.size > 0 and not np.isnan(nc_signal_values).all():
        nc_signal_reference = float(np.nanmean(nc_signal_values))
    else:
        sample_signal_values = (
            df.loc[~df["sample_name"].eq("NC"), time_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        nc_signal_reference = (
            float(np.nanpercentile(sample_signal_values, 95))
            if sample_signal_values.size > 0 and not np.isnan(sample_signal_values).all()
            else np.nan
        )

    df["activity_rate_corrected"] = (df["activity_rate"] - nc_rate).clip(lower=0)
    df["nc_signal_reference"] = nc_signal_reference

    return df[[
        "Well",
        "sample_name",
        "temperature_c",
        "activity_rate",
        "activity_rate_corrected",
        "minimum_signal",
        "nc_signal_reference"
    ]].copy()


def compile_activity_resistance(activity_dir, fit_until_s=540, reference_mode="max", reference_temperature=30, catalyzed_signal_floor_quantile=0.05):
    activity_dir = os.path.abspath(activity_dir)
    csv_paths = sorted(
        glob(os.path.join(activity_dir, "Activity_assay_temperature_*.csv")),
        key=lambda path: int(os.path.splitext(path)[0].split("_")[-1])
    )

    replicate_frames = [load_activity_temperature_file(path, fit_until_s=fit_until_s) for path in csv_paths]
    activity_replicates = pd.concat(replicate_frames, ignore_index=True)
    activity_replicates = activity_replicates.loc[~activity_replicates["sample_name"].eq("NC")].copy()

    catalyzed_signal_floor = activity_replicates["minimum_signal"].quantile(catalyzed_signal_floor_quantile)
    if pd.isna(catalyzed_signal_floor):
        catalyzed_signal_floor = activity_replicates["minimum_signal"].min()

    activity_replicates["fraction_substrate_catalyzed"] = np.where(
        activity_replicates["nc_signal_reference"] > catalyzed_signal_floor,
        (activity_replicates["nc_signal_reference"] - activity_replicates["minimum_signal"])
        / (activity_replicates["nc_signal_reference"] - catalyzed_signal_floor),
        np.nan
    )
    activity_replicates["fraction_substrate_catalyzed"] = activity_replicates["fraction_substrate_catalyzed"].clip(lower=0, upper=1)

    activity_summary = (
        activity_replicates.groupby(["sample_name", "temperature_c"], as_index=False)
        .agg(
            mean_activity=("activity_rate_corrected", "mean"),
            sd_activity=("activity_rate_corrected", "std"),
            n_replicates=("activity_rate_corrected", "size"),
            mean_minimum_signal=("minimum_signal", "mean"),
            mean_nc_signal_reference=("nc_signal_reference", "mean"),
            mean_fraction_substrate_catalyzed=("fraction_substrate_catalyzed", "mean"),
            sd_fraction_substrate_catalyzed=("fraction_substrate_catalyzed", "std")
        )
    )
    activity_summary["wildtype_id"] = activity_summary["sample_name"].str.split("_", n=1).str[0]

    if reference_mode == "temperature":
        reference_activity = (
            activity_summary.loc[activity_summary["temperature_c"] == reference_temperature, ["sample_name", "mean_activity"]]
            .rename(columns={"mean_activity": "reference_activity"})
        )
    elif reference_mode == "max":
        reference_activity = (
            activity_summary.groupby("sample_name", as_index=False)["mean_activity"]
            .max()
            .rename(columns={"mean_activity": "reference_activity"})
        )
    else:
        raise ValueError("reference_mode must be 'max' or 'temperature'")

    activity_summary = activity_summary.merge(reference_activity, on="sample_name", how="left")
    activity_summary["percent_active"] = np.where(
        activity_summary["reference_activity"] > 0,
        100 * activity_summary["mean_activity"] / activity_summary["reference_activity"],
        np.nan
    )
    activity_summary["percent_active"] = activity_summary["percent_active"].clip(lower=0)
    activity_summary["percent_substrate_catalyzed"] = 100 * activity_summary["mean_fraction_substrate_catalyzed"]
    activity_summary["catalyzed_signal_floor"] = float(catalyzed_signal_floor)

    sample_order = sorted(activity_summary["sample_name"].unique(), key=activity_sample_sort_key)
    temperature_order = sorted(activity_summary["temperature_c"].dropna().astype(int).unique())
    activity_percent_pivot = (
        activity_summary.pivot(index="sample_name", columns="temperature_c", values="percent_active")
        .reindex(index=sample_order, columns=temperature_order)
    )

    return activity_replicates, activity_summary, activity_percent_pivot


def plot_activity_resistance_heatmap(activity_percent_pivot, size_mm=(110, 150), font_size=5.5, vmin=0, vmax=100):
    rc_params = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size
    }

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))
        sns.heatmap(
            activity_percent_pivot,
            cmap="crest",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"label": "Relative activity (%)"},
            ax=ax
        )

        sample_order = list(activity_percent_pivot.index)
        wildtype_groups = [sample_name.split("_", 1)[0] for sample_name in sample_order]
        for idx in range(1, len(wildtype_groups)):
            if wildtype_groups[idx] != wildtype_groups[idx - 1]:
                ax.hlines(idx, *ax.get_xlim(), colors="black", linewidth=0.6)

        ax.set_xlabel("Temperature treatment (°C)")
        ax.set_ylabel("Sample")
        ax.set_title("Relative activity after 10 min heat treatment")
        ax.tick_params(axis="x", rotation=0, length=0)
        ax.tick_params(axis="y", length=0)
        plt.tight_layout(pad=0.4)

    return ax


def summarize_variant_retention_by_wildtype(
    activity_summary,
    retained_activity_threshold=50,
    include_wildtypes=False,
    activity_metric_column="percent_active",
    retained_flag_column="retained_activity",
    threshold_column_name="retained_activity_threshold"
):
    retention_df = activity_summary.copy()

    if not include_wildtypes:
        retention_df = retention_df.loc[
            retention_df["sample_name"] != retention_df["wildtype_id"]
        ].copy()

    retention_df = retention_df.dropna(subset=[activity_metric_column]).copy()
    retention_df[retained_flag_column] = retention_df[activity_metric_column] >= retained_activity_threshold

    retention_summary = (
        retention_df.groupby(["wildtype_id", "temperature_c"], as_index=False)
        .agg(
            n_variants_measured=("sample_name", "nunique"),
            n_variants_retained=(retained_flag_column, "sum")
        )
    )
    retention_summary["percent_variants_retained"] = np.where(
        retention_summary["n_variants_measured"] > 0,
        100 * retention_summary["n_variants_retained"] / retention_summary["n_variants_measured"],
        np.nan
    )
    retention_summary[threshold_column_name] = float(retained_activity_threshold)

    return retention_summary.sort_values(["wildtype_id", "temperature_c"]).reset_index(drop=True)


def summarize_wildtype_threshold_loss(activity_summary, retained_activity_threshold=50, minimum_temperature_c=None):
    wildtype_activity = activity_summary.loc[
        activity_summary["sample_name"] == activity_summary["wildtype_id"],
        [
            "wildtype_id",
            "temperature_c",
            "mean_activity",
            "sd_activity",
            "n_replicates",
            "percent_active"
        ]
    ].dropna(subset=["temperature_c", "percent_active"]).copy()

    if minimum_temperature_c is not None:
        wildtype_activity = wildtype_activity.loc[
            wildtype_activity["temperature_c"] >= minimum_temperature_c
        ].copy()

    summary_rows = []
    for wildtype_id, group_df in wildtype_activity.groupby("wildtype_id", sort=True):
        group_df = group_df.sort_values("temperature_c").reset_index(drop=True)
        retained_df = group_df.loc[group_df["percent_active"] >= retained_activity_threshold]
        loss_df = group_df.loc[group_df["percent_active"] < retained_activity_threshold]

        last_retained_row = retained_df.iloc[-1] if not retained_df.empty else None
        first_loss_row = loss_df.iloc[0] if not loss_df.empty else None

        summary_rows.append({
            "wildtype_id": wildtype_id,
            "last_temperature_retained_c": np.nan if last_retained_row is None else float(last_retained_row["temperature_c"]),
            "percent_active_at_last_retained": np.nan if last_retained_row is None else float(last_retained_row["percent_active"]),
            "first_temperature_below_threshold_c": np.nan if first_loss_row is None else float(first_loss_row["temperature_c"]),
            "percent_active_at_first_loss": np.nan if first_loss_row is None else float(first_loss_row["percent_active"]),
            "retained_activity_threshold": float(retained_activity_threshold),
            "evaluated_from_temperature_c": np.nan if minimum_temperature_c is None else float(minimum_temperature_c),
            "status": (
                "retained through highest tested temperature"
                if first_loss_row is None
                else "first tested temperature below threshold shown"
            )
        })

    return pd.DataFrame(summary_rows).sort_values("wildtype_id").reset_index(drop=True)


def summarize_wildtype_conversion_loss(activity_summary, substrate_conversion_threshold=50, minimum_temperature_c=None):
    wildtype_activity = activity_summary.loc[
        activity_summary["sample_name"] == activity_summary["wildtype_id"],
        [
            "wildtype_id",
            "temperature_c",
            "mean_minimum_signal",
            "mean_nc_signal_reference",
            "percent_substrate_catalyzed",
            "catalyzed_signal_floor",
            "n_replicates"
        ]
    ].dropna(subset=["temperature_c", "percent_substrate_catalyzed"]).copy()

    if minimum_temperature_c is not None:
        wildtype_activity = wildtype_activity.loc[
            wildtype_activity["temperature_c"] >= minimum_temperature_c
        ].copy()

    summary_rows = []
    for wildtype_id, group_df in wildtype_activity.groupby("wildtype_id", sort=True):
        group_df = group_df.sort_values("temperature_c").reset_index(drop=True)
        retained_df = group_df.loc[group_df["percent_substrate_catalyzed"] >= substrate_conversion_threshold]
        loss_df = group_df.loc[group_df["percent_substrate_catalyzed"] < substrate_conversion_threshold]

        last_retained_row = retained_df.iloc[-1] if not retained_df.empty else None
        first_loss_row = loss_df.iloc[0] if not loss_df.empty else None

        summary_rows.append({
            "wildtype_id": wildtype_id,
            "last_temperature_meeting_conversion_threshold_c": np.nan if last_retained_row is None else float(last_retained_row["temperature_c"]),
            "percent_substrate_catalyzed_at_last_retained": np.nan if last_retained_row is None else float(last_retained_row["percent_substrate_catalyzed"]),
            "first_temperature_below_conversion_threshold_c": np.nan if first_loss_row is None else float(first_loss_row["temperature_c"]),
            "percent_substrate_catalyzed_at_first_loss": np.nan if first_loss_row is None else float(first_loss_row["percent_substrate_catalyzed"]),
            "substrate_conversion_threshold": float(substrate_conversion_threshold),
            "evaluated_from_temperature_c": np.nan if minimum_temperature_c is None else float(minimum_temperature_c),
            "status": (
                "meets threshold through highest tested temperature"
                if first_loss_row is None
                else "first tested temperature below threshold shown"
            )
        })

    return pd.DataFrame(summary_rows).sort_values("wildtype_id").reset_index(drop=True)


def plot_variant_retention_by_wildtype(retention_summary, retained_activity_threshold=50, size_mm=(110, 70), font_size=5.5):
    rc_params = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size
    }
    wildtype_palette = {
        "MGYP001421927114": "#6ACC64",
        "A0A2S1LEZ1": "#EE854A",
        "A0A372IUB3": "#4878D0"
    }
    wildtype_order = sorted(retention_summary["wildtype_id"].dropna().unique())
    fallback_colors = sns.color_palette("tab10", n_colors=max(len(wildtype_order), 3)).as_hex()
    palette = {}
    fallback_idx = 0
    for wildtype_id in wildtype_order:
        if wildtype_id in wildtype_palette:
            palette[wildtype_id] = wildtype_palette[wildtype_id]
        else:
            while fallback_colors[fallback_idx] in wildtype_palette.values():
                fallback_idx += 1
            palette[wildtype_id] = fallback_colors[fallback_idx]
            fallback_idx += 1

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))
        sns.lineplot(
            data=retention_summary,
            x="temperature_c",
            y="percent_variants_retained",
            hue="wildtype_id",
            style="wildtype_id",
            hue_order=wildtype_order,
            markers=True,
            dashes=False,
            linewidth=1.0,
            palette=palette,
            ax=ax
        )

        ax.set_xlabel("Temperature treatment (°C)")
        ax.set_ylabel("Variants retained (%)")
        ax.set_title(f"Variants retaining at least {retained_activity_threshold}% relative activity")
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", rotation=0, length=2, width=0.5)
        ax.tick_params(axis="y", length=2, width=0.5)
        ax.legend(title="Wildtype family", frameon=False, loc="best")
        plt.tight_layout(pad=0.4)

    return ax


def plot_variant_retention_bar_by_wildtype(
    retention_summary,
    retained_activity_threshold=50,
    size_mm=(140, 70),
    font_size=5.5,
    retained_quantity_label="relative activity"
):
    rc_params = {
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size
    }
    wildtype_order = sorted(retention_summary["wildtype_id"].dropna().unique())
    temperature_order = sorted(retention_summary["temperature_c"].dropna().astype(int).unique())
    retention_plot_df = retention_summary.copy()
    retention_plot_df = (
        retention_plot_df.set_index(["wildtype_id", "temperature_c"])
        .reindex(pd.MultiIndex.from_product([wildtype_order, temperature_order], names=["wildtype_id", "temperature_c"]))
        .reset_index()
    )
    retention_plot_df["n_variants_measured"] = retention_plot_df["n_variants_measured"].fillna(0)
    retention_plot_df["n_variants_retained"] = retention_plot_df["n_variants_retained"].fillna(0)
    retention_plot_df["percent_variants_retained"] = retention_plot_df["percent_variants_retained"].fillna(0)
    retention_plot_df["temperature_label"] = retention_plot_df["temperature_c"].astype(int).astype(str)
    temperature_label_order = [str(temp) for temp in temperature_order]
    temperature_palette = dict(
        zip(temperature_label_order, sns.color_palette("magma", n_colors=len(temperature_label_order)).as_hex())
    )

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))
        x_positions = np.arange(len(wildtype_order), dtype=float)
        total_group_width = 0.84
        bar_width = total_group_width / max(len(temperature_order), 1)
        zero_bar_offset = 1.0

        for temperature_idx, temperature_c in enumerate(temperature_order):
            temp_label = str(temperature_c)
            temp_df = (
                retention_plot_df.loc[retention_plot_df["temperature_c"] == temperature_c]
                .set_index("wildtype_id")
                .reindex(wildtype_order)
                .reset_index()
            )
            bar_positions = (
                x_positions
                - total_group_width / 2
                + bar_width / 2
                + temperature_idx * bar_width
            )
            bar_heights = temp_df["percent_variants_retained"].to_numpy(dtype=float) + zero_bar_offset

            ax.bar(
                bar_positions,
                bar_heights,
                width=bar_width,
                bottom=-zero_bar_offset,
                color=temperature_palette[temp_label],
                edgecolor="white",
                linewidth=0.3,
                label=temp_label
            )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(wildtype_order)

        ax.set_xlabel("Wildtype family")
        ax.set_ylabel("Variants meeting threshold (%)")
        ax.set_title(f"Variants catalyzing at least {retained_activity_threshold}% {retained_quantity_label}")
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", rotation=20, length=2, width=0.5)
        ax.tick_params(axis="y", length=2, width=0.5)
        ax.legend(title="Temperature (°C)", frameon=False, loc="best")
        plt.tight_layout(pad=0.4)

    return ax


####################################################################################
########################         Kinetics analysis          ######################## 
####################################################################################

from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


MM_TO_INCH = 1 / 25.4
EXCEL_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "package": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def mm_to_inches_KINETICS(size_mm):
    return tuple(value * MM_TO_INCH for value in size_mm)


def excel_column_to_index(column_letters):
    index = 0
    for character in column_letters:
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    return index - 1


def parse_excel_cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    value_node = cell.find("main:v", EXCEL_NS)

    if cell_type == "inlineStr":
        text_nodes = cell.findall(".//main:t", EXCEL_NS)
        return "".join(node.text or "" for node in text_nodes)

    if value_node is None or value_node.text is None:
        return np.nan

    value = value_node.text
    if cell_type == "s":
        return shared_strings[int(value)]
    if cell_type == "b":
        return value == "1"
    return value


def get_worksheet_path(zip_file, sheet_name=None):
    workbook_root = ET.fromstring(zip_file.read("xl/workbook.xml"))
    workbook_rels_root = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))

    relationship_lookup = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in workbook_rels_root.findall("package:Relationship", EXCEL_NS)
    }

    sheet_nodes = workbook_root.findall("main:sheets/main:sheet", EXCEL_NS)
    if not sheet_nodes:
        raise ValueError("No worksheets were found in the Excel file.")

    selected_sheet = None
    if sheet_name is None:
        selected_sheet = sheet_nodes[0]
    else:
        for sheet in sheet_nodes:
            if sheet.attrib.get("name") == sheet_name:
                selected_sheet = sheet
                break

    if selected_sheet is None:
        available_names = [sheet.attrib.get("name", "") for sheet in sheet_nodes]
        raise ValueError(f"Worksheet '{sheet_name}' was not found. Available worksheets: {available_names}")

    relationship_id = selected_sheet.attrib[f"{{{EXCEL_NS['office']}}}id"]
    target = relationship_lookup[relationship_id].lstrip("/")
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def load_xlsx_sheet_without_openpyxl(xlsx_path, sheet_name=None):
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Could not find Excel file: {xlsx_path}")

    with ZipFile(xlsx_path) as zip_file:
        shared_strings = []
        if "xl/sharedStrings.xml" in zip_file.namelist():
            shared_root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", EXCEL_NS):
                text_nodes = item.findall(".//main:t", EXCEL_NS)
                shared_strings.append("".join(node.text or "" for node in text_nodes))

        worksheet_path = get_worksheet_path(zip_file, sheet_name=sheet_name)
        worksheet_root = ET.fromstring(zip_file.read(worksheet_path))

    rows = worksheet_root.findall(".//main:sheetData/main:row", EXCEL_NS)
    if not rows:
        return pd.DataFrame()

    header = None
    records = []
    for row in rows:
        row_values = {}
        for cell in row.findall("main:c", EXCEL_NS):
            reference = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", reference)
            if match is None:
                continue
            column_index = excel_column_to_index(match.group(1))
            row_values[column_index] = parse_excel_cell_value(cell, shared_strings)

        if not row_values:
            continue

        if header is None:
            max_index = max(row_values)
            header = []
            for index in range(max_index + 1):
                value = row_values.get(index, f"Column_{index + 1}")
                header.append(str(value).strip() or f"Column_{index + 1}")
            continue

        record = {}
        for index, column_name in enumerate(header):
            value = row_values.get(index, np.nan)
            if value == "":
                value = np.nan
            record[column_name] = value

        if any(pd.notna(value) for value in record.values()):
            records.append(record)

    return pd.DataFrame(records)


def load_batch5_kinetics(xlsx_path, epsilon=6220.0, pathlength_cm=0.45, sheet_name=None):
    df = load_xlsx_sheet_without_openpyxl(xlsx_path, sheet_name=sheet_name)
    df = df.rename(columns={
        "Content (mM)": "Content_mM",
        "Slope (ΔA340/s)": "Slope_AperS",
    })

    required_columns = [
        "Content_mM",
        "Replicate",
        "Slope_AperS",
        "Enzyme",
        "Temperature",
        "enzyme conc. ng/ul",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()

    for column in ["Content_mM", "Replicate", "Temperature", "enzyme conc. ng/ul"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Slope_AperS"] = (
        df["Slope_AperS"]
        .astype(str)
        .str.replace("−", "-", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["Slope_AperS"] = pd.to_numeric(df["Slope_AperS"], errors="coerce")
    df["Enzyme"] = df["Enzyme"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    df = df.dropna(subset=["Content_mM", "Replicate", "Slope_AperS", "Enzyme", "Temperature"]).copy()
    df["v_mMperS"] = (-df["Slope_AperS"] / (epsilon * pathlength_cm)) * 1000.0
    df["Temperature_label"] = df["Temperature"].map(
        lambda value: f"{int(value)} °C" if float(value).is_integer() else f"{value:g} °C"
    )
    return df


def inspect_kinetics_slope_signs(df):
    positive_slope_rows = (
        df[df["Slope_AperS"] > 0]
        .sort_values(["Temperature_label", "Enzyme", "Content_mM", "Replicate", "Slope_AperS"])
        .reset_index(drop=True)
    )

    mixed_sign_groups = (
        df.assign(
            has_positive_slope=df["Slope_AperS"] > 0,
            has_negative_slope=df["Slope_AperS"] < 0,
        )
        .groupby(["Enzyme", "Temperature_label", "Content_mM"], as_index=False)
        .agg(
            n_rows=("Slope_AperS", "size"),
            n_positive_slope=("has_positive_slope", "sum"),
            n_negative_slope=("has_negative_slope", "sum"),
            min_slope_AperS=("Slope_AperS", "min"),
            max_slope_AperS=("Slope_AperS", "max"),
            mean_rate_mMperS=("v_mMperS", "mean"),
        )
    )
    mixed_sign_groups = mixed_sign_groups[
        (mixed_sign_groups["n_positive_slope"] > 0) & (mixed_sign_groups["n_negative_slope"] > 0)
    ].sort_values(["Temperature_label", "Enzyme", "Content_mM"]).reset_index(drop=True)
    return positive_slope_rows, mixed_sign_groups


def prepare_enzyme_metadata(enzyme_metadata):
    metadata_df = pd.DataFrame(enzyme_metadata).copy()
    required_columns = ["Enzyme", "Display_name", "Color", "enzyme_ng_per_uL", "MW_g_per_mol"]
    missing_columns = [column for column in required_columns if column not in metadata_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required enzyme metadata columns: {missing_columns}")

    if "Variant_group" not in metadata_df.columns:
        metadata_df["Variant_group"] = metadata_df["Display_name"]

    metadata_df["plot_order"] = np.arange(len(metadata_df))
    return metadata_df


def exclude_kinetics_rows(df, exclusions=None):
    if not exclusions:
        return df.copy(), df.iloc[0:0].copy()

    combined_mask = pd.Series(False, index=df.index)
    for exclusion in exclusions:
        mask = pd.Series(True, index=df.index)
        for column, value in exclusion.items():
            if column not in df.columns:
                raise KeyError(f"Unknown exclusion column: {column}")

            if pd.api.types.is_numeric_dtype(df[column]):
                mask &= np.isclose(df[column].astype(float), float(value), equal_nan=False)
            else:
                mask &= df[column].astype(str) == str(value)

        combined_mask |= mask

    filtered_df = df.loc[~combined_mask].copy()
    excluded_df = df.loc[combined_mask].copy()
    return filtered_df, excluded_df


def michaelis_menten(substrate_mM, vmax_mM_per_s, km_mM):
    substrate_mM = np.asarray(substrate_mM, dtype=float)
    return (vmax_mM_per_s * substrate_mM) / (km_mM + substrate_mM)


def summarize_rates_by_substrate(group_df):
    mean_by_substrate = (
        group_df.groupby("Content_mM", as_index=False)
        .agg(
            mean_v=("v_mMperS", "mean"),
            sd_v=("v_mMperS", "std"),
            n=("v_mMperS", "size"),
        )
        .sort_values("Content_mM")
        .reset_index(drop=True)
    )
    mean_by_substrate["sd_v"] = mean_by_substrate["sd_v"].fillna(0.0)
    return mean_by_substrate


def calculate_turnover_metrics(vmax_mM_per_s, km_mM, enzyme_row):
    enzyme_conc_M = (enzyme_row["enzyme_ng_per_uL"] * 1e-3) / enzyme_row["MW_g_per_mol"]
    vmax_M_per_s = vmax_mM_per_s / 1000.0
    kcat_s_inv = vmax_M_per_s / enzyme_conc_M if enzyme_conc_M > 0 else np.nan
    kcat_over_Km = kcat_s_inv / km_mM if km_mM > 0 else np.nan
    return enzyme_conc_M, kcat_s_inv, kcat_over_Km


def fit_michaelis_menten_parameters(mean_by_substrate, enzyme_row, fit_points=250):
    substrate = mean_by_substrate["Content_mM"].to_numpy(dtype=float)
    mean_rate = mean_by_substrate["mean_v"].to_numpy(dtype=float)
    sigma = mean_by_substrate["sd_v"].replace(0.0, np.nan)
    sigma_floor = sigma[sigma > 0].min()
    if pd.isna(sigma_floor):
        sigma_floor = max(float(np.nanmax(mean_rate)) * 0.05, 1e-6)
    sigma = sigma.fillna(sigma_floor).to_numpy(dtype=float)

    p0 = [max(float(np.nanmax(mean_rate)), 1e-6), float(np.nanmedian(substrate))]

    popt, pcov = curve_fit(
        michaelis_menten,
        substrate,
        mean_rate,
        p0=p0,
        sigma=sigma,
        absolute_sigma=False,
        bounds=(0, np.inf),
        maxfev=10000,
    )

    vmax_mM_per_s, km_mM = popt
    parameter_errors = np.sqrt(np.diag(pcov)) if pcov.size else np.array([np.nan, np.nan])
    vmax_err_mM_per_s, km_err_mM = parameter_errors

    predicted_rate = michaelis_menten(substrate, vmax_mM_per_s, km_mM)
    ss_res = np.sum((mean_rate - predicted_rate) ** 2)
    ss_tot = np.sum((mean_rate - np.mean(mean_rate)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)

    enzyme_conc_M, kcat_s_inv, kcat_over_Km = calculate_turnover_metrics(vmax_mM_per_s, km_mM, enzyme_row)

    fit_result = {
        "Vmax_mM_per_s": vmax_mM_per_s,
        "Vmax_err_mM_per_s": vmax_err_mM_per_s,
        "Km_mM": km_mM,
        "Km_err_mM": km_err_mM,
        "kcat_s^-1": kcat_s_inv,
        "kcat_over_Km": kcat_over_Km,
        "r_squared": r_squared,
        "enzyme_conc_M": enzyme_conc_M,
    }

    if fit_points and fit_points > 0:
        fit_x = np.linspace(0, max(float(substrate.max()) * 1.05, 0.05), fit_points)
        fit_result["fit_x"] = fit_x
        fit_result["fit_y"] = michaelis_menten(fit_x, vmax_mM_per_s, km_mM)

    return fit_result


def bootstrap_kinetics_parameters(group_df, enzyme_row, n_bootstrap=1000, ci_percent=95, random_seed=0):
    metrics = ["Vmax_mM_per_s", "Km_mM", "kcat_s^-1", "kcat_over_Km"]
    if n_bootstrap <= 0:
        return {
            **{f"{metric}_bootstrap_sd": np.nan for metric in metrics},
            **{f"{metric}_ci_low": np.nan for metric in metrics},
            **{f"{metric}_ci_high": np.nan for metric in metrics},
            "bootstrap_n_success": 0,
            "bootstrap_n_attempted": 0,
            "bootstrap_ci_percent": ci_percent,
        }

    substrate_groups = [
        substrate_df.reset_index(drop=True)
        for _, substrate_df in group_df.groupby("Content_mM", sort=True)
    ]
    rng = np.random.default_rng(random_seed)
    bootstrap_records = []

    for _ in range(int(n_bootstrap)):
        sampled_frames = []
        for substrate_df in substrate_groups:
            sample_indices = rng.integers(0, len(substrate_df), size=len(substrate_df))
            sampled_frames.append(substrate_df.iloc[sample_indices].copy())

        bootstrap_group = pd.concat(sampled_frames, ignore_index=True)
        bootstrap_mean_by_substrate = summarize_rates_by_substrate(bootstrap_group)

        try:
            fit_result = fit_michaelis_menten_parameters(
                bootstrap_mean_by_substrate,
                enzyme_row,
                fit_points=0,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        bootstrap_records.append({metric: fit_result[metric] for metric in metrics})

    summary = {
        "bootstrap_n_success": len(bootstrap_records),
        "bootstrap_n_attempted": int(n_bootstrap),
        "bootstrap_ci_percent": ci_percent,
    }
    alpha = (100 - ci_percent) / 2

    if not bootstrap_records:
        for metric in metrics:
            summary[f"{metric}_bootstrap_sd"] = np.nan
            summary[f"{metric}_ci_low"] = np.nan
            summary[f"{metric}_ci_high"] = np.nan
        return summary

    bootstrap_df = pd.DataFrame(bootstrap_records)
    for metric in metrics:
        values = bootstrap_df[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            summary[f"{metric}_bootstrap_sd"] = np.nan
            summary[f"{metric}_ci_low"] = np.nan
            summary[f"{metric}_ci_high"] = np.nan
            continue

        summary[f"{metric}_bootstrap_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary[f"{metric}_ci_low"] = float(np.percentile(values, alpha))
        summary[f"{metric}_ci_high"] = float(np.percentile(values, 100 - alpha))

    return summary


def fit_kinetics_group(group_df, enzyme_row, fit_points=250, bootstrap_iterations=1000, bootstrap_ci_percent=95, bootstrap_seed=0):
    mean_by_substrate = summarize_rates_by_substrate(group_df)
    fit_result = fit_michaelis_menten_parameters(mean_by_substrate, enzyme_row, fit_points=fit_points)

    fit_curve = pd.DataFrame({
        "Content_mM": fit_result["fit_x"],
        "fitted_v": fit_result["fit_y"],
        "Enzyme": enzyme_row["Enzyme"],
        "Display_name": enzyme_row["Display_name"],
        "Color": enzyme_row["Color"],
        "Variant_group": enzyme_row["Variant_group"],
        "Temperature": group_df["Temperature"].iloc[0],
        "Temperature_label": group_df["Temperature_label"].iloc[0],
    })

    mean_points = mean_by_substrate.assign(
        Enzyme=enzyme_row["Enzyme"],
        Display_name=enzyme_row["Display_name"],
        Color=enzyme_row["Color"],
        Variant_group=enzyme_row["Variant_group"],
        Temperature=group_df["Temperature"].iloc[0],
        Temperature_label=group_df["Temperature_label"].iloc[0],
    )

    summary_row = {
        "Enzyme": enzyme_row["Enzyme"],
        "Display_name": enzyme_row["Display_name"],
        "Variant_group": enzyme_row["Variant_group"],
        "Color": enzyme_row["Color"],
        "Temperature": group_df["Temperature"].iloc[0],
        "Temperature_label": group_df["Temperature_label"].iloc[0],
        "n_points": len(group_df),
        "n_substrate_levels": mean_by_substrate["Content_mM"].nunique(),
        "Vmax_mM_per_s": fit_result["Vmax_mM_per_s"],
        "Vmax_err_mM_per_s": fit_result["Vmax_err_mM_per_s"],
        "Km_mM": fit_result["Km_mM"],
        "Km_err_mM": fit_result["Km_err_mM"],
        "kcat_s^-1": fit_result["kcat_s^-1"],
        "kcat_over_Km": fit_result["kcat_over_Km"],
        "r_squared": fit_result["r_squared"],
        "enzyme_ng_per_uL": enzyme_row["enzyme_ng_per_uL"],
        "MW_g_per_mol": enzyme_row["MW_g_per_mol"],
    }

    summary_row.update(
        bootstrap_kinetics_parameters(
            group_df,
            enzyme_row,
            n_bootstrap=bootstrap_iterations,
            ci_percent=bootstrap_ci_percent,
            random_seed=bootstrap_seed,
        )
    )
    return summary_row, mean_points, fit_curve


def fit_all_kinetics(df, enzyme_metadata, bootstrap_iterations=1000, bootstrap_ci_percent=95, bootstrap_seed=12345):
    metadata_df = prepare_enzyme_metadata(enzyme_metadata)
    merged_df = df.merge(metadata_df, on="Enzyme", how="left", validate="m:1")

    if merged_df[["Display_name", "Color", "enzyme_ng_per_uL", "MW_g_per_mol"]].isna().any().any():
        missing = merged_df.loc[
            merged_df[["Display_name", "Color", "enzyme_ng_per_uL", "MW_g_per_mol"]].isna().any(axis=1),
            "Enzyme",
        ].unique()
        raise ValueError(f"Missing metadata for enzymes: {sorted(missing)}")

    summary_rows = []
    mean_points_frames = []
    fit_curve_frames = []

    grouped = merged_df.groupby(["Enzyme", "Temperature"], sort=True)
    for group_index, ((_, _), group_df) in enumerate(grouped):
        enzyme_row = group_df.iloc[0]
        summary_row, mean_points, fit_curve = fit_kinetics_group(
            group_df,
            enzyme_row,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_ci_percent=bootstrap_ci_percent,
            bootstrap_seed=bootstrap_seed + group_index,
        )
        summary_rows.append(summary_row)
        mean_points_frames.append(mean_points)
        fit_curve_frames.append(fit_curve)

    fit_summary = pd.DataFrame(summary_rows)
    fit_summary["plot_order"] = fit_summary["Enzyme"].map(metadata_df.set_index("Enzyme")["plot_order"])
    fit_summary = fit_summary.sort_values(["Temperature", "plot_order"]).reset_index(drop=True)

    mean_points_df = pd.concat(mean_points_frames, ignore_index=True)
    fit_curve_df = pd.concat(fit_curve_frames, ignore_index=True)
    return fit_summary, mean_points_df, fit_curve_df


def build_parameter_table(summary_df, value_column, enzyme_metadata):
    metadata_df = prepare_enzyme_metadata(enzyme_metadata)
    display_order = metadata_df["Display_name"].tolist()
    temperature_order = summary_df.sort_values("Temperature")["Temperature_label"].drop_duplicates().tolist()

    parameter_table = summary_df.pivot(index="Display_name", columns="Temperature_label", values=value_column)
    parameter_table = parameter_table.reindex(display_order)
    parameter_table = parameter_table.reindex(columns=temperature_order)
    return parameter_table


def build_combined_parameter_table(summary_df, enzyme_metadata, parameter_specs):
    combined_tables = []
    for value_column, metric_label, _ in parameter_specs:
        parameter_table = build_parameter_table(
            summary_df,
            value_column=value_column,
            enzyme_metadata=enzyme_metadata,
        )
        parameter_table.columns = pd.MultiIndex.from_product([[metric_label], parameter_table.columns])
        combined_tables.append(parameter_table)

    combined_table = pd.concat(combined_tables, axis=1)
    combined_table.index.name = "Variant"
    return combined_table


def format_estimate_with_uncertainty(value, decimals=3, uncertainty_mode="none", sd_value=np.nan, ci_low=np.nan, ci_high=np.nan):
    estimate_text = format_table_value(value, decimals=decimals)
    if estimate_text == "" or uncertainty_mode == "none":
        return estimate_text

    if uncertainty_mode == "sd":
        if pd.isna(sd_value):
            return estimate_text
        return f"{estimate_text} ± {format_table_value(sd_value, decimals=decimals)}"

    if uncertainty_mode == "ci":
        if pd.isna(ci_low) or pd.isna(ci_high):
            return estimate_text
        return (
            f"{estimate_text} "
            f"[{format_table_value(ci_low, decimals=decimals)}, {format_table_value(ci_high, decimals=decimals)}]"
        )

    raise ValueError(f"Unknown uncertainty mode: {uncertainty_mode}")


def build_parameter_display_table(summary_df, enzyme_metadata, parameter_specs, uncertainty_mode="none"):
    display_tables = []
    for value_column, metric_label, decimals in parameter_specs:
        estimate_table = build_parameter_table(
            summary_df,
            value_column=value_column,
            enzyme_metadata=enzyme_metadata,
        )
        formatted_table = estimate_table.copy().astype(object)

        if uncertainty_mode == "sd":
            sd_table = build_parameter_table(
                summary_df,
                value_column=f"{value_column}_bootstrap_sd",
                enzyme_metadata=enzyme_metadata,
            )
        elif uncertainty_mode == "ci":
            ci_low_table = build_parameter_table(
                summary_df,
                value_column=f"{value_column}_ci_low",
                enzyme_metadata=enzyme_metadata,
            )
            ci_high_table = build_parameter_table(
                summary_df,
                value_column=f"{value_column}_ci_high",
                enzyme_metadata=enzyme_metadata,
            )

        for row_label in formatted_table.index:
            for column_label in formatted_table.columns:
                kwargs = {}
                if uncertainty_mode == "sd":
                    kwargs["sd_value"] = sd_table.loc[row_label, column_label]
                elif uncertainty_mode == "ci":
                    kwargs["ci_low"] = ci_low_table.loc[row_label, column_label]
                    kwargs["ci_high"] = ci_high_table.loc[row_label, column_label]

                formatted_table.loc[row_label, column_label] = format_estimate_with_uncertainty(
                    estimate_table.loc[row_label, column_label],
                    decimals=decimals,
                    uncertainty_mode=uncertainty_mode,
                    **kwargs,
                )

        formatted_table.columns = pd.MultiIndex.from_product([[metric_label], formatted_table.columns])
        display_tables.append(formatted_table)

    combined_table = pd.concat(display_tables, axis=1)
    combined_table.index.name = "Variant"
    return combined_table


def build_parameter_display_tables_by_temperature(summary_df, enzyme_metadata, parameter_specs, uncertainty_mode="none"):
    temperature_tables = {}
    temperature_order = summary_df.sort_values("Temperature")["Temperature_label"].drop_duplicates().tolist()

    for temperature_label in temperature_order:
        temperature_summary = summary_df[summary_df["Temperature_label"] == temperature_label].copy()
        temperature_table = build_parameter_display_table(
            temperature_summary,
            enzyme_metadata=enzyme_metadata,
            parameter_specs=parameter_specs,
            uncertainty_mode=uncertainty_mode,
        )
        temperature_table.columns = [metric_label for metric_label, _ in temperature_table.columns]
        temperature_table.index.name = "Variant"
        temperature_tables[temperature_label] = temperature_table

    return temperature_tables


def format_table_value(value, decimals=3):
    if pd.isna(value):
        return ""

    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1000 or (0 < abs_value < 10 ** (-decimals)):
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def format_combined_parameter_table(table_df, parameter_specs):
    decimals_lookup = {metric_label: decimals for _, metric_label, decimals in parameter_specs}
    formatted_table = table_df.copy()
    for column in formatted_table.columns:
        metric_label = column[0] if isinstance(column, tuple) else column
        decimals = decimals_lookup.get(metric_label, 3)
        formatted_table[column] = formatted_table[column].map(
            lambda value, decimals=decimals: format_table_value(value, decimals=decimals)
        )

    formatted_table.index.name = table_df.index.name
    return formatted_table


def save_dataframe_table_pdf(
    table_df,
    output_path,
    title=None,
    size_mm=(220, 70),
    font_size=6,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table_df.columns, pd.MultiIndex):
        column_labels = [table_df.index.name or "Variant"] + [
            f"{metric_label}\n{temperature_label}" for metric_label, temperature_label in table_df.columns
        ]
    else:
        column_labels = [table_df.index.name or "Variant"] + [str(column) for column in table_df.columns]

    cell_text = [[str(index), *row.tolist()] for index, row in table_df.iterrows()]

    fig, ax = plt.subplots(figsize=mm_to_inches_KINETICS(size_mm), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=column_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)
    if hasattr(table, "auto_set_column_width"):
        table.auto_set_column_width(col=list(range(len(column_labels))))

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#C7C7C7")
        cell.set_linewidth(0.45)
        if row_index == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(weight="bold")
        elif column_index == 0:
            cell.set_facecolor("#F7F7F7")
            cell.set_text_props(weight="bold")

    if title:
        ax.set_title(title, fontsize=font_size + 1, pad=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_rate_curves_by_temperature(
    mean_points_df,
    fit_curve_df,
    fit_summary,
    enzyme_metadata,
    size_mm=(85, 65),
    font_size=6,
    rate_scale=1000.0,
    rate_unit="µM/s",
    output_dir="./"
):
    metadata_df = prepare_enzyme_metadata(enzyme_metadata)
    enzyme_order = metadata_df["Enzyme"].tolist()
    palette = metadata_df.set_index("Enzyme")["Color"].to_dict()
    display_name_map = metadata_df.set_index("Enzyme")["Display_name"].to_dict()
    temperature_order = sorted(fit_summary["Temperature"].unique())

    with plt.rc_context({
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }):
        for temperature in temperature_order:
            temperature_means = mean_points_df[mean_points_df["Temperature"] == temperature]
            temperature_fits = fit_curve_df[fit_curve_df["Temperature"] == temperature]

            x_max = max(float(temperature_means["Content_mM"].max()), float(temperature_fits["Content_mM"].max())) * 1.02
            y_max = max(float(temperature_means["mean_v"].max()), float(temperature_fits["fitted_v"].max())) * rate_scale * 1.12

            fig, ax = plt.subplots(figsize=mm_to_inches_KINETICS(size_mm), dpi=300)
            for enzyme in enzyme_order:
                mean_subset = temperature_means[temperature_means["Enzyme"] == enzyme].sort_values("Content_mM")
                fit_subset = temperature_fits[temperature_fits["Enzyme"] == enzyme].sort_values("Content_mM")
                if mean_subset.empty:
                    continue

                color = palette[enzyme]
                ax.plot(
                    fit_subset["Content_mM"],
                    fit_subset["fitted_v"] * rate_scale,
                    color=color,
                    linewidth=1.4,
                    label=display_name_map[enzyme],
                )
                ax.errorbar(
                    mean_subset["Content_mM"],
                    mean_subset["mean_v"] * rate_scale,
                    yerr=mean_subset["sd_v"] * rate_scale,
                    fmt="o",
                    markersize=2.8,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    elinewidth=0.7,
                    capsize=1.5,
                )

            ax.set_title(f"Batch 5 kinetics, {int(temperature)} °C" if float(temperature).is_integer() else f"Batch 5 kinetics, {temperature:g} °C")
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("OAA concentration (mM)")
            ax.set_ylabel(f"Rate ({rate_unit})")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(frameon=False, ncol=2, loc="best")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/OAA_rate_temperature-{temperature}.pdf")
            plt.show()