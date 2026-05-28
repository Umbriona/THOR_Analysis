

####################################################################################
##                         Thermal Shift Assay helpers                            ##  
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


def select_best_variants(result_summary_df):
    blank_labels = {"blank", "blanks", "empty", "control"}
    _, variant_summary = summarize_tm_shifts(result_summary_df[["#Annotation", "#Tm"]])

    return (
        variant_summary.loc[
            ~variant_summary["wildtype_id"].str.lower().isin(blank_labels)
        ]
        .sort_values(
            ["wildtype_id", "Tm_difference_vs_wildtype", "variant_name"],
            ascending=[True, False, True]
        )
        .drop_duplicates("wildtype_id")
        .reset_index(drop=True)
    )


def well_index_sort_key(well_index):
    well_index = str(well_index)
    row_label = "".join(char for char in well_index if char.isalpha())
    column_label = "".join(char for char in well_index if char.isdigit())

    return row_label, int(column_label) if column_label else -1


def load_tsa_curve(curve_dir, well_index):
    from pathlib import Path

    curve_path = Path(curve_dir) / f"{well_index}.txt"
    data_rows = []

    for line in curve_path.read_text().splitlines():
        if not line:
            continue
        if not line.startswith("#"):
            data_rows.append(line.split("\t"))

    curve_df = pd.DataFrame(data_rows, columns=["temperature_c", "absorbance"])
    curve_df["temperature_c"] = pd.to_numeric(curve_df["temperature_c"], errors="coerce")
    curve_df["absorbance"] = pd.to_numeric(curve_df["absorbance"], errors="coerce")
    curve_df = curve_df.dropna(how="any").sort_values("temperature_c").reset_index(drop=True)

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


def interpolate_curve_y_at_temperature(curve_df, target_temperature):
    curve_sorted = curve_df.sort_values("temperature_c")
    x_values = curve_sorted["temperature_c"].to_numpy(dtype=float)
    y_values = curve_sorted["normalized_absorbance"].to_numpy(dtype=float)

    if len(x_values) == 0 or pd.isna(target_temperature):
        return np.nan
    if target_temperature < x_values.min() or target_temperature > x_values.max():
        return np.nan

    return float(np.interp(float(target_temperature), x_values, y_values))


def prepare_best_variant_curve_data(result_summary_df, curve_dir):
    result_df = result_summary_df.copy()
    result_df["row_order"] = np.arange(len(result_df))
    result_df["#Annotation"] = result_df["#Annotation"].astype(str).str.strip()
    result_df["#Tm"] = pd.to_numeric(result_df["#Tm"], errors="coerce")
    result_df = result_df.dropna(subset=["#Annotation", "#Tm", "#Well_index"])

    best_variants = select_best_variants(result_df)
    sample_lookup = pd.concat(
        [
            best_variants.assign(
                sample_name=best_variants["wildtype_id"],
                sample_type="Wildtype"
            ),
            best_variants.assign(
                sample_name=best_variants["variant_name"],
                sample_type="Best variant"
            )
        ],
        ignore_index=True
    )[[
        "wildtype_id",
        "variant_name",
        "sample_name",
        "sample_type",
        "Tm_difference_vs_wildtype",
        "variant_mean_Tm",
        "wildtype_mean_Tm",
        "n_replicates"
    ]]

    selected_replicates = (
        result_df.merge(sample_lookup, left_on="#Annotation", right_on="sample_name", how="inner")
        .reset_index(drop=True)
    )
    selected_replicates["well_sort_key"] = selected_replicates["#Well_index"].map(well_index_sort_key)
    selected_replicates = selected_replicates.sort_values(
        ["wildtype_id", "sample_type", "row_order", "well_sort_key"]
    ).reset_index(drop=True)
    selected_replicates["replicate_id"] = (
        selected_replicates.groupby(["wildtype_id", "sample_name"]).cumcount() + 1
    )

    curve_frames = []
    for _, row in selected_replicates.iterrows():
        curve_df = load_tsa_curve(curve_dir, row["#Well_index"])
        curve_df["wildtype_id"] = row["wildtype_id"]
        curve_df["variant_name"] = row["variant_name"]
        curve_df["sample_name"] = row["sample_name"]
        curve_df["sample_type"] = row["sample_type"]
        curve_df["well_index"] = row["#Well_index"]
        curve_df["replicate_id"] = row["replicate_id"]
        curve_df["Tm"] = row["#Tm"]
        curve_frames.append(curve_df)

    curve_data = pd.concat(curve_frames, ignore_index=True)

    return best_variants, selected_replicates, curve_data


def plot_best_variant_curves(curve_data, best_variants, replicate_to_plot=1, size_mm=(120, 60), font_size=5.5, order=None):
    from matplotlib.lines import Line2D

    if order is None:
        order = best_variants["wildtype_id"].tolist()
    available_replicates = sorted(curve_data["replicate_id"].dropna().astype(int).unique())
    if replicate_to_plot not in available_replicates:
        raise ValueError(f"Replicate {replicate_to_plot} not found. Available replicates: {available_replicates}")

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
    line_alphas = {"Wildtype": 0.55, "Best variant": 0.85}

    with plt.rc_context(rc_params):
        fig, ax = plt.subplots(figsize=(mm_to_inches(size_mm[0]), mm_to_inches(size_mm[1])))
        replicate_subset = curve_data.loc[curve_data["replicate_id"] == replicate_to_plot]

        for wildtype_id in order:
            for sample_type in ["Wildtype", "Best variant"]:
                subset = replicate_subset.loc[
                    (replicate_subset["wildtype_id"] == wildtype_id)
                    & (replicate_subset["sample_type"] == sample_type)
                ]

                for _, curve in subset.groupby("well_index"):
                    curve = curve.sort_values("temperature_c")
                    color = wildtype_palette.get(wildtype_id, "#666666")
                    alpha = line_alphas[sample_type]

                    ax.plot(
                        curve["temperature_c"],
                        curve["normalized_absorbance"],
                        color=color,
                        linestyle=line_styles[sample_type],
                        linewidth=0.9,
                        alpha=alpha
                    )

                    tm_value = pd.to_numeric(curve["Tm"], errors="coerce").dropna()
                    if not tm_value.empty:
                        tm_value = float(tm_value.iloc[0])
                        tm_curve_y = interpolate_curve_y_at_temperature(curve, tm_value)

                        if pd.notna(tm_curve_y):
                            ax.vlines(
                                tm_value,
                                ymin=0,
                                ymax=tm_curve_y,
                                color=color,
                                linestyle=(0, (3, 2)),
                                linewidth=0.7,
                                alpha=min(alpha + 0.1, 1.0)
                            )
                            ax.scatter(
                                [tm_value],
                                [tm_curve_y],
                                color=color,
                                s=10,
                                edgecolors="white",
                                linewidths=0.25,
                                alpha=1.0,
                                zorder=4
                            )

        ax.set_title(f"Batch 3 TSA curves, replicate {replicate_to_plot}")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Normalized absorbance")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([30, 40, 50, 60, 70, 80, 90, 100])
        ax.tick_params(axis="x", length=2, width=0.5)
        ax.tick_params(axis="y", length=2, width=0.5)

        legend_handles = []
        for wildtype_id in order:
            best_variant_name = best_variants.loc[
                best_variants["wildtype_id"] == wildtype_id,
                "variant_name"
            ].iloc[0]
            color = wildtype_palette.get(wildtype_id, "#666666")
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle="-", linewidth=1.0, label=f"{wildtype_id} WT")
            )
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle="--", linewidth=1.0, label=best_variant_name)
            )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="#666666",
                linestyle=(0, (3, 2)),
                marker="o",
                markerfacecolor="#666666",
                markeredgecolor="white",
                markeredgewidth=0.25,
                markersize=3.0,
                linewidth=0.8,
                label="Measured Tm"
            )
        )

        ax.legend(handles=legend_handles, frameon=False, ncol=1, loc="best")
        plt.tight_layout(pad=0.4)

    return ax
