"""Small analysis helpers used by ``statements.ipynb``."""

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import tempfile

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def summarize_activity_retention_above_wt_tm(
    activity_summary,
    tm_summary,
    retained_activity_threshold=50,
):
    """Quantify variants retaining activity at a measured temperature above WT Tm.

    A variant is counted as retained when its relative activity is at least
    ``retained_activity_threshold`` at one or more assayed temperatures strictly
    above the median melting temperature of its corresponding wild type.

    Parameters
    ----------
    activity_summary : pandas.DataFrame
        Output from ``helper_figure3.compile_activity_resistance``.
    tm_summary : pandas.DataFrame
        TSA data containing ``#Annotation`` and ``#Tm``.
    retained_activity_threshold : float, default 50
        Minimum relative activity (%) required for retention.

    Returns
    -------
    variant_summary, overall_summary : tuple[pandas.DataFrame, pandas.DataFrame]
        Per-variant results and a one-row overall count/fraction summary.
    """
    required_activity_columns = {
        "sample_name",
        "wildtype_id",
        "temperature_c",
        "percent_active",
    }
    missing_activity_columns = required_activity_columns.difference(activity_summary.columns)
    if missing_activity_columns:
        raise ValueError(
            "activity_summary is missing columns: "
            + ", ".join(sorted(missing_activity_columns))
        )

    required_tm_columns = {"#Annotation", "#Tm"}
    missing_tm_columns = required_tm_columns.difference(tm_summary.columns)
    if missing_tm_columns:
        raise ValueError(
            "tm_summary is missing columns: " + ", ".join(sorted(missing_tm_columns))
        )

    tm_data = tm_summary.loc[:, ["#Annotation", "#Tm"]].copy()
    tm_data["#Annotation"] = tm_data["#Annotation"].astype(str).str.strip()
    tm_data["#Tm"] = pd.to_numeric(tm_data["#Tm"], errors="coerce")
    median_tm = tm_data.groupby("#Annotation", as_index=False)["#Tm"].median()

    wildtype_tm = median_tm.rename(
        columns={"#Annotation": "wildtype_id", "#Tm": "wildtype_tm_c"}
    )

    variants = activity_summary.loc[
        activity_summary["sample_name"] != activity_summary["wildtype_id"]
    ].copy()
    variants = variants.merge(wildtype_tm, on="wildtype_id", how="left")

    missing_wildtypes = sorted(
        variants.loc[variants["wildtype_tm_c"].isna(), "wildtype_id"].unique()
    )
    if missing_wildtypes:
        raise ValueError(
            "No melting temperature found for wild type(s): "
            + ", ".join(missing_wildtypes)
        )

    above_tm = variants.loc[
        variants["temperature_c"] > variants["wildtype_tm_c"]
    ].dropna(subset=["percent_active"])

    variant_summary = (
        above_tm.groupby(["wildtype_id", "sample_name", "wildtype_tm_c"], as_index=False)
        .agg(
            n_temperatures_tested_above_wt_tm=("temperature_c", "nunique"),
            maximum_temperature_tested_c=("temperature_c", "max"),
            maximum_percent_activity_above_wt_tm=("percent_active", "max"),
        )
    )
    variant_summary["retains_activity_above_wt_tm"] = (
        variant_summary["maximum_percent_activity_above_wt_tm"]
        >= retained_activity_threshold
    )

    n_variants = len(variant_summary)
    n_retained = int(variant_summary["retains_activity_above_wt_tm"].sum())
    overall_summary = pd.DataFrame(
        {
            "n_variants_with_measurements_above_wt_tm": [n_variants],
            "n_variants_retaining_activity_above_wt_tm": [n_retained],
            "fraction_variants_retaining_activity_above_wt_tm": [
                n_retained / n_variants if n_variants else np.nan
            ],
            "percent_variants_retaining_activity_above_wt_tm": [
                100 * n_retained / n_variants if n_variants else np.nan
            ],
            "retained_activity_threshold_percent": [
                float(retained_activity_threshold)
            ],
        }
    )

    return (
        variant_summary.sort_values(["wildtype_id", "sample_name"]).reset_index(drop=True),
        overall_summary,
    )


def _load_extreme_temperature_metadata(
    fasta_path,
    thermophile_temperature=60,
    mesophile_temperature=45,
):
    """Read IDs and temperatures without retaining the large sequence strings."""
    metadata = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        try:
            temperature = float(record.description.rsplit(maxsplit=1)[-1])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Could not parse a temperature from FASTA header: {record.description}"
            ) from error

        if temperature >= thermophile_temperature:
            group = "thermophile"
        elif temperature < mesophile_temperature:
            group = "mesophile"
        else:
            continue

        metadata[record.id] = {
            "temperature_c": temperature,
            "temperature_group": group,
        }

    return metadata


def _add_root_cogs_from_emapper(metadata, annotation_path):
    """Add the root COG using the same first-eggNOG-OG rule as Figure 1."""
    with open(annotation_path, encoding="utf-8") as annotation_handle:
        header = None
        for line in annotation_handle:
            if line.startswith("#query\t"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                break

        if header is None:
            raise ValueError("No '#query' header found in the eggNOG annotation file")

        reader = csv.DictReader(annotation_handle, fieldnames=header, delimiter="\t")
        for row in reader:
            query_id = row["query"]
            if query_id not in metadata:
                continue

            eggnog_ogs = row.get("eggNOG_OGs", "-")
            if not eggnog_ogs or eggnog_ogs == "-":
                continue

            metadata[query_id]["root_cog"] = eggnog_ogs.split(",", 1)[0].split("@", 1)[0]


def load_thermophile_mesophile_cogs(
    fasta_path,
    annotation_path,
    thermophile_temperature=60,
    mesophile_temperature=45,
):
    """Load sequences from COGs containing both temperature groups.

    The large input files are streamed. Sequence strings are retained only for
    members of COGs containing at least one thermophile and one mesophile.
    """
    metadata = _load_extreme_temperature_metadata(
        fasta_path,
        thermophile_temperature=thermophile_temperature,
        mesophile_temperature=mesophile_temperature,
    )
    _add_root_cogs_from_emapper(metadata, annotation_path)

    catalog = pd.DataFrame.from_dict(metadata, orient="index")
    catalog.index.name = "sequence_id"
    catalog = catalog.dropna(subset=["root_cog"]).reset_index()

    group_counts = (
        catalog.groupby(["root_cog", "temperature_group"])["sequence_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    eligible_cogs = group_counts.index[
        (group_counts.get("thermophile", 0) > 0)
        & (group_counts.get("mesophile", 0) > 0)
    ]
    catalog = catalog.loc[catalog["root_cog"].isin(eligible_cogs)].copy()

    selected_ids = set(catalog["sequence_id"])
    sequence_lookup = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id in selected_ids:
            sequence_lookup[record.id] = str(record.seq)

    missing_ids = selected_ids.difference(sequence_lookup)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} annotated sequence(s) were not found in the FASTA file"
        )

    catalog["sequence"] = catalog["sequence_id"].map(sequence_lookup)
    return catalog.sort_values(
        ["root_cog", "temperature_group", "sequence_id"]
    ).reset_index(drop=True)


def _read_identity_blocks(
    matrix_path,
    n_thermophiles,
    n_mesophiles,
):
    """Read cross-group and unique within-group pairs from a Clustal matrix."""
    cross_group_rows = []
    thermophile_chunks = []
    mesophile_chunks = []
    expected_matrix_size = n_thermophiles + n_mesophiles

    with open(matrix_path, encoding="utf-8") as matrix_handle:
        matrix_size = int(matrix_handle.readline().strip())
        if matrix_size != expected_matrix_size:
            raise ValueError(
                f"Clustal matrix contains {matrix_size} sequences; "
                f"expected {expected_matrix_size}"
            )

        for row_index in range(matrix_size):
            row = matrix_handle.readline().strip()
            _, values_text = row.split(maxsplit=1)
            values = np.fromstring(values_text, sep=" ")
            if len(values) != matrix_size:
                raise ValueError(
                    f"Clustal matrix row {row_index} contains {len(values)} values; "
                    f"expected {matrix_size}"
                )

            if row_index < n_thermophiles:
                cross_group_rows.append(values[n_thermophiles:])
                if row_index + 1 < n_thermophiles:
                    thermophile_chunks.append(
                        values[row_index + 1:n_thermophiles]
                    )
            else:
                mesophile_index = row_index - n_thermophiles
                if mesophile_index + 1 < n_mesophiles:
                    mesophile_chunks.append(values[row_index + 1:])

    return {
        "cross_group": np.concatenate(cross_group_rows),
        "thermophile": (
            np.concatenate(thermophile_chunks)
            if thermophile_chunks
            else np.array([], dtype=float)
        ),
        "mesophile": (
            np.concatenate(mesophile_chunks)
            if mesophile_chunks
            else np.array([], dtype=float)
        ),
    }


def _read_two_sequence_alignment_identity(alignment_path):
    """Match Clustal's percent-ID convention for its two-sequence edge case."""
    aligned_records = list(SeqIO.parse(alignment_path, "fasta"))
    if len(aligned_records) != 2:
        raise ValueError(
            f"Expected a two-sequence alignment, found {len(aligned_records)}"
        )

    first_sequence = str(aligned_records[0].seq)
    second_sequence = str(aligned_records[1].seq)
    matches = sum(
        first_residue == second_residue and first_residue != "-"
        for first_residue, second_residue in zip(first_sequence, second_sequence)
    )
    shorter_sequence_length = min(
        len(first_sequence.replace("-", "")),
        len(second_sequence.replace("-", "")),
    )
    return np.array([100 * matches / shorter_sequence_length])


def _identity_statistics(identity_values, prefix):
    """Create count and descriptive-statistic fields for one pair type."""
    if identity_values.size == 0:
        return {
            f"n_{prefix}_pairs": 0,
            f"mean_{prefix}_identity_percent": np.nan,
            f"median_{prefix}_identity_percent": np.nan,
            f"minimum_{prefix}_identity_percent": np.nan,
            f"maximum_{prefix}_identity_percent": np.nan,
        }

    return {
        f"n_{prefix}_pairs": int(identity_values.size),
        f"mean_{prefix}_identity_percent": float(np.mean(identity_values)),
        f"median_{prefix}_identity_percent": float(np.median(identity_values)),
        f"minimum_{prefix}_identity_percent": float(np.min(identity_values)),
        f"maximum_{prefix}_identity_percent": float(np.max(identity_values)),
    }


def summarize_cog_thermophile_mesophile_identity(
    fasta_path,
    annotation_path,
    thermophile_temperature=60,
    mesophile_temperature=45,
    clustalo_executable="clustalo",
    threads=1,
    output_csv=None,
    force=False,
):
    """Summarize cross-temperature identity from Clustal percent-ID matrices.

    Each COG FASTA is ordered with thermophiles first and mesophiles second.
    Clustal Omega calculates its full percent-identity matrix, from which only
    the thermophile × mesophile block is retained. No sequences are sampled.

    When ``output_csv`` is supplied, the per-COG results are cached there and
    the pooled results are cached in a companion ``*_overall.csv`` file.
    Existing complete caches are returned immediately unless ``force=True``.
    """
    output_path = Path(output_csv) if output_csv is not None else None
    overall_output_path = None
    if output_path is not None:
        if output_path.suffix.lower() != ".csv":
            raise ValueError("output_csv must have a .csv extension")
        overall_output_path = output_path.with_name(
            f"{output_path.stem}_overall{output_path.suffix}"
        )

        if not force and output_path.exists() and overall_output_path.exists():
            cog_summary = pd.read_csv(output_path)
            overall_summary = pd.read_csv(overall_output_path)

            required_cog_columns = {
                "median_sequence_identity_percent",
                "median_thermophile_identity_percent",
                "median_mesophile_identity_percent",
            }
            missing_cog_columns = required_cog_columns.difference(
                cog_summary.columns
            )
            if missing_cog_columns:
                raise ValueError(
                    "Cached COG results predate the within-group identity "
                    "analysis. Use force=True once to recompute them."
                )

            expected_thresholds = {
                "thermophile_temperature_threshold_c": float(
                    thermophile_temperature
                ),
                "mesophile_temperature_threshold_c": float(
                    mesophile_temperature
                ),
            }
            for column, expected_value in expected_thresholds.items():
                if column not in overall_summary.columns:
                    raise ValueError(
                        f"Cached summary lacks '{column}'. Use force=True to "
                        "recompute it with the current analysis."
                    )
                cached_value = float(overall_summary.loc[0, column])
                if cached_value != expected_value:
                    raise ValueError(
                        f"Cached {column} is {cached_value}, but the requested "
                        f"value is {expected_value}. Use force=True to recompute."
                    )

            print(f"Loaded cached COG identity results from {output_path}")
            return cog_summary, overall_summary

    clustalo_path = shutil.which(clustalo_executable)
    if clustalo_path is None:
        raise FileNotFoundError(
            f"Could not find the MSA executable '{clustalo_executable}'"
        )

    catalog = load_thermophile_mesophile_cogs(
        fasta_path,
        annotation_path,
        thermophile_temperature=thermophile_temperature,
        mesophile_temperature=mesophile_temperature,
    )

    cog_rows = []
    all_cross_group_arrays = []
    all_thermophile_arrays = []
    all_mesophile_arrays = []
    with tempfile.TemporaryDirectory(prefix="cog_identity_") as temporary_directory:
        temporary_path = Path(temporary_directory)

        for cog_index, (root_cog, cog_catalog) in enumerate(
            catalog.groupby("root_cog", sort=True)
        ):
            input_path = temporary_path / f"cog_{cog_index}.fasta"
            alignment_path = temporary_path / f"cog_{cog_index}_aligned.fasta"
            matrix_path = temporary_path / f"cog_{cog_index}_identity.mat"

            thermophile_catalog = cog_catalog.loc[
                cog_catalog["temperature_group"] == "thermophile"
            ]
            mesophile_catalog = cog_catalog.loc[
                cog_catalog["temperature_group"] == "mesophile"
            ]
            ordered_catalog = pd.concat(
                [thermophile_catalog, mesophile_catalog],
                ignore_index=True,
            )
            n_thermophiles = len(thermophile_catalog)
            n_mesophiles = len(mesophile_catalog)

            records = []
            for sequence_index, row in enumerate(
                ordered_catalog.itertuples(index=False)
            ):
                alignment_id = f"sequence_{sequence_index}"
                records.append(
                    SeqRecord(
                        Seq(row.sequence),
                        id=alignment_id,
                        description="",
                    )
                )
            SeqIO.write(records, input_path, "fasta")

            subprocess.run(
                [
                    clustalo_path,
                    "--infile",
                    str(input_path),
                    "--outfile",
                    str(alignment_path),
                    "--outfmt",
                    "fasta",
                    "--distmat-out",
                    str(matrix_path),
                    "--percent-id",
                    "--full",
                    "--force",
                    "--threads",
                    str(threads),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            if matrix_path.exists():
                identity_blocks = _read_identity_blocks(
                    matrix_path,
                    n_thermophiles=n_thermophiles,
                    n_mesophiles=n_mesophiles,
                )
            elif len(ordered_catalog) == 2:
                # Clustal warns and omits --distmat-out when given two sequences.
                identity_blocks = {
                    "cross_group": _read_two_sequence_alignment_identity(
                        alignment_path
                    ),
                    "thermophile": np.array([], dtype=float),
                    "mesophile": np.array([], dtype=float),
                }
            else:
                raise FileNotFoundError(
                    f"Clustal did not create the identity matrix for {root_cog}"
                )

            cross_group_identities = identity_blocks["cross_group"]
            thermophile_identities = identity_blocks["thermophile"]
            mesophile_identities = identity_blocks["mesophile"]
            all_cross_group_arrays.append(cross_group_identities)
            if thermophile_identities.size:
                all_thermophile_arrays.append(thermophile_identities)
            if mesophile_identities.size:
                all_mesophile_arrays.append(mesophile_identities)

            cog_row = {
                "root_cog": root_cog,
                "n_thermophiles": n_thermophiles,
                "n_mesophiles": n_mesophiles,
                "n_cross_group_pairs": cross_group_identities.size,
                "mean_sequence_identity_percent": np.mean(
                    cross_group_identities
                ),
                "median_sequence_identity_percent": np.median(
                    cross_group_identities
                ),
                "minimum_sequence_identity_percent": np.min(
                    cross_group_identities
                ),
                "maximum_sequence_identity_percent": np.max(
                    cross_group_identities
                ),
            }
            cog_row.update(
                _identity_statistics(
                    thermophile_identities,
                    prefix="thermophile",
                )
            )
            cog_row.update(
                _identity_statistics(
                    mesophile_identities,
                    prefix="mesophile",
                )
            )
            cog_rows.append(cog_row)

    cog_summary = pd.DataFrame(cog_rows)
    all_cross_group_identities = np.concatenate(all_cross_group_arrays)
    all_thermophile_identities = (
        np.concatenate(all_thermophile_arrays)
        if all_thermophile_arrays
        else np.array([], dtype=float)
    )
    all_mesophile_identities = (
        np.concatenate(all_mesophile_arrays)
        if all_mesophile_arrays
        else np.array([], dtype=float)
    )
    overall_row = {
        "n_cogs": len(cog_summary),
        "n_thermophile_mesophile_pairs": all_cross_group_identities.size,
        "mean_sequence_identity_percent": np.mean(
            all_cross_group_identities
        ),
        "median_sequence_identity_percent": np.median(
            all_cross_group_identities
        ),
        "minimum_sequence_identity_percent": np.min(
            all_cross_group_identities
        ),
        "maximum_sequence_identity_percent": np.max(
            all_cross_group_identities
        ),
        "thermophile_temperature_threshold_c": float(
            thermophile_temperature
        ),
        "mesophile_temperature_threshold_c": float(
            mesophile_temperature
        ),
        "thermophile_temperature_rule": (
            f">= {thermophile_temperature} °C"
        ),
        "mesophile_temperature_rule": f"< {mesophile_temperature} °C",
    }
    overall_row.update(
        _identity_statistics(
            all_thermophile_identities,
            prefix="thermophile",
        )
    )
    overall_row.update(
        _identity_statistics(
            all_mesophile_identities,
            prefix="mesophile",
        )
    )
    overall_summary = pd.DataFrame([overall_row])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cog_summary.to_csv(output_path, index=False)
        overall_summary.to_csv(overall_output_path, index=False)
        print(f"Saved COG identity results to {output_path}")
        print(f"Saved overall identity summary to {overall_output_path}")

    return cog_summary, overall_summary


def plot_cog_median_identity_histogram(
    cog_summary,
    bin_width=5,
    cog_size_bins=20,
    size_mm=(150, 60),
    font_size=7,
):
    """Plot identity and COG-size histograms in side-by-side panels."""
    identity_series = {
        "thermophile_mesophile": (
            "Thermophile–mesophile",
            "median_sequence_identity_percent",
            "0.20",
        ),
        "within_thermophiles": (
            "Within thermophiles",
            "median_thermophile_identity_percent",
            "#d95f02",
        ),
        "within_mesophiles": (
            "Within mesophiles",
            "median_mesophile_identity_percent",
            "#1b9e77",
        ),
    }
    missing_columns = {
        column
        for _, column, _ in identity_series.values()
        if column not in cog_summary.columns
    }
    missing_columns.update(
        {"n_thermophiles", "n_mesophiles"}.difference(cog_summary.columns)
    )
    if missing_columns:
        raise ValueError(
            "cog_summary is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if bin_width <= 0:
        raise ValueError("bin_width must be greater than zero")
    if cog_size_bins <= 0:
        raise ValueError("cog_size_bins must be greater than zero")

    bins = np.arange(0, 100 + bin_width, bin_width)

    rc_params = {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    }
    axes = {}
    with plt.rc_context(rc_params):
        for plot_name, (label, column, color) in identity_series.items():
            plot_data = pd.DataFrame(
                {
                    "identity": pd.to_numeric(
                        cog_summary[column],
                        errors="coerce",
                    ),
                    "cog_size": (
                        pd.to_numeric(
                            cog_summary["n_thermophiles"],
                            errors="coerce",
                        )
                        + pd.to_numeric(
                            cog_summary["n_mesophiles"],
                            errors="coerce",
                        )
                    ),
                }
            ).dropna()
            values = plot_data["identity"]
            if values.empty:
                continue

            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(size_mm[0] / 25.4, size_mm[1] / 25.4),
            )
            ax1.hist(
                values,
                bins=bins,
                color=color,
                alpha=0.35,
                edgecolor=color,
                linewidth=0.7,
                label=f"COGs (n = {len(values):,})",
            )
            median_identity = float(values.median())
            ax1.axvline(
                median_identity,
                color=color,
                linestyle="--",
                linewidth=1,
                label=f"Median identity: {median_identity:.1f}%",
            )
            ax1.set_xlim(0, 100)
            ax1.set_xlabel(
                "Median pairwise sequence identity per COG (%)"
            )
            ax1.set_ylabel("Number of COGs")
            ax1.legend(frameon=False)
            ax1.spines[["top", "right"]].set_visible(False)

            ax2.hist(
                plot_data["cog_size"],
                bins=cog_size_bins,
                color="0.65",
                edgecolor="black",
                linewidth=0.5,
            )
            ax2.set_xlabel("COG size (number of sequences)")
            ax2.set_ylabel("Number of COGs")
            ax2.spines[["top", "right"]].set_visible(False)

            fig.suptitle(label)
            fig.tight_layout()
            axes[plot_name] = (ax1, ax2)

    return axes
