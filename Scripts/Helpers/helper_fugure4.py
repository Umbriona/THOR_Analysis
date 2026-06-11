from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


DEFAULT_RMSF_DELTA_CMAP = LinearSegmentedColormap.from_list(
    "rmsf_delta",
    ["#2166ac", "#f7f7f7", "#b2182b"],
)


def mm_to_inches(*mm_values: float) -> tuple[float, ...]:
    return tuple(float(value) / 25.4 for value in mm_values)


def short_variant_label(sample_name: str, family_id: str) -> str:
    prefix = f"{family_id}_"
    return sample_name[len(prefix):] if sample_name.startswith(prefix) else sample_name


def choose_residue_ticks(residue_numbers: np.ndarray) -> np.ndarray:
    residue_numbers = np.asarray(residue_numbers, dtype=int)
    if not len(residue_numbers):
        return np.array([], dtype=int)

    min_residue = int(residue_numbers.min())
    max_residue = int(residue_numbers.max())
    residue_span = max_residue - min_residue

    if residue_span <= 80:
        step = 10
    elif residue_span <= 180:
        step = 25
    elif residue_span <= 360:
        step = 50
    else:
        step = 100

    tick_values = np.arange(step, max_residue + 1, step, dtype=int)
    tick_values = np.unique(np.concatenate(([min_residue], tick_values, [max_residue])))
    return tick_values


def parse_ca_trace(pdb_path: Path) -> pd.DataFrame:
    rows = []
    seen = set()

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue

            chain_id = line[21].strip() or "A"
            residue_number = int(line[22:26])
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_number, insertion_code)
            if residue_key in seen:
                continue

            seen.add(residue_key)
            rows.append(
                {
                    "chain_id": chain_id,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "residue_name": line[17:20].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                }
            )

    if not rows:
        raise ValueError(f"No CA atoms found in {pdb_path}")

    return pd.DataFrame(rows)


def discover_structure_families(structure_dir: Path) -> pd.DataFrame:
    pdb_paths = sorted(structure_dir.glob("*.pdb"))
    rows = []

    for path in pdb_paths:
        family_id = path.stem.split("_", 1)[0]
        rows.append(
            {
                "family_id": family_id,
                "sample_name": path.stem,
                "pdb_path": path,
                "is_wildtype": path.stem == family_id,
            }
        )

    family_df = pd.DataFrame(rows)
    if family_df.empty:
        return pd.DataFrame(columns=["family_id", "sample_name", "pdb_path", "is_wildtype"])

    return family_df.sort_values(
        ["family_id", "is_wildtype", "sample_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def resolve_tmalign_executable(preferred_executable: str | None = None) -> str | None:
    candidate_names = []
    if preferred_executable:
        candidate_names.append(str(Path(preferred_executable).expanduser()))
        candidate_names.append(preferred_executable)
    candidate_names.extend(["TMalign", "USalign"])

    seen = set()
    for candidate_name in candidate_names:
        if candidate_name in seen:
            continue
        seen.add(candidate_name)

        resolved = shutil.which(candidate_name)
        if resolved:
            return resolved

        candidate_path = Path(candidate_name).expanduser()
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path)

    return None


def parse_tmalign_stdout(stdout_text: str) -> dict[str, float | int | None]:
    aligned_match = re.search(
        r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)",
        stdout_text,
    )
    tm_matches = re.findall(
        r"TM-score=\s*([0-9.]+)\s*\((?:if\s+)?normalized by length of (?:Chain|Structure)_(\d)(?::[^)]*)?\)",
        stdout_text,
    )

    if len(tm_matches) < 2:
        ordered_tm_scores = re.findall(r"TM-score=\s*([0-9.]+)", stdout_text)
        if len(ordered_tm_scores) >= 2:
            tm_matches = [(ordered_tm_scores[0], "1"), (ordered_tm_scores[1], "2")]

    if not aligned_match or len(tm_matches) < 2:
        stdout_preview = "\n".join(stdout_text.strip().splitlines()[:20])
        raise ValueError(
            "Could not parse TM-align output. Check the executable and stdout format. "
            f"Stdout preview:\n{stdout_preview}"
        )

    tm_score_by_chain = {int(chain_id): float(score) for score, chain_id in tm_matches}
    return {
        "aligned_length": int(aligned_match.group(1)),
        "rmsd_angstrom": float(aligned_match.group(2)),
        "seq_identity_aligned": float(aligned_match.group(3)),
        "tm_score_to_structure_a": tm_score_by_chain.get(1),
        "tm_score_to_structure_b": tm_score_by_chain.get(2),
    }


def run_tmalign_pair(
    structure_a_path: Path,
    structure_b_path: Path,
    executable_path: str,
) -> dict[str, float | int | None]:
    completed_process = subprocess.run(
        [executable_path, str(structure_a_path), str(structure_b_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed_process.returncode != 0:
        raise RuntimeError(
            completed_process.stderr.strip()
            or completed_process.stdout.strip()
            or "TM-align execution failed"
        )

    parsed_metrics = parse_tmalign_stdout(completed_process.stdout)
    parsed_metrics["raw_stdout"] = completed_process.stdout
    return parsed_metrics


def tm_score_d0(length: int) -> float:
    if length <= 15:
        return 0.5
    return max(0.5, 1.24 * np.cbrt(length - 15) - 1.8)


def kabsch_superpose(reference_xyz: np.ndarray, mobile_xyz: np.ndarray) -> np.ndarray:
    reference_center = reference_xyz.mean(axis=0)
    mobile_center = mobile_xyz.mean(axis=0)

    reference_centered = reference_xyz - reference_center
    mobile_centered = mobile_xyz - mobile_center

    covariance_matrix = mobile_centered.T @ reference_centered
    left_singular_vectors, _, right_singular_vectors_t = np.linalg.svd(covariance_matrix)
    rotation_matrix = right_singular_vectors_t.T @ left_singular_vectors.T

    if np.linalg.det(rotation_matrix) < 0:
        right_singular_vectors_t[-1, :] *= -1
        rotation_matrix = right_singular_vectors_t.T @ left_singular_vectors.T

    return mobile_centered @ rotation_matrix + reference_center


def compute_ca_superposition_metrics(
    structure_a_trace: pd.DataFrame,
    structure_b_trace: pd.DataFrame,
) -> dict[str, float | int | None]:
    paired_trace_df = structure_a_trace.merge(
        structure_b_trace,
        on=["chain_id", "residue_number", "insertion_code"],
        suffixes=("_a", "_b"),
    )
    if len(paired_trace_df) < 3:
        raise ValueError("At least three matched CA residues are required for structural superposition")

    structure_a_xyz = paired_trace_df[["x_a", "y_a", "z_a"]].to_numpy(dtype=float)
    structure_b_xyz = paired_trace_df[["x_b", "y_b", "z_b"]].to_numpy(dtype=float)
    aligned_structure_b_xyz = kabsch_superpose(reference_xyz=structure_a_xyz, mobile_xyz=structure_b_xyz)

    residue_distances = np.linalg.norm(structure_a_xyz - aligned_structure_b_xyz, axis=1)
    rmsd_angstrom = float(np.sqrt(np.mean(np.square(residue_distances))))
    residue_identity = (
        paired_trace_df["residue_name_a"] == paired_trace_df["residue_name_b"]
    ).mean()

    structure_a_length = int(len(structure_a_trace))
    structure_b_length = int(len(structure_b_trace))
    structure_a_d0 = tm_score_d0(structure_a_length)
    structure_b_d0 = tm_score_d0(structure_b_length)

    tm_score_to_structure_a = float(
        np.mean(1.0 / (1.0 + np.square(residue_distances / structure_a_d0)))
    )
    tm_score_to_structure_b = float(
        np.mean(1.0 / (1.0 + np.square(residue_distances / structure_b_d0)))
    )

    return {
        "aligned_length": int(len(paired_trace_df)),
        "rmsd_angstrom": rmsd_angstrom,
        "seq_identity_aligned": float(residue_identity),
        "tm_score_to_structure_a": tm_score_to_structure_a,
        "tm_score_to_structure_b": tm_score_to_structure_b,
        "aligned_fraction_to_structure_a": float(len(paired_trace_df) / max(1, structure_a_length)),
        "aligned_fraction_to_structure_b": float(len(paired_trace_df) / max(1, structure_b_length)),
    }


def compute_wildtype_similarity_table(
    structure_family_df: pd.DataFrame,
    preferred_executable: str | None = None,
    allow_ca_fallback: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    executable_path = resolve_tmalign_executable(preferred_executable)
    wildtype_structure_df = (
        structure_family_df.loc[
            structure_family_df["is_wildtype"],
            ["family_id", "sample_name", "pdb_path"],
        ]
        .sort_values("family_id")
        .reset_index(drop=True)
    )
    wildtype_trace_cache = {
        row["family_id"]: parse_ca_trace(row["pdb_path"])
        for _, row in wildtype_structure_df.iterrows()
    }
    similarity_rows = []

    for first_index in range(len(wildtype_structure_df)):
        first_row = wildtype_structure_df.iloc[first_index]
        for second_index in range(first_index + 1, len(wildtype_structure_df)):
            second_row = wildtype_structure_df.iloc[second_index]

            base_row = {
                "wildtype_a": first_row["family_id"],
                "wildtype_b": second_row["family_id"],
                "wildtype_a_pdb_path": first_row["pdb_path"],
                "wildtype_b_pdb_path": second_row["pdb_path"],
                "wildtype_a_length_ca": int(len(wildtype_trace_cache[first_row["family_id"]])),
                "wildtype_b_length_ca": int(len(wildtype_trace_cache[second_row["family_id"]])),
            }

            try:
                if executable_path:
                    metric_row = run_tmalign_pair(
                        structure_a_path=first_row["pdb_path"],
                        structure_b_path=second_row["pdb_path"],
                        executable_path=executable_path,
                    )
                    alignment_method = "official_tmalign"
                    alignment_note = "Official TM-align/USalign executable"
                elif allow_ca_fallback:
                    metric_row = compute_ca_superposition_metrics(
                        structure_a_trace=wildtype_trace_cache[first_row["family_id"]],
                        structure_b_trace=wildtype_trace_cache[second_row["family_id"]],
                    )
                    alignment_method = "ca_superposition_fallback"
                    alignment_note = (
                        "CA-based TM-score approximation after Kabsch superposition. "
                        "Use TM-align/USalign for biological interpretation."
                    )
                else:
                    metric_row = {
                        "aligned_length": np.nan,
                        "rmsd_angstrom": np.nan,
                        "seq_identity_aligned": np.nan,
                        "tm_score_to_structure_a": np.nan,
                        "tm_score_to_structure_b": np.nan,
                    }
                    alignment_method = "not_computed"
                    alignment_note = "No TM-align executable found and fallback disabled"
            except Exception as exc:
                if allow_ca_fallback:
                    metric_row = compute_ca_superposition_metrics(
                        structure_a_trace=wildtype_trace_cache[first_row["family_id"]],
                        structure_b_trace=wildtype_trace_cache[second_row["family_id"]],
                    )
                    alignment_method = "ca_superposition_fallback"
                    alignment_note = (
                        "Fallback used because TM-align failed: "
                        f"{exc}. Use TM-align/USalign for biological interpretation."
                    )
                else:
                    metric_row = {
                        "aligned_length": np.nan,
                        "rmsd_angstrom": np.nan,
                        "seq_identity_aligned": np.nan,
                        "tm_score_to_structure_a": np.nan,
                        "tm_score_to_structure_b": np.nan,
                    }
                    alignment_method = "failed"
                    alignment_note = str(exc)

            similarity_rows.append(
                {
                    **base_row,
                    "aligned_length": metric_row.get("aligned_length"),
                    "rmsd_angstrom": metric_row.get("rmsd_angstrom"),
                    "seq_identity_aligned": metric_row.get("seq_identity_aligned"),
                    "tm_score_to_wildtype_a": metric_row.get("tm_score_to_structure_a"),
                    "tm_score_to_wildtype_b": metric_row.get("tm_score_to_structure_b"),
                    "mean_tm_score": float(
                        np.nanmean(
                            [
                                metric_row.get("tm_score_to_structure_a"),
                                metric_row.get("tm_score_to_structure_b"),
                            ]
                        )
                    ),
                    "alignment_method": alignment_method,
                    "alignment_note": alignment_note,
                }
            )

    if not similarity_rows:
        empty_columns = [
            "wildtype_a",
            "wildtype_b",
            "wildtype_a_pdb_path",
            "wildtype_b_pdb_path",
            "wildtype_a_length_ca",
            "wildtype_b_length_ca",
            "aligned_length",
            "rmsd_angstrom",
            "seq_identity_aligned",
            "tm_score_to_wildtype_a",
            "tm_score_to_wildtype_b",
            "mean_tm_score",
            "alignment_method",
            "alignment_note",
        ]
        return pd.DataFrame(columns=empty_columns), executable_path

    wildtype_similarity_df = pd.DataFrame(similarity_rows).sort_values(
        ["mean_tm_score", "wildtype_a", "wildtype_b"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return wildtype_similarity_df, executable_path


def build_symmetric_pair_metric_matrix(
    pairwise_df: pd.DataFrame,
    value_column: str,
    diagonal_value: float,
    labels: list[str],
) -> pd.DataFrame:
    metric_matrix_df = pd.DataFrame(diagonal_value, index=labels, columns=labels, dtype=float)
    for _, row in pairwise_df.iterrows():
        metric_matrix_df.loc[row["wildtype_a"], row["wildtype_b"]] = row[value_column]
        metric_matrix_df.loc[row["wildtype_b"], row["wildtype_a"]] = row[value_column]
    return metric_matrix_df


def plot_pairwise_metric_heatmap(
    metric_matrix_df: pd.DataFrame,
    title: str,
    cmap: str,
    value_format: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    matrix_values = metric_matrix_df.to_numpy(dtype=float)
    image = axis.imshow(matrix_values, cmap=cmap, vmin=vmin, vmax=vmax)

    axis.set_xticks(np.arange(len(metric_matrix_df.columns)))
    axis.set_yticks(np.arange(len(metric_matrix_df.index)))
    axis.set_xticklabels(metric_matrix_df.columns, rotation=45, ha="right")
    axis.set_yticklabels(metric_matrix_df.index)
    axis.set_title(title)

    finite_values = matrix_values[np.isfinite(matrix_values)]
    midpoint = float(np.nanmean(finite_values)) if len(finite_values) else 0.0
    for row_index in range(metric_matrix_df.shape[0]):
        for column_index in range(metric_matrix_df.shape[1]):
            value = matrix_values[row_index, column_index]
            if np.isnan(value):
                label_text = "NA"
                text_color = "black"
            else:
                label_text = value_format.format(value)
                text_color = "white" if value > midpoint else "black"
            axis.text(
                column_index,
                row_index,
                label_text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=9)
    figure.tight_layout()
    return figure, axis


def plot_wildtype_tm_score_boxplot(
    wildtype_similarity_df: pd.DataFrame,
    value_column: str = "mean_tm_score",
    figure_width_mm: float = 60,
    figure_height_mm: float = 55,
    title: str = "Wildtype pairwise TM-scores",
    ylabel: str = "TM-score",
    title_size_pt: float = 5.5,
    label_size_pt: float = 5.5,
    tick_size_pt: float = 5.5,
    strip_marker_size: float = 16,
    box_facecolor: str = "#d8e7f5",
    box_edgecolor: str = "#2166ac",
    point_color: str = "#2166ac",
):
    if wildtype_similarity_df.empty:
        raise ValueError("wildtype_similarity_df is empty")
    if value_column not in wildtype_similarity_df.columns:
        raise ValueError(f"'{value_column}' is not present in wildtype_similarity_df")

    tm_scores = wildtype_similarity_df[value_column].dropna().to_numpy(dtype=float)
    if not len(tm_scores):
        raise ValueError(f"No finite TM-score values were found in column '{value_column}'")

    figure_width_inch, figure_height_inch = mm_to_inches(figure_width_mm, figure_height_mm)
    figure, axis = plt.subplots(figsize=(figure_width_inch, figure_height_inch), constrained_layout=True)

    axis.boxplot(
        [tm_scores],
        positions=[1.0],
        widths=0.32,
        patch_artist=True,
        boxprops={"facecolor": box_facecolor, "edgecolor": box_edgecolor, "linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )

    if len(tm_scores) == 1:
        jitter_offsets = np.array([0.0], dtype=float)
    else:
        jitter_offsets = np.linspace(-0.07, 0.07, len(tm_scores), dtype=float)

    axis.scatter(
        1.0 + jitter_offsets,
        tm_scores,
        s=strip_marker_size,
        color=point_color,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.9,
        zorder=3,
    )

    y_padding = max(0.01, 0.08 * float(tm_scores.max() - tm_scores.min() if len(tm_scores) > 1 else 0.05))
    axis.set_ylim(max(0.0, float(tm_scores.min()) - y_padding), min(1.01, float(tm_scores.max()) + y_padding))
    axis.set_xlim(0.6, 1.4)
    axis.set_xticks([1.0])
    axis.set_xticklabels([f"WT pairs\n(n={len(tm_scores)})"], fontsize=tick_size_pt)
    axis.set_ylabel(ylabel, fontsize=label_size_pt)
    axis.set_title(title, fontsize=title_size_pt)
    axis.tick_params(axis="y", labelsize=tick_size_pt)
    axis.grid(axis="y", alpha=0.18, linewidth=0.5)

    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        axis.spines[spine].set_linewidth(0.6)

    return figure, axis


def discover_md_rmsf_catalog(
    md_root: Path,
    allowed_family_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    allowed_family_id_set = set(allowed_family_ids) if allowed_family_ids is not None else None
    catalog_rows = []

    for family_dir in sorted(path for path in md_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        family_id = family_dir.name
        if allowed_family_id_set is not None and family_id not in allowed_family_id_set:
            continue

        for sample_dir in sorted(path for path in family_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            rmsf_paths = sorted(sample_dir.glob("rmsf_*.csv"))
            if not rmsf_paths:
                continue

            rmsf_path = rmsf_paths[-1]
            stem_parts = rmsf_path.stem.split("_")
            window_start_frame = int(stem_parts[1]) if len(stem_parts) >= 3 and stem_parts[1].isdigit() else np.nan
            window_stop_frame = int(stem_parts[2]) if len(stem_parts) >= 3 and stem_parts[2].isdigit() else np.nan
            is_wildtype = sample_dir.name in {family_id, f"{family_id}_WT"}
            aligned_sample_name = family_id if is_wildtype else sample_dir.name

            catalog_rows.append(
                {
                    "family_id": family_id,
                    "sample_name": aligned_sample_name,
                    "md_sample_name": sample_dir.name,
                    "is_wildtype": is_wildtype,
                    "rmsf_path": rmsf_path,
                    "window_start_frame": window_start_frame,
                    "window_stop_frame": window_stop_frame,
                }
            )

    if not catalog_rows:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "md_sample_name",
                "is_wildtype",
                "rmsf_path",
                "window_start_frame",
                "window_stop_frame",
            ]
        )

    return pd.DataFrame(catalog_rows).sort_values(
        ["family_id", "is_wildtype", "sample_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def load_md_rmsf_long_df(md_rmsf_catalog_df: pd.DataFrame) -> pd.DataFrame:
    if md_rmsf_catalog_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "md_sample_name",
                "is_wildtype",
                "residue_number",
                "rmsf_angstrom",
            ]
        )

    profile_tables = []
    for row in md_rmsf_catalog_df.itertuples(index=False):
        profile_df = pd.read_csv(row.rmsf_path)
        if "Frame" not in profile_df.columns or "RMSF" not in profile_df.columns:
            raise ValueError(f"Unexpected RMSF file format: {row.rmsf_path}")

        profile_df = profile_df.rename(columns={"Frame": "residue_number", "RMSF": "rmsf_angstrom"})
        profile_df["residue_number"] = pd.to_numeric(profile_df["residue_number"], errors="coerce").round().astype("Int64")
        profile_df["rmsf_angstrom"] = pd.to_numeric(profile_df["rmsf_angstrom"], errors="coerce")
        profile_df = profile_df.dropna(subset=["residue_number", "rmsf_angstrom"]).copy()
        profile_df["residue_number"] = profile_df["residue_number"].astype(int)
        profile_df["family_id"] = row.family_id
        profile_df["sample_name"] = row.sample_name
        profile_df["md_sample_name"] = row.md_sample_name
        profile_df["is_wildtype"] = bool(row.is_wildtype)
        profile_tables.append(
            profile_df[
                [
                    "family_id",
                    "sample_name",
                    "md_sample_name",
                    "is_wildtype",
                    "residue_number",
                    "rmsf_angstrom",
                ]
            ]
        )

    return pd.concat(profile_tables, ignore_index=True).sort_values(
        ["family_id", "is_wildtype", "sample_name", "residue_number"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def generate_structure_sample_aliases(sample_name: str, family_id: str) -> set[str]:
    aliases = {sample_name}

    if sample_name == family_id:
        aliases.add(family_id)
        return aliases

    for suffix in ["_New_17444_0", "_New_19XXX_0"]:
        if sample_name.endswith(suffix):
            aliases.add(sample_name[: -len(suffix)])

    model_match = re.match(rf"^{re.escape(family_id)}_(.+?)_model_x_(\d+)$", sample_name)
    if model_match:
        design_id, replicate_id = model_match.groups()
        aliases.add(f"{family_id}_{design_id}_x_{replicate_id}")
        aliases.add(f"{family_id}_{design_id}")

    return aliases


def build_unique_sample_alias_map(sample_catalog_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    alias_to_samples: dict[tuple[str, str], set[str]] = {}

    for row in sample_catalog_df[["family_id", "sample_name"]].drop_duplicates().itertuples(index=False):
        family_id = row.family_id
        sample_name = row.sample_name
        if pd.isna(sample_name):
            continue
        for alias in generate_structure_sample_aliases(str(sample_name), str(family_id)):
            alias_key = (str(family_id), alias)
            alias_to_samples.setdefault(alias_key, set()).add(str(sample_name))

    return {
        alias_key: next(iter(sample_names))
        for alias_key, sample_names in alias_to_samples.items()
        if len(sample_names) == 1
    }


def infer_family_id_from_annotation(annotation: str, family_ids: Iterable[str]) -> str | None:
    family_id_list = sorted(set(str(family_id) for family_id in family_ids), key=len, reverse=True)
    for family_id in family_id_list:
        if annotation == family_id or annotation.startswith(f"{family_id}_"):
            return family_id
    return None


def load_tsa_tm_delta_summary(
    result_summary_path: Path,
    sample_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    tsa_df = pd.read_csv(result_summary_path)
    tsa_df = tsa_df.rename(columns={column_name: column_name.lstrip("#") for column_name in tsa_df.columns})

    required_columns = {"Annotation", "Tm"}
    missing_columns = required_columns.difference(tsa_df.columns)
    if missing_columns:
        missing_column_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"TSA summary is missing required columns: {missing_column_text}")

    tsa_df["Annotation"] = tsa_df["Annotation"].astype(str).str.strip()
    tsa_df["Tm"] = pd.to_numeric(tsa_df["Tm"], errors="coerce")
    tsa_df = tsa_df.dropna(subset=["Tm"]).copy()
    tsa_df = tsa_df.loc[~tsa_df["Annotation"].isin({"BLANK", "N/A", ""})].copy()

    family_ids = sample_catalog_df["family_id"].dropna().astype(str).unique().tolist()
    tsa_df["family_id"] = tsa_df["Annotation"].map(
        lambda annotation: infer_family_id_from_annotation(annotation, family_ids)
    )
    tsa_df = tsa_df.dropna(subset=["family_id"]).copy()

    tm_summary_df = (
        tsa_df.groupby(["family_id", "Annotation"], as_index=False)
        .agg(
            n_replicates=("Tm", "size"),
            median_tm_celsius=("Tm", "median"),
            mean_tm_celsius=("Tm", "mean"),
            tm_std_celsius=("Tm", "std"),
        )
        .rename(columns={"Annotation": "assay_annotation"})
    )

    wildtype_tm_df = tm_summary_df.loc[
        tm_summary_df["assay_annotation"] == tm_summary_df["family_id"],
        ["family_id", "median_tm_celsius"],
    ].rename(columns={"median_tm_celsius": "wildtype_median_tm_celsius"})

    tm_summary_df = tm_summary_df.merge(wildtype_tm_df, on="family_id", how="left")
    tm_summary_df["delta_tm_to_wildtype_celsius"] = (
        tm_summary_df["median_tm_celsius"] - tm_summary_df["wildtype_median_tm_celsius"]
    )

    alias_map = build_unique_sample_alias_map(sample_catalog_df)
    tm_summary_df["sample_name"] = tm_summary_df.apply(
        lambda row: alias_map.get((str(row["family_id"]), str(row["assay_annotation"]))),
        axis=1,
    )
    tm_summary_df["is_wildtype"] = tm_summary_df["assay_annotation"] == tm_summary_df["family_id"]

    return tm_summary_df.sort_values(
        ["family_id", "is_wildtype", "assay_annotation"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_tm_delta_row_label_map(tm_summary_df: pd.DataFrame) -> dict[str, str]:
    if tm_summary_df.empty:
        return {}

    row_label_map = {}
    variant_tm_summary_df = tm_summary_df.loc[
        tm_summary_df["sample_name"].notna() & (~tm_summary_df["is_wildtype"])
    ].copy()

    for row in variant_tm_summary_df.itertuples(index=False):
        short_label = short_variant_label(row.sample_name, row.family_id)
        delta_tm = row.delta_tm_to_wildtype_celsius
        if pd.isna(delta_tm):
            row_label_map[row.sample_name] = short_label
        else:
            row_label_map[row.sample_name] = f"{short_label}  (dTm {float(delta_tm):+0.1f} C)"

    return row_label_map


def build_tm_median_sort_map(tm_summary_df: pd.DataFrame) -> dict[str, float]:
    if tm_summary_df.empty:
        return {}

    variant_tm_summary_df = tm_summary_df.loc[
        tm_summary_df["sample_name"].notna() & (~tm_summary_df["is_wildtype"])
    ].copy()

    return {
        row.sample_name: float(row.median_tm_celsius)
        for row in variant_tm_summary_df.itertuples(index=False)
        if pd.notna(row.median_tm_celsius)
    }


def filter_tm_summary_by_minimum_melting_temperature(
    tm_summary_df: pd.DataFrame,
    minimum_median_tm_celsius: float | None = None,
    exclude_missing_tm: bool = False,
) -> pd.DataFrame:
    if tm_summary_df.empty or minimum_median_tm_celsius is None:
        return tm_summary_df.copy()

    filtered_tm_summary_df = tm_summary_df.copy()
    variant_keep_mask = filtered_tm_summary_df["median_tm_celsius"] >= float(minimum_median_tm_celsius)
    if not exclude_missing_tm:
        variant_keep_mask = variant_keep_mask | filtered_tm_summary_df["median_tm_celsius"].isna()

    keep_mask = filtered_tm_summary_df["is_wildtype"] | variant_keep_mask
    return filtered_tm_summary_df.loc[keep_mask].copy().reset_index(drop=True)


def filter_rmsf_to_allowed_variants(
    md_rmsf_df: pd.DataFrame,
    variant_delta_rmsf_df: pd.DataFrame,
    allowed_variant_names: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_variant_name_set = {str(sample_name) for sample_name in allowed_variant_names if pd.notna(sample_name)}

    filtered_md_rmsf_df = md_rmsf_df.loc[
        md_rmsf_df["is_wildtype"] | md_rmsf_df["sample_name"].isin(allowed_variant_name_set)
    ].copy()
    filtered_variant_delta_rmsf_df = variant_delta_rmsf_df.loc[
        variant_delta_rmsf_df["sample_name"].isin(allowed_variant_name_set)
    ].copy()

    return (
        filtered_md_rmsf_df.reset_index(drop=True),
        filtered_variant_delta_rmsf_df.reset_index(drop=True),
    )


def compute_variant_delta_rmsf_df(md_rmsf_df: pd.DataFrame) -> pd.DataFrame:
    if md_rmsf_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "residue_number",
                "rmsf_angstrom",
                "wildtype_rmsf_angstrom",
                "delta_rmsf_angstrom",
                "abs_delta_rmsf_angstrom",
            ]
        )

    wildtype_rmsf_df = md_rmsf_df.loc[
        md_rmsf_df["is_wildtype"],
        ["family_id", "residue_number", "rmsf_angstrom"],
    ].rename(columns={"rmsf_angstrom": "wildtype_rmsf_angstrom"})

    variant_delta_rmsf_df = md_rmsf_df.loc[~md_rmsf_df["is_wildtype"]].merge(
        wildtype_rmsf_df,
        on=["family_id", "residue_number"],
        how="inner",
    )
    variant_delta_rmsf_df["delta_rmsf_angstrom"] = (
        variant_delta_rmsf_df["rmsf_angstrom"] - variant_delta_rmsf_df["wildtype_rmsf_angstrom"]
    )
    variant_delta_rmsf_df["abs_delta_rmsf_angstrom"] = variant_delta_rmsf_df["delta_rmsf_angstrom"].abs()
    return variant_delta_rmsf_df.sort_values(
        ["family_id", "sample_name", "residue_number"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def filter_variant_delta_rmsf_df(
    variant_delta_rmsf_df: pd.DataFrame,
    selected_variants_by_family: dict[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    if selected_variants_by_family is None:
        return variant_delta_rmsf_df.copy()

    selected_rows = []
    for family_id, variant_names in selected_variants_by_family.items():
        for variant_name in variant_names:
            selected_rows.append({"family_id": family_id, "sample_name": str(variant_name)})

    if not selected_rows:
        return variant_delta_rmsf_df.iloc[0:0].copy()

    selected_variant_df = pd.DataFrame(selected_rows).drop_duplicates()
    return selected_variant_df.merge(
        variant_delta_rmsf_df,
        on=["family_id", "sample_name"],
        how="inner",
    ).sort_values(
        ["family_id", "sample_name", "residue_number"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def plot_family_rmsf_overview(
    family_id: str,
    md_rmsf_df: pd.DataFrame,
    variant_delta_rmsf_df: pd.DataFrame,
    variant_rmsf_site_annotation_df: pd.DataFrame | None = None,
    selected_variant_names: Iterable[str] | None = None,
    sample_row_label_map: dict[str, str] | None = None,
    sample_sort_value_map: dict[str, float] | None = None,
    figure_width_mm: float = 150,
    figure_height_mm: float = 100,
    title_size_pt: float = 5.5,
    label_size_pt: float = 5.5,
    tick_size_pt: float = 5.5,
    legend_size_pt: float = 5.5,
    delta_cmap=DEFAULT_RMSF_DELTA_CMAP,
):
    family_rmsf_df = md_rmsf_df.loc[md_rmsf_df["family_id"] == family_id].copy()
    family_delta_df = variant_delta_rmsf_df.loc[variant_delta_rmsf_df["family_id"] == family_id].copy()

    if variant_rmsf_site_annotation_df is None:
        family_site_df = pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "residue_number",
                "substitution_label",
                "candidate_interactions",
                "is_interesting_site",
            ]
        )
    else:
        family_site_df = variant_rmsf_site_annotation_df.loc[
            variant_rmsf_site_annotation_df["family_id"] == family_id
        ].copy()

    if family_rmsf_df.empty:
        raise ValueError(f"No RMSF data found for family '{family_id}'")

    wildtype_rmsf_df = family_rmsf_df.loc[family_rmsf_df["is_wildtype"]].sort_values("residue_number")
    variant_profile_df = family_rmsf_df.loc[~family_rmsf_df["is_wildtype"]].copy()
    if wildtype_rmsf_df.empty:
        raise ValueError(f"No wildtype RMSF profile found for family '{family_id}'")
    if family_delta_df.empty:
        raise ValueError(f"No variant RMSF profiles found for family '{family_id}' after filtering")

    variant_summary_df = (
        family_delta_df.groupby("sample_name", as_index=False)
        .agg(
            mean_abs_delta_rmsf_angstrom=("abs_delta_rmsf_angstrom", "mean"),
            max_abs_delta_rmsf_angstrom=("abs_delta_rmsf_angstrom", "max"),
        )
        .sort_values(
            ["mean_abs_delta_rmsf_angstrom", "max_abs_delta_rmsf_angstrom", "sample_name"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    highlight_ranking_df = variant_summary_df.sort_values(
        ["mean_abs_delta_rmsf_angstrom", "max_abs_delta_rmsf_angstrom", "sample_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    sample_sort_value_map = sample_sort_value_map or {}
    if sample_sort_value_map:
        variant_summary_df["sample_sort_value"] = variant_summary_df["sample_name"].map(sample_sort_value_map)
        variant_summary_df["has_sample_sort_value"] = variant_summary_df["sample_sort_value"].notna()
        variant_summary_df = variant_summary_df.sort_values(
            [
                "has_sample_sort_value",
                "sample_sort_value",
                "mean_abs_delta_rmsf_angstrom",
                "max_abs_delta_rmsf_angstrom",
                "sample_name",
            ],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

    variant_order = list(variant_summary_df["sample_name"])
    if selected_variant_names is None:
        highlighted_variant_names = list(highlight_ranking_df["sample_name"][: min(4, len(highlight_ranking_df))])
    else:
        available_variant_names = set(variant_summary_df["sample_name"])
        highlighted_variant_names = [name for name in selected_variant_names if name in available_variant_names]

    if not variant_order:
        raise ValueError(f"No variant RMSF profiles remain for family '{family_id}'")

    sample_row_label_map = sample_row_label_map or {}
    highlighted_color_values = plt.cm.Set2(np.linspace(0, 1, max(len(highlighted_variant_names), 1)))
    highlighted_color_map = {
        sample_name: highlighted_color_values[idx]
        for idx, sample_name in enumerate(highlighted_variant_names)
    }

    variant_quantile_df = (
        variant_profile_df.groupby("residue_number")["rmsf_angstrom"]
        .agg(
            q25=lambda values: values.quantile(0.25),
            median="median",
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )

    heatmap_df = (
        family_delta_df.pivot(index="sample_name", columns="residue_number", values="delta_rmsf_angstrom")
        .reindex(index=variant_order)
    )
    residue_numbers = heatmap_df.columns.to_numpy(dtype=int)
    delta_matrix = heatmap_df.to_numpy(dtype=float)
    if not len(residue_numbers):
        raise ValueError(f"No residue-wise delta RMSF matrix could be built for family '{family_id}'")

    rmsf_max = float(
        np.nanmax(
            np.concatenate(
                [
                    wildtype_rmsf_df["rmsf_angstrom"].to_numpy(dtype=float),
                    variant_profile_df["rmsf_angstrom"].to_numpy(dtype=float),
                ]
            )
        )
    )
    delta_abs_values = np.abs(delta_matrix[np.isfinite(delta_matrix)])
    delta_color_limit = float(np.nanquantile(delta_abs_values, 0.97)) if len(delta_abs_values) else 1.0
    delta_color_limit = max(delta_color_limit, 0.25)

    figure_width_inch, figure_height_inch = mm_to_inches(figure_width_mm, figure_height_mm)
    figure = plt.figure(figsize=(figure_width_inch, figure_height_inch), constrained_layout=True)
    grid_spec = figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.25], hspace=0.04)
    profile_axis = figure.add_subplot(grid_spec[0, 0])
    heatmap_axis = figure.add_subplot(grid_spec[1, 0], sharex=profile_axis)

    profile_axis.fill_between(
        variant_quantile_df["residue_number"],
        variant_quantile_df["q25"],
        variant_quantile_df["q75"],
        color="#d9d9d9",
        alpha=0.55,
        linewidth=0.0,
        label="Variant IQR",
    )
    profile_axis.plot(
        variant_quantile_df["residue_number"],
        variant_quantile_df["median"],
        color="#c2185b",
        linewidth=1.0,
        linestyle="--",
        label="Variant median",
    )
    profile_axis.plot(
        wildtype_rmsf_df["residue_number"],
        wildtype_rmsf_df["rmsf_angstrom"],
        color="black",
        linewidth=1.15,
        label="Wildtype",
    )

    for sample_name in variant_order:
        sample_profile_df = variant_profile_df.loc[
            variant_profile_df["sample_name"] == sample_name
        ].sort_values("residue_number")
        line_color = highlighted_color_map.get(sample_name, "#bdbdbd")
        line_alpha = 0.95 if sample_name in highlighted_color_map else 0.28
        line_width = 0.9 if sample_name in highlighted_color_map else 0.45
        line_label = short_variant_label(sample_name, family_id) if sample_name in highlighted_color_map else None
        profile_axis.plot(
            sample_profile_df["residue_number"],
            sample_profile_df["rmsf_angstrom"],
            color=line_color,
            alpha=line_alpha,
            linewidth=line_width,
            label=line_label,
            zorder=3 if sample_name in highlighted_color_map else 1,
        )

    if not family_site_df.empty and "is_interesting_site" in family_site_df.columns:
        interesting_family_sites = sorted(
            family_site_df.loc[family_site_df["is_interesting_site"], "residue_number"].unique()
        )
    else:
        interesting_family_sites = []

    if interesting_family_sites:
        profile_axis.scatter(
            interesting_family_sites,
            np.full(len(interesting_family_sites), rmsf_max * 1.03),
            s=18,
            marker="v",
            color="#f2a900",
            edgecolors="black",
            linewidths=0.35,
            clip_on=False,
            label="Interaction candidate site",
        )

    profile_axis.set_ylim(0.0, rmsf_max * 1.12)
    profile_axis.set_ylabel("RMSF (A)", fontsize=label_size_pt)
    profile_axis.set_title(f"{family_id}: RMSF profile and delta to wildtype", fontsize=title_size_pt)
    profile_axis.tick_params(axis="both", labelsize=tick_size_pt)
    profile_axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    profile_axis.legend(loc="upper right", fontsize=legend_size_pt, frameon=False, ncol=2)
    for spine in ["top", "right"]:
        profile_axis.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        profile_axis.spines[spine].set_linewidth(0.6)

    heatmap_image = heatmap_axis.imshow(
        delta_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=delta_cmap,
        norm=TwoSlopeNorm(vmin=-delta_color_limit, vcenter=0.0, vmax=delta_color_limit),
        extent=[residue_numbers.min() - 0.5, residue_numbers.max() + 0.5, -0.5, len(variant_order) - 0.5],
        origin="lower",
    )
    heatmap_axis.invert_yaxis()

    if not family_site_df.empty:
        row_lookup = {sample_name: row_index for row_index, sample_name in enumerate(variant_order)}
        site_marker_df = family_site_df.loc[family_site_df["sample_name"].isin(set(row_lookup))].copy()
        regular_site_df = site_marker_df.loc[~site_marker_df.get("is_interesting_site", False)]
        interesting_site_df = site_marker_df.loc[site_marker_df.get("is_interesting_site", False)]

        if not regular_site_df.empty:
            heatmap_axis.scatter(
                regular_site_df["residue_number"],
                regular_site_df["sample_name"].map(row_lookup),
                s=12,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )
        if not interesting_site_df.empty:
            heatmap_axis.scatter(
                interesting_site_df["residue_number"],
                interesting_site_df["sample_name"].map(row_lookup),
                s=36,
                marker="*",
                color="#f2a900",
                edgecolors="black",
                linewidths=0.35,
                zorder=4,
            )

    heatmap_axis.set_yticks(np.arange(len(variant_order)))
    heatmap_axis.set_yticklabels(
        [sample_row_label_map.get(sample_name, short_variant_label(sample_name, family_id)) for sample_name in variant_order],
        fontsize=tick_size_pt,
    )
    heatmap_axis.set_ylabel("Variants", fontsize=label_size_pt)
    heatmap_axis.set_xlabel("Residue number", fontsize=label_size_pt)
    heatmap_axis.tick_params(axis="x", labelsize=tick_size_pt)
    heatmap_axis.tick_params(axis="y", labelsize=tick_size_pt)
    heatmap_axis.set_xticks(choose_residue_ticks(residue_numbers))
    for spine in ["top", "right"]:
        heatmap_axis.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        heatmap_axis.spines[spine].set_linewidth(0.6)

    colorbar = figure.colorbar(heatmap_image, ax=heatmap_axis, fraction=0.02, pad=0.02)
    colorbar.set_label("Variant - WT RMSF (A)", fontsize=label_size_pt)
    colorbar.ax.tick_params(labelsize=tick_size_pt)

    return figure


def build_variant_pair_summary_df(variant_delta_rmsf_df: pd.DataFrame) -> pd.DataFrame:
    if variant_delta_rmsf_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "n_residues",
                "mean_delta_rmsf_angstrom",
                "mean_abs_delta_rmsf_angstrom",
                "max_abs_delta_rmsf_angstrom",
                "peak_residue_number",
                "peak_delta_rmsf_angstrom",
            ]
        )

    summary_rows = []
    for (family_id, sample_name), pair_df in variant_delta_rmsf_df.groupby(["family_id", "sample_name"], sort=True):
        peak_row = pair_df.loc[pair_df["abs_delta_rmsf_angstrom"].idxmax()]
        summary_rows.append(
            {
                "family_id": family_id,
                "sample_name": sample_name,
                "n_residues": int(len(pair_df)),
                "mean_delta_rmsf_angstrom": float(pair_df["delta_rmsf_angstrom"].mean()),
                "mean_abs_delta_rmsf_angstrom": float(pair_df["abs_delta_rmsf_angstrom"].mean()),
                "max_abs_delta_rmsf_angstrom": float(pair_df["abs_delta_rmsf_angstrom"].max()),
                "peak_residue_number": int(peak_row["residue_number"]),
                "peak_delta_rmsf_angstrom": float(peak_row["delta_rmsf_angstrom"]),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["family_id", "mean_abs_delta_rmsf_angstrom", "sample_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def compute_replica_rmsf_statistics(replica_rmsf_df: pd.DataFrame) -> pd.DataFrame:
    if replica_rmsf_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "is_wildtype",
                "residue_number",
                "mean_rmsf_angstrom",
                "std_rmsf_angstrom",
                "min_rmsf_angstrom",
                "max_rmsf_angstrom",
                "n_replicates",
            ]
        )

    required_columns = {
        "family_id",
        "sample_name",
        "is_wildtype",
        "residue_number",
        "rmsf_angstrom",
    }
    missing_columns = required_columns.difference(replica_rmsf_df.columns)
    if missing_columns:
        missing_column_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Replica RMSF table is missing required columns: "
            f"{missing_column_text}"
        )

    replica_stats_df = (
        replica_rmsf_df.groupby(
            ["family_id", "sample_name", "is_wildtype", "residue_number"],
            as_index=False,
        )
        .agg(
            mean_rmsf_angstrom=("rmsf_angstrom", "mean"),
            std_rmsf_angstrom=("rmsf_angstrom", "std"),
            min_rmsf_angstrom=("rmsf_angstrom", "min"),
            max_rmsf_angstrom=("rmsf_angstrom", "max"),
            n_replicates=("rmsf_angstrom", "size"),
        )
        .sort_values(
            ["family_id", "is_wildtype", "sample_name", "residue_number"],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )
    replica_stats_df["std_rmsf_angstrom"] = replica_stats_df["std_rmsf_angstrom"].fillna(0.0)
    return replica_stats_df


def build_replica_variant_pair_summary_df(
    replica_rmsf_stats_df: pd.DataFrame,
    variant_metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if replica_rmsf_stats_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "sample_name",
                "n_residues",
                "n_replicates",
                "mean_delta_rmsf_angstrom",
                "mean_abs_delta_rmsf_angstrom",
                "max_abs_delta_rmsf_angstrom",
                "peak_residue_number",
                "peak_delta_rmsf_angstrom",
                "median_tm_celsius",
                "delta_tm_to_wildtype_celsius",
            ]
        )

    wildtype_stats_df = replica_rmsf_stats_df.loc[
        replica_rmsf_stats_df["is_wildtype"],
        ["family_id", "residue_number", "mean_rmsf_angstrom"],
    ].rename(columns={"mean_rmsf_angstrom": "wildtype_mean_rmsf_angstrom"})

    replica_delta_df = replica_rmsf_stats_df.loc[
        ~replica_rmsf_stats_df["is_wildtype"]
    ].merge(
        wildtype_stats_df,
        on=["family_id", "residue_number"],
        how="inner",
    )
    replica_delta_df["delta_rmsf_angstrom"] = (
        replica_delta_df["mean_rmsf_angstrom"] - replica_delta_df["wildtype_mean_rmsf_angstrom"]
    )
    replica_delta_df["abs_delta_rmsf_angstrom"] = replica_delta_df["delta_rmsf_angstrom"].abs()

    summary_rows = []
    for (family_id, sample_name), pair_df in replica_delta_df.groupby(
        ["family_id", "sample_name"],
        sort=True,
    ):
        peak_row = pair_df.loc[pair_df["abs_delta_rmsf_angstrom"].idxmax()]
        summary_rows.append(
            {
                "family_id": family_id,
                "sample_name": sample_name,
                "n_residues": int(len(pair_df)),
                "n_replicates": int(pair_df["n_replicates"].min()),
                "mean_delta_rmsf_angstrom": float(pair_df["delta_rmsf_angstrom"].mean()),
                "mean_abs_delta_rmsf_angstrom": float(pair_df["abs_delta_rmsf_angstrom"].mean()),
                "max_abs_delta_rmsf_angstrom": float(pair_df["abs_delta_rmsf_angstrom"].max()),
                "peak_residue_number": int(peak_row["residue_number"]),
                "peak_delta_rmsf_angstrom": float(peak_row["delta_rmsf_angstrom"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return summary_df

    if variant_metadata_df is not None and not variant_metadata_df.empty:
        metadata_columns = [
            column_name
            for column_name in [
                "family_id",
                "sample_name",
                "median_tm_celsius",
                "delta_tm_to_wildtype_celsius",
            ]
            if column_name in variant_metadata_df.columns
        ]
        if metadata_columns:
            summary_df = summary_df.merge(
                variant_metadata_df[metadata_columns].drop_duplicates(
                    subset=["family_id", "sample_name"]
                ),
                on=["family_id", "sample_name"],
                how="left",
            )

    sort_columns = ["family_id", "mean_abs_delta_rmsf_angstrom", "sample_name"]
    ascending = [True, False, True]
    if "delta_tm_to_wildtype_celsius" in summary_df.columns:
        sort_columns = [
            "family_id",
            "delta_tm_to_wildtype_celsius",
            "mean_abs_delta_rmsf_angstrom",
            "sample_name",
        ]
        ascending = [True, False, False, True]

    return summary_df.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def plot_family_replica_rmsf_grid(
    family_id: str,
    replica_rmsf_df: pd.DataFrame,
    variant_metadata_df: pd.DataFrame | None = None,
    selected_variant_names: Iterable[str] | None = None,
    figure_width_mm: float = 180,
    panel_height_mm: float = 36,
    max_columns: int = 3,
    title_size_pt: float = 6.4,
    label_size_pt: float = 5.8,
    tick_size_pt: float = 5.2,
):
    family_replica_df = replica_rmsf_df.loc[replica_rmsf_df["family_id"] == family_id].copy()
    if family_replica_df.empty:
        raise ValueError(f"No replica RMSF data found for family '{family_id}'")

    wildtype_replica_df = family_replica_df.loc[family_replica_df["is_wildtype"]].copy()
    if wildtype_replica_df.empty:
        raise ValueError(f"No wildtype replica RMSF profile found for family '{family_id}'")

    family_stats_df = compute_replica_rmsf_statistics(family_replica_df)
    wildtype_stats_df = family_stats_df.loc[family_stats_df["is_wildtype"]].sort_values("residue_number")
    family_summary_df = build_replica_variant_pair_summary_df(
        family_stats_df,
        variant_metadata_df=variant_metadata_df,
    )
    family_summary_df = family_summary_df.loc[family_summary_df["family_id"] == family_id].copy()
    if family_summary_df.empty:
        raise ValueError(f"No variant replica RMSF profiles found for family '{family_id}'")

    if selected_variant_names is not None:
        available_variant_name_set = set(family_summary_df["sample_name"])
        variant_order = [sample_name for sample_name in selected_variant_names if sample_name in available_variant_name_set]
        family_summary_df = family_summary_df.set_index("sample_name").loc[variant_order].reset_index()
    else:
        family_summary_df = family_summary_df.sort_values(
            [
                "delta_tm_to_wildtype_celsius" if "delta_tm_to_wildtype_celsius" in family_summary_df.columns else "mean_abs_delta_rmsf_angstrom",
                "mean_abs_delta_rmsf_angstrom",
                "sample_name",
            ],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        variant_order = family_summary_df["sample_name"].tolist()

    if not variant_order:
        raise ValueError(f"No variant replica RMSF profiles remain for family '{family_id}'")

    if len(variant_order) == 1:
        variant_color_map = {variant_order[0]: "#c2185b"}
    else:
        color_values = plt.cm.viridis(np.linspace(0.15, 0.9, len(variant_order)))
        variant_color_map = {
            sample_name: color_values[index]
            for index, sample_name in enumerate(variant_order)
        }

    residue_numbers = wildtype_stats_df["residue_number"].to_numpy(dtype=int)
    x_ticks = choose_residue_ticks(residue_numbers)
    if len(x_ticks) >= 3:
        typical_tick_step = float(np.median(np.diff(x_ticks[:-1])))
        if typical_tick_step > 0 and float(x_ticks[-1] - x_ticks[-2]) < 0.45 * typical_tick_step:
            x_ticks = np.concatenate([x_ticks[:-2], x_ticks[-1:]])
    rmsf_max = float(family_replica_df["rmsf_angstrom"].max())

    n_variants = len(variant_order)
    n_columns = max(1, min(int(max_columns), n_variants))
    n_rows = int(np.ceil(n_variants / n_columns))
    figure_width_inch, = mm_to_inches(figure_width_mm)
    figure_height_inch, = mm_to_inches(panel_height_mm * n_rows)

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(figure_width_inch, figure_height_inch),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    wt_mean_label = "Wildtype mean"
    wt_replica_label = "Wildtype replicates"
    variant_mean_label = "Variant mean"
    variant_replica_label = "Variant replicates"

    for axis_index, (axis, sample_name) in enumerate(zip(flat_axes, variant_order)):
        variant_replica_df = family_replica_df.loc[family_replica_df["sample_name"] == sample_name].copy()
        variant_stats_df = family_stats_df.loc[
            (family_stats_df["family_id"] == family_id)
            & (family_stats_df["sample_name"] == sample_name)
        ].sort_values("residue_number")
        if variant_replica_df.empty or variant_stats_df.empty:
            raise ValueError(f"No replica RMSF profile found for sample '{sample_name}'")

        summary_row = family_summary_df.loc[family_summary_df["sample_name"] == sample_name].iloc[0]
        variant_color = variant_color_map[sample_name]

        for replicate_index, (_, replicate_df) in enumerate(
            wildtype_replica_df.groupby("replicate", sort=True)
        ):
            replicate_df = replicate_df.sort_values("residue_number")
            axis.plot(
                replicate_df["residue_number"],
                replicate_df["rmsf_angstrom"],
                color="#9e9e9e",
                linewidth=0.55,
                alpha=0.45,
                label=wt_replica_label if axis_index == 0 and replicate_index == 0 else None,
                zorder=1,
            )

        for replicate_index, (_, replicate_df) in enumerate(
            variant_replica_df.groupby("replicate", sort=True)
        ):
            replicate_df = replicate_df.sort_values("residue_number")
            axis.plot(
                replicate_df["residue_number"],
                replicate_df["rmsf_angstrom"],
                color=variant_color,
                linewidth=0.55,
                alpha=0.28,
                label=variant_replica_label if axis_index == 0 and replicate_index == 0 else None,
                zorder=1,
            )

        axis.plot(
            wildtype_stats_df["residue_number"],
            wildtype_stats_df["mean_rmsf_angstrom"],
            color="black",
            linewidth=1.2,
            label=wt_mean_label if axis_index == 0 else None,
            zorder=3,
        )
        axis.plot(
            variant_stats_df["residue_number"],
            variant_stats_df["mean_rmsf_angstrom"],
            color=variant_color,
            linewidth=1.2,
            label=variant_mean_label if axis_index == 0 else None,
            zorder=4,
        )

        peak_residue_number = int(summary_row["peak_residue_number"])
        peak_variant_value_series = variant_stats_df.loc[
            variant_stats_df["residue_number"] == peak_residue_number,
            "mean_rmsf_angstrom",
        ]
        if not peak_variant_value_series.empty:
            axis.scatter(
                [peak_residue_number],
                [float(peak_variant_value_series.iloc[0])],
                s=14,
                color=variant_color,
                edgecolors="white",
                linewidths=0.35,
                zorder=5,
            )

        title_lines = [short_variant_label(sample_name, family_id)]
        subtitle_parts = []
        if "delta_tm_to_wildtype_celsius" in summary_row.index and pd.notna(
            summary_row["delta_tm_to_wildtype_celsius"]
        ):
            subtitle_parts.append(
                f"dTm {float(summary_row['delta_tm_to_wildtype_celsius']):+0.1f} C"
            )
        subtitle_parts.append(
            f"mean |dRMSF| {float(summary_row['mean_abs_delta_rmsf_angstrom']):0.2f} A"
        )
        title_lines.append(" | ".join(subtitle_parts))
        axis.set_title("\n".join(title_lines), fontsize=title_size_pt)
        axis.set_xlim(int(residue_numbers.min()), int(residue_numbers.max()))
        axis.set_ylim(0.0, rmsf_max * 1.08)
        axis.set_xticks(x_ticks)
        axis.grid(axis="y", alpha=0.18, linewidth=0.45)
        axis.tick_params(axis="both", labelsize=tick_size_pt)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)

    for axis in flat_axes[n_variants:]:
        axis.remove()

    for row_index in range(n_rows):
        axes[row_index, 0].set_ylabel("RMSF (A)", fontsize=label_size_pt)
    for column_index in range(n_columns):
        axes[n_rows - 1, column_index].set_xlabel("Residue number", fontsize=label_size_pt)

    flat_axes[0].legend(loc="upper right", fontsize=tick_size_pt, frameon=False, ncol=2)
    figure.suptitle(
        f"{family_id}: RMSF comparison vs wildtype across replicas",
        fontsize=title_size_pt + 0.6,
    )

    return figure, axes


def plot_variant_pair_rmsf(
    family_id: str,
    sample_name: str,
    md_rmsf_df: pd.DataFrame,
    variant_delta_rmsf_df: pd.DataFrame,
    figure_width_mm: float = 145,
    figure_height_mm: float = 90,
    title_size_pt: float = 7.0,
    label_size_pt: float = 6.2,
    tick_size_pt: float = 5.6,
):
    wildtype_df = md_rmsf_df.loc[
        (md_rmsf_df["family_id"] == family_id) & (md_rmsf_df["is_wildtype"])
    ].sort_values("residue_number")
    variant_df = md_rmsf_df.loc[
        (md_rmsf_df["family_id"] == family_id) & (md_rmsf_df["sample_name"] == sample_name)
    ].sort_values("residue_number")
    delta_df = variant_delta_rmsf_df.loc[
        (variant_delta_rmsf_df["family_id"] == family_id)
        & (variant_delta_rmsf_df["sample_name"] == sample_name)
    ].sort_values("residue_number")

    if wildtype_df.empty:
        raise ValueError(f"No wildtype RMSF profile found for family '{family_id}'")
    if variant_df.empty:
        raise ValueError(f"No variant RMSF profile found for sample '{sample_name}'")
    if delta_df.empty:
        raise ValueError(f"No delta RMSF profile found for sample '{sample_name}'")

    plot_df = (
        wildtype_df[["residue_number", "rmsf_angstrom"]]
        .rename(columns={"rmsf_angstrom": "wildtype_rmsf_angstrom"})
        .merge(
            variant_df[["residue_number", "rmsf_angstrom"]].rename(
                columns={"rmsf_angstrom": "variant_rmsf_angstrom"}
            ),
            on="residue_number",
            how="inner",
        )
        .merge(
            delta_df[["residue_number", "delta_rmsf_angstrom", "abs_delta_rmsf_angstrom"]],
            on="residue_number",
            how="inner",
        )
        .sort_values("residue_number")
        .reset_index(drop=True)
    )

    peak_row = plot_df.loc[plot_df["abs_delta_rmsf_angstrom"].idxmax()]
    residue_numbers = plot_df["residue_number"].to_numpy(dtype=int)
    xticks = choose_residue_ticks(residue_numbers)

    figure_width_inch, figure_height_inch = mm_to_inches(figure_width_mm, figure_height_mm)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width_inch, figure_height_inch),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.55, 1.0]},
    )
    profile_axis, delta_axis = axes

    profile_axis.plot(
        plot_df["residue_number"],
        plot_df["wildtype_rmsf_angstrom"],
        color="black",
        linewidth=1.15,
        label="Wildtype",
    )
    profile_axis.plot(
        plot_df["residue_number"],
        plot_df["variant_rmsf_angstrom"],
        color="#c2185b",
        linewidth=1.0,
        label=short_variant_label(sample_name, family_id),
    )
    profile_axis.scatter(
        [peak_row["residue_number"]],
        [peak_row["variant_rmsf_angstrom"]],
        s=18,
        color="#c2185b",
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
    )

    summary_text = "\n".join(
        [
            f"mean dRMSF = {plot_df['delta_rmsf_angstrom'].mean():+.3f} A",
            f"mean |dRMSF| = {plot_df['abs_delta_rmsf_angstrom'].mean():.3f} A",
            f"max |dRMSF| = {peak_row['abs_delta_rmsf_angstrom']:.3f} A at {int(peak_row['residue_number'])}",
        ]
    )
    profile_axis.text(
        0.012,
        0.98,
        summary_text,
        transform=profile_axis.transAxes,
        va="top",
        ha="left",
        fontsize=tick_size_pt,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.25"},
    )
    profile_axis.set_ylabel("RMSF (A)", fontsize=label_size_pt)
    profile_axis.set_title(
        f"{family_id}: {short_variant_label(sample_name, family_id)} vs wildtype",
        fontsize=title_size_pt,
    )
    profile_axis.legend(loc="upper right", fontsize=tick_size_pt, frameon=False)

    delta_values = plot_df["delta_rmsf_angstrom"].to_numpy(dtype=float)
    delta_axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    delta_axis.plot(
        plot_df["residue_number"],
        delta_values,
        color="#444444",
        linewidth=0.9,
    )
    delta_axis.fill_between(
        plot_df["residue_number"],
        0.0,
        delta_values,
        where=delta_values >= 0.0,
        color="#d6604d",
        alpha=0.4,
        interpolate=True,
    )
    delta_axis.fill_between(
        plot_df["residue_number"],
        0.0,
        delta_values,
        where=delta_values < 0.0,
        color="#4393c3",
        alpha=0.4,
        interpolate=True,
    )
    delta_axis.scatter(
        [peak_row["residue_number"]],
        [peak_row["delta_rmsf_angstrom"]],
        s=18,
        color="#444444",
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        axis.tick_params(axis="both", labelsize=tick_size_pt)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.6)
        axis.spines["bottom"].set_linewidth(0.6)
        axis.set_xlim(int(residue_numbers.min()), int(residue_numbers.max()))

    delta_axis.set_xticks(xticks)
    delta_axis.set_xlabel("Residue number", fontsize=label_size_pt)
    delta_axis.set_ylabel("Variant - WT\nRMSF (A)", fontsize=label_size_pt)

    return figure, axes
