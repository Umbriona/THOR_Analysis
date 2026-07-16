from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_hex
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


DEFAULT_RMSF_DELTA_CMAP = LinearSegmentedColormap.from_list(
    "rmsf_delta",
    ["#2166ac", "#f7f7f7", "#b2182b"],
)


def mm_to_inches(*mm_values: float) -> tuple[float, ...]:
    return tuple(float(value) / 25.4 for value in mm_values)


def format_fixed_width_monospace_labels(
    labels: Iterable[object],
    extra_columns: int = 0,
) -> tuple[list[str], int]:
    label_strings = [str(label) for label in labels]
    if not label_strings:
        return [], 0

    target_width = max(len(label) for label in label_strings) + max(int(extra_columns), 0)
    return [label.rjust(target_width) for label in label_strings], target_width


def estimate_monospace_label_width_inches(
    num_columns: int,
    font_size_pt: float,
) -> float:
    if num_columns <= 0:
        return 0.0
    return float(num_columns) * float(font_size_pt) * 0.62 / 72.0


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
    rotation_matrix, mobile_center, reference_center = compute_kabsch_transform(
        reference_xyz=reference_xyz,
        mobile_xyz=mobile_xyz,
    )
    return apply_kabsch_transform(
        xyz=mobile_xyz,
        rotation_matrix=rotation_matrix,
        mobile_center=mobile_center,
        reference_center=reference_center,
    )


def compute_kabsch_transform(
    reference_xyz: np.ndarray,
    mobile_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_xyz = np.asarray(reference_xyz, dtype=float)
    mobile_xyz = np.asarray(mobile_xyz, dtype=float)
    reference_center = reference_xyz.mean(axis=0)
    mobile_center = mobile_xyz.mean(axis=0)

    reference_centered = reference_xyz - reference_center
    mobile_centered = mobile_xyz - mobile_center

    covariance_matrix = mobile_centered.T @ reference_centered
    left_singular_vectors, _, right_singular_vectors_t = np.linalg.svd(covariance_matrix)
    rotation_matrix = left_singular_vectors @ right_singular_vectors_t

    if np.linalg.det(rotation_matrix) < 0:
        left_singular_vectors[:, -1] *= -1
        rotation_matrix = left_singular_vectors @ right_singular_vectors_t

    return rotation_matrix, mobile_center, reference_center


def apply_kabsch_transform(
    xyz: np.ndarray,
    rotation_matrix: np.ndarray,
    mobile_center: np.ndarray,
    reference_center: np.ndarray,
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    return (xyz - mobile_center) @ rotation_matrix + reference_center


def compute_ca_superposition_alignment(
    structure_a_trace: pd.DataFrame,
    structure_b_trace: pd.DataFrame,
) -> dict[str, object]:
    paired_trace_df = structure_a_trace.merge(
        structure_b_trace,
        on=["chain_id", "residue_number", "insertion_code"],
        suffixes=("_a", "_b"),
    )
    if len(paired_trace_df) < 3:
        raise ValueError("At least three matched CA residues are required for structural superposition")

    structure_a_xyz = paired_trace_df[["x_a", "y_a", "z_a"]].to_numpy(dtype=float)
    structure_b_xyz = paired_trace_df[["x_b", "y_b", "z_b"]].to_numpy(dtype=float)
    rotation_matrix, mobile_center, reference_center = compute_kabsch_transform(
        reference_xyz=structure_a_xyz,
        mobile_xyz=structure_b_xyz,
    )
    aligned_structure_b_xyz = apply_kabsch_transform(
        xyz=structure_b_xyz,
        rotation_matrix=rotation_matrix,
        mobile_center=mobile_center,
        reference_center=reference_center,
    )

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
        "paired_trace_df": paired_trace_df,
        "rotation_matrix": rotation_matrix,
        "mobile_center": mobile_center,
        "reference_center": reference_center,
        "aligned_mobile_xyz": aligned_structure_b_xyz,
        "aligned_length": int(len(paired_trace_df)),
        "rmsd_angstrom": rmsd_angstrom,
        "seq_identity_aligned": float(residue_identity),
        "tm_score_to_structure_a": tm_score_to_structure_a,
        "tm_score_to_structure_b": tm_score_to_structure_b,
        "aligned_fraction_to_structure_a": float(len(paired_trace_df) / max(1, structure_a_length)),
        "aligned_fraction_to_structure_b": float(len(paired_trace_df) / max(1, structure_b_length)),
    }


def compute_ca_superposition_metrics(
    structure_a_trace: pd.DataFrame,
    structure_b_trace: pd.DataFrame,
) -> dict[str, float | int | None]:
    alignment_result = compute_ca_superposition_alignment(
        structure_a_trace=structure_a_trace,
        structure_b_trace=structure_b_trace,
    )
    return {
        "aligned_length": alignment_result["aligned_length"],
        "rmsd_angstrom": alignment_result["rmsd_angstrom"],
        "seq_identity_aligned": alignment_result["seq_identity_aligned"],
        "tm_score_to_structure_a": alignment_result["tm_score_to_structure_a"],
        "tm_score_to_structure_b": alignment_result["tm_score_to_structure_b"],
        "aligned_fraction_to_structure_a": alignment_result["aligned_fraction_to_structure_a"],
        "aligned_fraction_to_structure_b": alignment_result["aligned_fraction_to_structure_b"],
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


def format_tm_delta_label(base_label: str, delta_tm_celsius: object) -> str:
    if pd.isna(delta_tm_celsius):
        return str(base_label)
    return f"{base_label}  (dTm {float(delta_tm_celsius):+0.1f} C)"


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
        row_label_map[row.sample_name] = format_tm_delta_label(short_label, delta_tm)

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


##########################################################################################
# MD Variant Family Comparison Helpers
##########################################################################################

MD_VARIANT_DEFAULT_FAMILY_ORDER = [
    "A0A2S1LEZ1",
    "A0A372IUB3",
    "MGYP001421927114",
]
MD_VARIANT_PAIR_METRICS = [
    "residue_contact_pairs",
    "saltbridge_pairs",
    "hbond_pairs",
]
MD_VARIANT_INTERACTION_COUNT_METRICS = [
    "residue_contacts",
    "hbond_counts",
    "saltbridge",
]
MD_VARIANT_INTERACTION_COUNT_COLUMN_MAP = {
    "residue_contacts": "residue_contact_count",
    "hbond_counts": "hbond_count",
    "saltbridge": "saltbridge_count",
}
MD_VARIANT_INTERACTION_COUNT_LABEL_MAP = {
    "residue_contacts": "Residue contacts",
    "hbond_counts": "Hydrogen bonds",
    "saltbridge": "Salt bridges",
}
MD_VARIANT_SIGNAL_COLUMNS = [
    "rmsf_neighborhood_gain",
    "contact_gain_total",
    "saltbridge_gain_total",
    "hbond_gain_total",
    "sasa_neighborhood_gain",
    "ordered_ss_neighborhood_gain",
]
MD_VARIANT_DELTA_CMAP = LinearSegmentedColormap.from_list(
    "md_variant_delta",
    ["#2166ac", "#f7f7f7", "#b2182b"],
)
MD_VARIANT_OCCUPANCY_CMAP = LinearSegmentedColormap.from_list(
    "md_variant_occupancy",
    ["#f7fbff", "#6baed6", "#08306b"],
)
MD_VARIANT_SIGNIFICANCE_THRESHOLDS = (
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
)
MD_VARIANT_BLOCK_SIZE_FRAMES = 100
MD_VARIANT_BLOCK_PERMUTATIONS = 2000
MD_VARIANT_BLOCK_TEST_SEED = 20260617


@dataclass
class MDVariantAnalysisBundle:
    metric_window_suffix: str
    metric_window_start: int
    metric_window_stop: int
    metadata_df: pd.DataFrame
    replicate_count_by_system: dict[str, int]
    local_residue_sets: dict[str, set[int]]
    family_ids: list[str]
    rmsf_summary_df: pd.DataFrame
    sasa_summary_df: pd.DataFrame
    ss_counts_summary_df: pd.DataFrame
    ss_residue_summary_df: pd.DataFrame
    interaction_count_summary_by_metric: dict[str, pd.DataFrame]
    interaction_count_replicate_by_metric: dict[str, pd.DataFrame]
    interaction_count_block_by_metric: dict[str, pd.DataFrame]
    global_sasa_replicate_df: pd.DataFrame
    pair_replicate_by_metric: dict[str, pd.DataFrame]
    pair_residue_replicate_by_metric: dict[str, pd.DataFrame]
    pair_summary_by_metric: dict[str, pd.DataFrame]
    rmsf_delta_df: pd.DataFrame
    sasa_delta_df: pd.DataFrame
    ss_residue_delta_df: pd.DataFrame
    interaction_count_delta_by_metric: dict[str, pd.DataFrame]
    pair_delta_by_metric: dict[str, pd.DataFrame]
    variant_signal_df: pd.DataFrame
    family_pattern_summary_df: pd.DataFrame


@dataclass
class MDVariantMolstarExportBundle:
    viewer_index_df: pd.DataFrame
    score_table_df: pd.DataFrame
    global_rmsd_summary_df: pd.DataFrame


@dataclass
class WildtypeMolstarOverlayExportBundle:
    viewer_index_df: pd.DataFrame
    alignment_summary_df: pd.DataFrame


@dataclass
class WildtypeMolstarSuperpositionExport:
    html_path: Path
    pdb_path: Path
    alignment_summary_df: pd.DataFrame


@dataclass
class WildtypeChimeraXCommandBundle:
    residue_color_df: pd.DataFrame
    apply_command: str
    open_command: str


def normalize_md_variant_metric_column_name(column_name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", column_name.strip()).strip("_").lower()


def parse_md_variant_residue_list(value: object) -> list[int]:
    if pd.isna(value):
        return []
    value_text = str(value).strip()
    if not value_text:
        return []
    return [int(token) for token in re.split(r"\s+", value_text) if token]


def parse_md_variant_mutation_label_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    value_text = str(value).strip()
    if not value_text:
        return []
    return [token for token in value_text.split(";") if token]


def wrap_md_variant_label(text: str, width: int = 28) -> str:
    words = str(text).split()
    if not words:
        return str(text)

    lines = []
    current_line = []
    current_length = 0
    for word in words:
        next_length = current_length + len(word) + (1 if current_line else 0)
        if next_length > width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length = next_length

    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)


def save_md_variant_figure(figure, output_path: Path, dpi: int = 300) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")


def order_md_variant_families(metadata_df: pd.DataFrame) -> list[str]:
    family_set = set(metadata_df["family"].dropna().astype(str))
    ordered_families = [
        family_id for family_id in MD_VARIANT_DEFAULT_FAMILY_ORDER if family_id in family_set
    ]
    ordered_families.extend(sorted(family_set.difference(ordered_families)))
    return ordered_families


def load_md_variant_metadata(
    mutation_metadata_path: Path,
) -> pd.DataFrame:
    metadata_df = pd.read_csv(mutation_metadata_path)
    metadata_df = metadata_df.loc[
        metadata_df["comparison_status"].isin(["wildtype", "ok"])
    ].copy()
    metadata_df["family"] = metadata_df["wt_system"]
    metadata_df["is_wildtype"] = metadata_df["comparison_status"].eq("wildtype")
    metadata_df["mutation_resids_list"] = metadata_df["mutation_resids"].apply(
        parse_md_variant_residue_list
    )
    metadata_df["mutation_labels_list"] = metadata_df["mutation_labels"].apply(
        parse_md_variant_mutation_label_list
    )
    metadata_df["short_label"] = metadata_df.apply(
        lambda row: "WT"
        if row["is_wildtype"]
        else short_variant_label(str(row["system"]), str(row["family"])),
        axis=1,
    )
    metadata_df = metadata_df.sort_values(
        ["family", "is_wildtype", "mutation_count", "system"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return metadata_df


def annotate_md_variant_metadata_with_tm(
    metadata_df: pd.DataFrame,
    tm_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    annotated_df = metadata_df.copy()
    tm_value_columns = [
        "median_tm_celsius",
        "wildtype_median_tm_celsius",
        "delta_tm_to_wildtype_celsius",
    ]

    if tm_summary_df is not None and not tm_summary_df.empty:
        tm_lookup_df = tm_summary_df[
            [
                "family_id",
                "sample_name",
                "median_tm_celsius",
                "wildtype_median_tm_celsius",
                "delta_tm_to_wildtype_celsius",
            ]
        ].drop_duplicates(subset=["family_id", "sample_name"]).rename(
            columns={"family_id": "family", "sample_name": "system"}
        )
        annotated_df = annotated_df.drop(
            columns=[column_name for column_name in tm_value_columns if column_name in annotated_df.columns],
            errors="ignore",
        ).merge(
            tm_lookup_df,
            on=["family", "system"],
            how="left",
        )
    else:
        for column_name in tm_value_columns:
            if column_name not in annotated_df.columns:
                annotated_df[column_name] = np.nan

    annotated_df["tm_display_label"] = annotated_df.apply(
        lambda row: row["short_label"]
        if bool(row["is_wildtype"])
        else format_tm_delta_label(row["short_label"], row["delta_tm_to_wildtype_celsius"]),
        axis=1,
    )
    return annotated_df


def build_md_variant_label_lookup(metadata_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "family_order",
        "family_id",
        "wildtype_system",
        "system",
        "is_wildtype",
        "plot_order_within_family",
        "variant_rank_in_family",
        "simple_label",
        "simple_tm_display_label",
        "short_label",
        "tm_display_label",
        "mutation_count",
        "mutation_resids",
        "mutation_labels",
        "median_tm_celsius",
        "wildtype_median_tm_celsius",
        "delta_tm_to_wildtype_celsius",
        "sequence_length",
        "wt_sequence_length",
        "structure_file",
        "wt_structure_file",
        "comparison_status",
    ]
    if metadata_df.empty:
        return pd.DataFrame(columns=output_columns)

    lookup_df = metadata_df.copy()
    if "family" not in lookup_df.columns:
        lookup_df["family"] = lookup_df["wt_system"]
    if "is_wildtype" not in lookup_df.columns:
        lookup_df["is_wildtype"] = lookup_df["comparison_status"].eq("wildtype")

    tm_value_columns = [
        "median_tm_celsius",
        "wildtype_median_tm_celsius",
        "delta_tm_to_wildtype_celsius",
    ]
    for column_name in tm_value_columns:
        if column_name not in lookup_df.columns:
            lookup_df[column_name] = np.nan

    if "tm_display_label" not in lookup_df.columns:
        lookup_df["tm_display_label"] = lookup_df.apply(
            lambda row: row["short_label"]
            if bool(row["is_wildtype"])
            else format_tm_delta_label(row["short_label"], row["delta_tm_to_wildtype_celsius"]),
            axis=1,
        )

    family_order_map = {
        family_id: index + 1
        for index, family_id in enumerate(order_md_variant_families(lookup_df))
    }
    lookup_df["family_order"] = lookup_df["family"].map(family_order_map).astype(int)
    lookup_df["family_id"] = lookup_df["family"]
    lookup_df["wildtype_system"] = lookup_df["wt_system"]
    lookup_df = lookup_df.sort_values(
        ["family_order", "is_wildtype", "mutation_count", "system"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    lookup_df["plot_order_within_family"] = lookup_df.groupby("family_id").cumcount() + 1
    lookup_df["variant_rank_in_family"] = 0
    variant_mask = ~lookup_df["is_wildtype"].astype(bool)
    lookup_df.loc[variant_mask, "variant_rank_in_family"] = (
        lookup_df.loc[variant_mask]
        .groupby("family_id")
        .cumcount()
        .add(1)
    )
    lookup_df["simple_label"] = lookup_df["variant_rank_in_family"].map(
        lambda rank: "WT" if int(rank) == 0 else f"Variant{int(rank)}"
    )
    lookup_df["simple_tm_display_label"] = lookup_df.apply(
        lambda row: row["simple_label"]
        if bool(row["is_wildtype"])
        else format_tm_delta_label(row["simple_label"], row["delta_tm_to_wildtype_celsius"]),
        axis=1,
    )
    return lookup_df[output_columns].copy()


def export_md_variant_label_lookup(
    metadata_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    lookup_df = build_md_variant_label_lookup(metadata_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_df.to_csv(output_path, index=False, float_format="%.4f")
    return lookup_df


def load_md_variant_label_lookup(
    label_lookup_path: Path,
) -> pd.DataFrame:
    lookup_df = pd.read_csv(label_lookup_path)
    required_columns = {"family_id", "system", "simple_label", "simple_tm_display_label"}
    missing_columns = required_columns.difference(lookup_df.columns)
    if missing_columns:
        missing_column_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"MD variant label lookup is missing required columns: {missing_column_text}"
        )
    return lookup_df


def annotate_md_variant_metadata_with_label_lookup(
    metadata_df: pd.DataFrame,
    label_lookup_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if label_lookup_df is None or label_lookup_df.empty:
        return metadata_df

    merge_columns = [
        column_name
        for column_name in [
            "family_id",
            "system",
            "variant_rank_in_family",
            "simple_label",
            "simple_tm_display_label",
        ]
        if column_name in label_lookup_df.columns
    ]
    if not {"family_id", "system"}.issubset(merge_columns):
        return metadata_df

    annotated_df = metadata_df.drop(
        columns=[
            column_name
            for column_name in ["variant_rank_in_family", "simple_label", "simple_tm_display_label"]
            if column_name in metadata_df.columns
        ],
        errors="ignore",
    ).merge(
        label_lookup_df[merge_columns].drop_duplicates(subset=["family_id", "system"]).rename(
            columns={"family_id": "family"}
        ),
        on=["family", "system"],
        how="left",
        validate="one_to_one",
    )
    return annotated_df


def _get_md_variant_row_value(row: object, column_name: str) -> object:
    if isinstance(row, pd.Series):
        return row.get(column_name, np.nan)
    return getattr(row, column_name, np.nan)


def get_md_variant_base_label_from_row(row: object) -> str:
    simple_label = _get_md_variant_row_value(row, "simple_label")
    if pd.notna(simple_label):
        return str(simple_label)
    return str(_get_md_variant_row_value(row, "short_label"))


def get_md_variant_base_label(metadata_df: pd.DataFrame, system_name: str) -> str:
    system_row_df = metadata_df.loc[metadata_df["system"].eq(system_name)]
    if system_row_df.empty:
        raise KeyError(f"Unknown MD variant system label: {system_name}")
    return get_md_variant_base_label_from_row(system_row_df.iloc[0])


def get_md_variant_display_label(metadata_df: pd.DataFrame, system_name: str) -> str:
    system_row_df = metadata_df.loc[metadata_df["system"].eq(system_name)]
    if system_row_df.empty:
        raise KeyError(f"Unknown MD variant system label: {system_name}")

    if (
        "simple_tm_display_label" in system_row_df.columns
        and pd.notna(system_row_df["simple_tm_display_label"].iat[0])
    ):
        return str(system_row_df["simple_tm_display_label"].iat[0])
    if "tm_display_label" in system_row_df.columns and pd.notna(system_row_df["tm_display_label"].iat[0]):
        return str(system_row_df["tm_display_label"].iat[0])
    if "simple_label" in system_row_df.columns and pd.notna(system_row_df["simple_label"].iat[0]):
        return str(system_row_df["simple_label"].iat[0])
    return str(system_row_df["short_label"].iat[0])


def get_md_variant_display_label_from_row(row: object) -> str:
    display_label = _get_md_variant_row_value(row, "simple_tm_display_label")
    if pd.notna(display_label):
        return str(display_label)
    display_label = _get_md_variant_row_value(row, "tm_display_label")
    if pd.notna(display_label):
        return str(display_label)
    base_label = _get_md_variant_row_value(row, "simple_label")
    if pd.notna(base_label):
        return str(base_label)
    return str(_get_md_variant_row_value(row, "short_label"))


def build_md_family_rmsf_panel_title(
    variant_metadata: pd.Series,
    label_wrap_width: int = 18,
) -> str:
    title_line = wrap_md_variant_label(
        get_md_variant_base_label_from_row(variant_metadata),
        width=label_wrap_width,
    )
    subtitle_parts = []
    if "delta_tm_to_wildtype_celsius" in variant_metadata.index and pd.notna(
        variant_metadata["delta_tm_to_wildtype_celsius"]
    ):
        subtitle_parts.append(
            f"dTm {float(variant_metadata['delta_tm_to_wildtype_celsius']):+0.1f} C"
        )
    subtitle_parts.append(f"{int(variant_metadata['mutation_count'])} mutations")
    return "\n".join([title_line, " | ".join(subtitle_parts)])


def parse_md_variant_metric_window_suffix(metric_window_suffix: str) -> tuple[int, int]:
    suffix_match = re.fullmatch(r"(\d+)_(\d+)", str(metric_window_suffix))
    if not suffix_match:
        raise ValueError(
            "Metric window suffix must look like '0_100000' or '0_300000', "
            f"got '{metric_window_suffix}'"
        )
    return int(suffix_match.group(1)), int(suffix_match.group(2))


def parse_md_variant_metric_window_suffix_if_numeric(
    metric_window_suffix: str,
) -> tuple[int, int] | None:
    suffix_match = re.fullmatch(r"(\d+)_(\d+)", str(metric_window_suffix))
    if not suffix_match:
        return None
    return int(suffix_match.group(1)), int(suffix_match.group(2))


def discover_md_variant_metric_window_suffixes(
    metric_dir: Path,
    metric_name: str,
) -> list[str]:
    pattern = re.compile(rf"^all_systems_{re.escape(metric_name)}_(.+)\.csv$")
    suffixes = []
    for metric_path in sorted(metric_dir.glob(f"all_systems_{metric_name}_*.csv")):
        match = pattern.match(metric_path.name)
        if match:
            suffixes.append(match.group(1))
    return suffixes


def resolve_md_variant_metric_window_bounds(
    metric_dir: Path,
    metric_names: Iterable[str],
    metric_window_suffix: str,
) -> tuple[int, int]:
    metric_names = [str(metric_name) for metric_name in metric_names]
    for metric_name in metric_names:
        metric_path = (
            metric_dir / f"all_systems_{metric_name}_{metric_window_suffix}.csv"
        )
        if not metric_path.exists():
            continue
        try:
            metric_bounds_df = pd.read_csv(
                metric_path,
                usecols=["start", "stop"],
                low_memory=False,
            )
        except ValueError:
            continue
        start_values = pd.to_numeric(metric_bounds_df["start"], errors="coerce").dropna()
        stop_values = pd.to_numeric(metric_bounds_df["stop"], errors="coerce").dropna()
        if not start_values.empty and not stop_values.empty:
            return int(start_values.min()), int(stop_values.max())

    numeric_bounds = parse_md_variant_metric_window_suffix_if_numeric(
        metric_window_suffix
    )
    if numeric_bounds is not None:
        return numeric_bounds
    raise ValueError(
        "Unable to determine MD metric window bounds for suffix "
        f"'{metric_window_suffix}'"
    )


def resolve_md_variant_metric_window_suffix(
    metric_dir: Path,
    required_metrics: Iterable[str],
    metric_window: str | int | None = "latest",
) -> str:
    required_metrics = [str(metric_name) for metric_name in required_metrics]
    available_suffixes_by_metric = {
        metric_name: set(
            discover_md_variant_metric_window_suffixes(metric_dir, metric_name)
        )
        for metric_name in required_metrics
    }
    missing_metrics = [
        metric_name
        for metric_name, suffixes in available_suffixes_by_metric.items()
        if not suffixes
    ]
    if missing_metrics:
        raise FileNotFoundError(
            "Missing combined metric CSVs for: "
            + ", ".join(sorted(missing_metrics))
        )

    common_suffixes = sorted(
        set.intersection(*available_suffixes_by_metric.values()),
        key=lambda suffix: resolve_md_variant_metric_window_bounds(
            metric_dir=metric_dir,
            metric_names=required_metrics,
            metric_window_suffix=suffix,
        ),
    )
    if not common_suffixes:
        raise FileNotFoundError(
            "No shared metric window was found across the required metrics. "
            "Check that all combined CSVs were generated for the same simulation window."
        )

    if metric_window in (None, "latest"):
        return common_suffixes[-1]

    if isinstance(metric_window, int):
        requested_suffix = f"0_{metric_window}"
    else:
        requested_text = str(metric_window).strip()
        if re.fullmatch(r"\d+", requested_text):
            requested_suffix = f"0_{requested_text}"
        elif f"0_{requested_text}" in common_suffixes:
            requested_suffix = f"0_{requested_text}"
        else:
            requested_suffix = requested_text

    if requested_suffix not in common_suffixes:
        raise FileNotFoundError(
            f"Requested metric window '{requested_suffix}' is not available for all required metrics. "
            f"Shared available windows: {', '.join(common_suffixes)}"
        )
    return requested_suffix


def read_md_variant_metric_table(
    metric_dir: Path,
    metric_name: str,
    metric_window_suffix: str,
) -> pd.DataFrame:
    metric_path = metric_dir / f"all_systems_{metric_name}_{metric_window_suffix}.csv"
    metric_df = pd.read_csv(metric_path, low_memory=False)
    metric_df.columns = [
        normalize_md_variant_metric_column_name(column_name)
        for column_name in metric_df.columns
    ]
    if metric_name == "rmsf":
        if "resid" not in metric_df.columns and "frame" in metric_df.columns:
            metric_df["resid"] = (
                pd.to_numeric(metric_df["frame"], errors="coerce")
                .round()
                .astype("Int64")
            )
        if "resname" not in metric_df.columns:
            metric_df["resname"] = pd.NA
    return metric_df


def compute_md_variant_replicate_counts(metric_df: pd.DataFrame) -> dict[str, int]:
    return (
        metric_df.groupby("system")["replicate"].nunique().astype(int).to_dict()
    )


def backfill_md_variant_rmsf_resnames(
    rmsf_df: pd.DataFrame,
    residue_reference_df: pd.DataFrame,
) -> pd.DataFrame:
    if "resname" in rmsf_df.columns and rmsf_df["resname"].notna().all():
        return rmsf_df

    residue_lookup_df = (
        residue_reference_df[
            ["system", "replicate", "replicate_index", "resid", "resname"]
        ]
        .dropna(subset=["resid"])
        .drop_duplicates()
    )
    enriched_df = rmsf_df.merge(
        residue_lookup_df,
        on=["system", "replicate", "replicate_index", "resid"],
        how="left",
        suffixes=("", "_reference"),
    )
    if "resname_reference" in enriched_df.columns:
        enriched_df["resname"] = enriched_df["resname"].fillna(
            enriched_df["resname_reference"]
        )
        enriched_df = enriched_df.drop(columns=["resname_reference"])
    enriched_df["resname"] = enriched_df["resname"].fillna("UNK")
    return enriched_df


def load_md_variant_local_residue_sets(
    metadata_df: pd.DataFrame,
    rmsf_raw_df: pd.DataFrame,
    mutation_neighborhood_df: pd.DataFrame,
) -> dict[str, set[int]]:
    neighborhood_residue_sets = {}
    if not mutation_neighborhood_df.empty:
        neighborhood_residue_sets = (
            mutation_neighborhood_df.groupby("system")["neighbor_resid"]
            .apply(lambda values: {int(value) for value in values})
            .to_dict()
        )

    rmsf_near_sets = {}
    if "near_mutation" in rmsf_raw_df.columns:
        rmsf_near_sets = (
            rmsf_raw_df.loc[rmsf_raw_df["near_mutation"].fillna(False)]
            .groupby("system")["resid"]
            .apply(lambda values: {int(value) for value in values})
            .to_dict()
        )

    local_residue_sets = {}
    for row in metadata_df.itertuples(index=False):
        system_name = str(row.system)
        residue_set = set(int(value) for value in row.mutation_resids_list)
        residue_set.update(neighborhood_residue_sets.get(system_name, set()))
        residue_set.update(rmsf_near_sets.get(system_name, set()))
        local_residue_sets[system_name] = residue_set

    return local_residue_sets


def aggregate_md_variant_residue_metric(
    metric_df: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    optional_columns = [
        column_name
        for column_name in ["contains_mutation_residue", "near_mutation"]
        if column_name in metric_df.columns
    ]
    per_replicate_df = (
        metric_df[
            [
                "system",
                "replicate",
                "replicate_index",
                "resid",
                "resname",
                *optional_columns,
                *value_columns,
            ]
        ]
        .groupby(
            ["system", "replicate", "replicate_index", "resid", "resname"],
            as_index=False,
        )
        .agg(
            {
                **{column_name: "mean" for column_name in value_columns},
                **{column_name: "max" for column_name in optional_columns},
            }
        )
    )

    system_level_df = (
        per_replicate_df.groupby(["system", "resid"], as_index=False)
        .agg(
            {
                "resname": "first",
                "replicate": "nunique",
                **{column_name: "max" for column_name in optional_columns},
                **{column_name: ["mean", "std"] for column_name in value_columns},
            }
        )
    )
    system_level_df.columns = [
        "system",
        "resid",
        "resname",
        "n_replicates",
        *optional_columns,
        *[
            f"{column_name}_{statistic}"
            for column_name in value_columns
            for statistic in ["mean", "std"]
        ],
    ]
    for column_name in value_columns:
        std_column = f"{column_name}_std"
        system_level_df[std_column] = system_level_df[std_column].fillna(0.0)
    return system_level_df


def aggregate_md_variant_secondary_structure_counts(metric_df: pd.DataFrame) -> pd.DataFrame:
    per_replicate_df = (
        metric_df.groupby(["system", "replicate", "replicate_index"], as_index=False)
        .agg(
            helix_fraction=("helix_fraction", "mean"),
            strand_fraction=("strand_fraction", "mean"),
            loop_fraction=("loop_fraction", "mean"),
        )
    )
    per_replicate_df["ordered_fraction"] = (
        per_replicate_df["helix_fraction"] + per_replicate_df["strand_fraction"]
    )

    system_level_df = (
        per_replicate_df.groupby("system", as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            helix_fraction_mean=("helix_fraction", "mean"),
            helix_fraction_std=("helix_fraction", "std"),
            strand_fraction_mean=("strand_fraction", "mean"),
            strand_fraction_std=("strand_fraction", "std"),
            loop_fraction_mean=("loop_fraction", "mean"),
            loop_fraction_std=("loop_fraction", "std"),
            ordered_fraction_mean=("ordered_fraction", "mean"),
            ordered_fraction_std=("ordered_fraction", "std"),
        )
    )
    std_columns = [
        column_name
        for column_name in system_level_df.columns
        if column_name.endswith("_std")
    ]
    system_level_df[std_columns] = system_level_df[std_columns].fillna(0.0)
    return system_level_df


def aggregate_md_variant_interaction_count_metric(
    metric_df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    prepared_df = metric_df.copy()
    prepared_df["frame"] = pd.to_numeric(prepared_df["frame"], errors="coerce").round().astype("Int64")
    prepared_df[value_column] = pd.to_numeric(prepared_df[value_column], errors="coerce")
    prepared_df = prepared_df.dropna(subset=["frame", value_column]).copy()
    prepared_df["frame"] = prepared_df["frame"].astype(int)

    per_replicate_df = (
        prepared_df[["system", "replicate", "replicate_index", "frame", value_column]]
        .groupby(["system", "replicate", "replicate_index", "frame"], as_index=False)
        .agg({value_column: "mean"})
    )

    system_level_df = (
        per_replicate_df.groupby(["system", "frame"], as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            count_mean=(value_column, "mean"),
            count_std=(value_column, "std"),
        )
    )
    system_level_df["count_std"] = system_level_df["count_std"].fillna(0.0)
    system_level_df["time_ps"] = system_level_df["frame"].astype(float)
    system_level_df["time_ns"] = system_level_df["time_ps"] / 1000.0
    return system_level_df.sort_values(["system", "frame"]).reset_index(drop=True)


def compute_md_variant_interaction_count_replicate_table(
    metric_df: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    prepared_df = metric_df.copy()
    prepared_df["frame"] = pd.to_numeric(prepared_df["frame"], errors="coerce").round().astype("Int64")
    prepared_df[value_column] = pd.to_numeric(prepared_df[value_column], errors="coerce")
    prepared_df = prepared_df.dropna(subset=["frame", value_column]).copy()
    prepared_df["frame"] = prepared_df["frame"].astype(int)

    per_replicate_df = (
        prepared_df.groupby(["system", "replicate", "replicate_index"], as_index=False)
        .agg(
            n_frames=("frame", "nunique"),
            replicate_mean=(value_column, "mean"),
            replicate_std=(value_column, "std"),
        )
    )
    per_replicate_df["replicate_std"] = per_replicate_df["replicate_std"].fillna(0.0)
    return per_replicate_df.sort_values(
        ["system", "replicate_index", "replicate"]
    ).reset_index(drop=True)


def compute_md_variant_interaction_count_block_table(
    metric_df: pd.DataFrame,
    value_column: str,
    block_size_frames: int = MD_VARIANT_BLOCK_SIZE_FRAMES,
) -> pd.DataFrame:
    if block_size_frames <= 0:
        raise ValueError("block_size_frames must be positive")

    prepared_df = metric_df.copy()
    prepared_df["frame"] = pd.to_numeric(prepared_df["frame"], errors="coerce").round().astype("Int64")
    prepared_df[value_column] = pd.to_numeric(prepared_df[value_column], errors="coerce")
    prepared_df = prepared_df.dropna(subset=["frame", value_column]).copy()
    prepared_df["frame"] = prepared_df["frame"].astype(int)
    prepared_df = prepared_df.sort_values(["system", "replicate_index", "replicate", "frame"]).reset_index(drop=True)
    prepared_df["frame_rank"] = prepared_df.groupby(
        ["system", "replicate", "replicate_index"]
    ).cumcount()
    prepared_df["block_index"] = prepared_df["frame_rank"] // int(block_size_frames)

    block_df = (
        prepared_df.groupby(["system", "replicate", "replicate_index", "block_index"], as_index=False)
        .agg(
            block_start_frame=("frame", "min"),
            block_stop_frame=("frame", "max"),
            n_frames=("frame", "size"),
            block_mean=(value_column, "mean"),
            block_std=(value_column, "std"),
        )
    )
    block_df["block_std"] = block_df["block_std"].fillna(0.0)
    block_df["block_size_frames"] = int(block_size_frames)
    block_df["block_mid_frame"] = (
        block_df["block_start_frame"] + block_df["block_stop_frame"]
    ) / 2.0
    block_df["block_mid_ns"] = block_df["block_mid_frame"] / 1000.0
    return block_df.sort_values(
        ["system", "replicate_index", "replicate", "block_index"]
    ).reset_index(drop=True)


def compute_md_variant_global_sasa_replicate_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"system", "replicate", "replicate_index", "resid", "mean_sasa"}
    missing_columns = required_columns.difference(metric_df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise KeyError(
            "Cannot compute global SASA replicate table because required columns are "
            f"missing: {missing_str}"
        )

    per_replicate_df = (
        metric_df.groupby(["system", "replicate", "replicate_index"], as_index=False)
        .agg(
            residue_count=("resid", "nunique"),
            global_mean_sasa=("mean_sasa", "sum"),
        )
    )
    per_replicate_df["mean_sasa_per_residue"] = (
        per_replicate_df["global_mean_sasa"] / per_replicate_df["residue_count"]
    )
    return per_replicate_df.sort_values(
        ["system", "replicate_index", "replicate"]
    ).reset_index(drop=True)


def prepare_md_variant_pair_metric(metric_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    prepared_df = metric_df.copy()

    if metric_name == "hbond_pairs":
        prepared_df["resid_a"] = prepared_df["donor_resid"].astype(int)
        prepared_df["resname_a"] = prepared_df["donor_resname"].astype(str)
        prepared_df["resid_b"] = prepared_df["acceptor_resid"].astype(int)
        prepared_df["resname_b"] = prepared_df["acceptor_resname"].astype(str)
        prepared_df["pair_key"] = (
            prepared_df["resid_a"].astype(str)
            + "|"
            + prepared_df["resname_a"]
            + "->"
            + prepared_df["resid_b"].astype(str)
            + "|"
            + prepared_df["resname_b"]
        )
        prepared_df["pair_label"] = (
            prepared_df["resname_a"]
            + prepared_df["resid_a"].astype(str)
            + " -> "
            + prepared_df["resname_b"]
            + prepared_df["resid_b"].astype(str)
        )
    else:
        first_is_lower = (
            prepared_df["resid_i"].astype(int) <= prepared_df["resid_j"].astype(int)
        )
        prepared_df["resid_a"] = np.where(
            first_is_lower,
            prepared_df["resid_i"],
            prepared_df["resid_j"],
        ).astype(int)
        prepared_df["resname_a"] = np.where(
            first_is_lower,
            prepared_df["resname_i"],
            prepared_df["resname_j"],
        ).astype(str)
        prepared_df["resid_b"] = np.where(
            first_is_lower,
            prepared_df["resid_j"],
            prepared_df["resid_i"],
        ).astype(int)
        prepared_df["resname_b"] = np.where(
            first_is_lower,
            prepared_df["resname_j"],
            prepared_df["resname_i"],
        ).astype(str)
        prepared_df["pair_key"] = (
            prepared_df["resid_a"].astype(str)
            + "|"
            + prepared_df["resname_a"]
            + "|"
            + prepared_df["resid_b"].astype(str)
            + "|"
            + prepared_df["resname_b"]
        )
        prepared_df["pair_label"] = (
            prepared_df["resname_a"]
            + prepared_df["resid_a"].astype(str)
            + " - "
            + prepared_df["resname_b"]
            + prepared_df["resid_b"].astype(str)
        )

    prepared_df["occupancy"] = prepared_df["occupancy"].astype(float)
    return prepared_df


def significance_stars(p_value: float | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return ""
    for threshold, stars in MD_VARIANT_SIGNIFICANCE_THRESHOLDS:
        if p_value < threshold:
            return stars
    return ""


def _compute_welch_ttest(
    variant_values: np.ndarray,
    wildtype_values: np.ndarray,
) -> tuple[float, float]:
    variant_values = np.asarray(variant_values, dtype=float)
    wildtype_values = np.asarray(wildtype_values, dtype=float)
    variant_values = variant_values[np.isfinite(variant_values)]
    wildtype_values = wildtype_values[np.isfinite(wildtype_values)]

    if variant_values.size < 2 or wildtype_values.size < 2:
        return np.nan, np.nan

    if (
        np.allclose(variant_values, variant_values[0], equal_nan=False)
        and np.allclose(wildtype_values, wildtype_values[0], equal_nan=False)
        and np.isclose(variant_values[0], wildtype_values[0])
    ):
        return 0.0, 1.0

    statistic, p_value = ttest_ind(
        variant_values,
        wildtype_values,
        equal_var=False,
        nan_policy="omit",
    )
    if not np.isfinite(statistic) or not np.isfinite(p_value):
        return np.nan, np.nan
    return float(statistic), float(p_value)


def _compute_two_sample_ks_statistic(
    variant_values: np.ndarray,
    wildtype_values: np.ndarray,
) -> float:
    variant_values = np.sort(np.asarray(variant_values, dtype=float))
    wildtype_values = np.sort(np.asarray(wildtype_values, dtype=float))
    combined_values = np.sort(np.concatenate([variant_values, wildtype_values]))
    if combined_values.size == 0:
        return np.nan

    variant_cdf = np.searchsorted(variant_values, combined_values, side="right") / variant_values.size
    wildtype_cdf = np.searchsorted(wildtype_values, combined_values, side="right") / wildtype_values.size
    return float(np.max(np.abs(variant_cdf - wildtype_cdf)))


def _compute_block_permutation_ks_test(
    variant_values: np.ndarray,
    wildtype_values: np.ndarray,
    n_permutations: int = MD_VARIANT_BLOCK_PERMUTATIONS,
    random_seed: int = MD_VARIANT_BLOCK_TEST_SEED,
) -> tuple[float, float]:
    variant_values = np.asarray(variant_values, dtype=float)
    wildtype_values = np.asarray(wildtype_values, dtype=float)
    variant_values = variant_values[np.isfinite(variant_values)]
    wildtype_values = wildtype_values[np.isfinite(wildtype_values)]

    if variant_values.size < 2 or wildtype_values.size < 2:
        return np.nan, np.nan

    observed_statistic = _compute_two_sample_ks_statistic(variant_values, wildtype_values)
    if not np.isfinite(observed_statistic):
        return np.nan, np.nan
    if np.isclose(observed_statistic, 0.0):
        return 0.0, 1.0

    pooled_values = np.concatenate([variant_values, wildtype_values])
    n_variant = int(variant_values.size)
    rng = np.random.default_rng(random_seed)
    exceed_count = 1

    for _ in range(int(n_permutations)):
        permuted_values = rng.permutation(pooled_values)
        permuted_variant = permuted_values[:n_variant]
        permuted_wildtype = permuted_values[n_variant:]
        permuted_statistic = _compute_two_sample_ks_statistic(
            permuted_variant,
            permuted_wildtype,
        )
        if permuted_statistic >= observed_statistic - 1e-12:
            exceed_count += 1

    p_value = exceed_count / (int(n_permutations) + 1)
    return float(observed_statistic), float(p_value)


def compute_md_variant_absolute_metric_significance_df(
    family_id: str,
    metadata_df: pd.DataFrame,
    interaction_count_block_by_metric: dict[str, pd.DataFrame],
    interaction_count_replicate_by_metric: dict[str, pd.DataFrame],
    global_sasa_replicate_df: pd.DataFrame,
    adjust_method: str = "fdr_bh",
) -> pd.DataFrame:
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    if family_metadata_df.empty:
        return pd.DataFrame()

    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    system_order = compute_md_variant_system_order(metadata_df, family_id)
    variant_systems = [system_name for system_name in system_order if system_name != wildtype_system]

    metric_specs = [
        (
            "residue_contacts",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["residue_contacts"],
            interaction_count_block_by_metric["residue_contacts"],
            "block_mean",
            "block_permutation_ks",
        ),
        (
            "hbond_counts",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["hbond_counts"],
            interaction_count_block_by_metric["hbond_counts"],
            "block_mean",
            "block_permutation_ks",
        ),
        (
            "saltbridge",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["saltbridge"],
            interaction_count_block_by_metric["saltbridge"],
            "block_mean",
            "block_permutation_ks",
        ),
        (
            "global_sasa",
            "Global SASA",
            global_sasa_replicate_df.loc[global_sasa_replicate_df["family"].eq(family_id)].copy(),
            "global_mean_sasa",
            "welch_ttest",
        ),
    ]

    rows = []
    for metric_key, metric_label, metric_df, value_column, test_name in metric_specs:
        metric_df = metric_df.loc[metric_df["system"].isin(system_order)].copy()
        wildtype_values = (
            metric_df.loc[metric_df["system"].eq(wildtype_system), value_column]
            .astype(float)
            .dropna()
            .to_numpy()
        )
        for system_name in variant_systems:
            variant_values = (
                metric_df.loc[metric_df["system"].eq(system_name), value_column]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            if test_name == "block_permutation_ks":
                statistic, p_value = _compute_block_permutation_ks_test(
                    variant_values,
                    wildtype_values,
                )
            else:
                statistic, p_value = _compute_welch_ttest(variant_values, wildtype_values)
            variant_label = family_metadata_df.loc[
                family_metadata_df["system"].eq(system_name),
                "short_label",
            ].iat[0]
            rows.append(
                {
                    "family": family_id,
                    "system": system_name,
                    "short_label": variant_label,
                    "wildtype_system": wildtype_system,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "test_name": test_name,
                    "n_variant_samples": int(variant_values.size),
                    "n_wildtype_samples": int(wildtype_values.size),
                    "variant_mean": float(np.mean(variant_values)) if variant_values.size else np.nan,
                    "wildtype_mean": float(np.mean(wildtype_values)) if wildtype_values.size else np.nan,
                    "mean_difference": (
                        float(np.mean(variant_values) - np.mean(wildtype_values))
                        if variant_values.size and wildtype_values.size
                        else np.nan
                    ),
                    "test_statistic": statistic,
                    "p_value": p_value,
                }
            )

    significance_df = pd.DataFrame(rows)
    if significance_df.empty:
        return significance_df

    significance_df["p_value_fdr"] = np.nan
    significance_df["significant_fdr"] = False
    valid_mask = significance_df["p_value"].notna()
    if valid_mask.any():
        reject, p_value_fdr, _, _ = multipletests(
            significance_df.loc[valid_mask, "p_value"].to_numpy(dtype=float),
            method=adjust_method,
        )
        significance_df.loc[valid_mask, "p_value_fdr"] = p_value_fdr
        significance_df.loc[valid_mask, "significant_fdr"] = reject

    significance_df["stars"] = significance_df["p_value_fdr"].map(significance_stars)
    significance_df["direction_vs_wt"] = np.where(
        significance_df["mean_difference"] > 0,
        "higher",
        np.where(significance_df["mean_difference"] < 0, "lower", "same"),
    )
    return significance_df.sort_values(
        ["family", "metric_key", "system"]
    ).reset_index(drop=True)


def aggregate_md_variant_pair_metric(
    metric_df: pd.DataFrame,
    replicate_count_by_system: dict[str, int],
) -> pd.DataFrame:
    per_replicate_df = (
        metric_df.groupby(["system", "replicate", "pair_key"], as_index=False)
        .agg(
            pair_label=("pair_label", "first"),
            resid_a=("resid_a", "first"),
            resname_a=("resname_a", "first"),
            resid_b=("resid_b", "first"),
            resname_b=("resname_b", "first"),
            occupancy=("occupancy", "mean"),
        )
    )
    summary_df = (
        per_replicate_df.groupby(["system", "pair_key"], as_index=False)
        .agg(
            pair_label=("pair_label", "first"),
            resid_a=("resid_a", "first"),
            resname_a=("resname_a", "first"),
            resid_b=("resid_b", "first"),
            resname_b=("resname_b", "first"),
            occupancy_sum=("occupancy", "sum"),
            replicates_present=("replicate", "nunique"),
        )
    )
    summary_df["replicate_count"] = summary_df["system"].map(replicate_count_by_system).astype(int)
    summary_df["occupancy_mean"] = summary_df["occupancy_sum"] / summary_df["replicate_count"]
    return summary_df


def compute_md_variant_pair_replicate_table(
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(
            columns=[
                "system",
                "replicate",
                "replicate_index",
                "pair_key",
                "pair_label",
                "resid_a",
                "resname_a",
                "resid_b",
                "resname_b",
                "occupancy_mean",
            ]
        )

    return (
        metric_df.groupby(
            ["system", "replicate", "replicate_index", "pair_key"],
            as_index=False,
        )
        .agg(
            pair_label=("pair_label", "first"),
            resid_a=("resid_a", "first"),
            resname_a=("resname_a", "first"),
            resid_b=("resid_b", "first"),
            resname_b=("resname_b", "first"),
            occupancy_mean=("occupancy", "mean"),
        )
        .sort_values(["system", "replicate_index", "pair_key"])
        .reset_index(drop=True)
    )


def compute_md_variant_pair_residue_replicate_table(
    metric_df: pd.DataFrame,
) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(
            columns=[
                "system",
                "replicate",
                "replicate_index",
                "resid",
                "resname",
                "occupancy_sum",
                "n_pairs",
            ]
        )

    per_replicate_df = (
        metric_df.groupby(
            ["system", "replicate", "replicate_index", "pair_key"],
            as_index=False,
        )
        .agg(
            resid_a=("resid_a", "first"),
            resname_a=("resname_a", "first"),
            resid_b=("resid_b", "first"),
            resname_b=("resname_b", "first"),
            occupancy=("occupancy", "mean"),
        )
    )

    endpoint_tables = []
    for suffix in ("a", "b"):
        endpoint_tables.append(
            per_replicate_df[
                [
                    "system",
                    "replicate",
                    "replicate_index",
                    "pair_key",
                    f"resid_{suffix}",
                    f"resname_{suffix}",
                    "occupancy",
                ]
            ].rename(
                columns={
                    f"resid_{suffix}": "resid",
                    f"resname_{suffix}": "resname",
                }
            )
        )

    residue_replicate_df = pd.concat(endpoint_tables, ignore_index=True)
    residue_replicate_df["resid"] = pd.to_numeric(
        residue_replicate_df["resid"],
        errors="coerce",
    ).astype("Int64")
    residue_replicate_df = residue_replicate_df.dropna(subset=["resid"]).copy()
    residue_replicate_df["resid"] = residue_replicate_df["resid"].astype(int)
    residue_replicate_df["occupancy"] = pd.to_numeric(
        residue_replicate_df["occupancy"],
        errors="coerce",
    ).fillna(0.0)

    return (
        residue_replicate_df.groupby(
            ["system", "replicate", "replicate_index", "resid", "resname"],
            as_index=False,
        )
        .agg(
            occupancy_sum=("occupancy", "sum"),
            n_pairs=("pair_key", "nunique"),
        )
        .sort_values(["system", "replicate_index", "resid"])
        .reset_index(drop=True)
    )


def compute_md_variant_system_order(metadata_df: pd.DataFrame, family_id: str) -> list[str]:
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    family_metadata_df["sort_order"] = np.where(
        family_metadata_df["is_wildtype"],
        -1,
        family_metadata_df["mutation_count"],
    )
    family_metadata_df = family_metadata_df.sort_values(
        ["is_wildtype", "sort_order", "system"],
        ascending=[False, True, True],
    )
    return family_metadata_df["system"].tolist()


def compute_md_variant_order(metadata_df: pd.DataFrame, family_id: str) -> list[str]:
    family_metadata_df = metadata_df.loc[
        metadata_df["family"].eq(family_id) & (~metadata_df["is_wildtype"])
    ].copy()
    family_metadata_df = family_metadata_df.sort_values(
        ["mutation_count", "system"],
        ascending=[True, True],
    )
    return family_metadata_df["system"].tolist()


def build_md_variant_family_color_map(
    metadata_df: pd.DataFrame,
    family_id: str,
) -> dict[str, object]:
    variant_order = compute_md_variant_order(metadata_df, family_id)
    color_values = plt.get_cmap("tab20", max(1, len(variant_order)))
    return {
        system_name: color_values(index)
        for index, system_name in enumerate(variant_order)
    }


def compute_md_variant_residue_delta_table(
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    family_id: str,
    value_columns: list[str],
) -> pd.DataFrame:
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    wildtype_df = summary_df.loc[summary_df["system"].eq(wildtype_system)].copy()

    wildtype_column_map = {"resname": "wt_resname"}
    for value_column in value_columns:
        wildtype_column_map[f"{value_column}_mean"] = f"wt_{value_column}_mean"
        wildtype_column_map[f"{value_column}_std"] = f"wt_{value_column}_std"
    wildtype_df = wildtype_df.rename(columns=wildtype_column_map)
    wildtype_df = wildtype_df[
        ["resid", *wildtype_column_map.values()]
    ]

    family_rows = []
    for row in family_metadata_df.loc[~family_metadata_df["is_wildtype"]].itertuples(index=False):
        variant_df = summary_df.loc[summary_df["system"].eq(row.system)].copy()
        variant_df = variant_df.merge(wildtype_df, on="resid", how="inner")
        variant_df["family"] = family_id
        variant_df["wt_system"] = wildtype_system
        variant_df["variant_system"] = row.system
        variant_df["variant_label"] = get_md_variant_display_label_from_row(row)
        variant_df["mutation_count"] = int(row.mutation_count)
        variant_df["is_mutation_site"] = variant_df["resid"].isin(row.mutation_resids_list)
        for value_column in value_columns:
            variant_df[f"delta_{value_column}"] = (
                variant_df[f"{value_column}_mean"] - variant_df[f"wt_{value_column}_mean"]
            )
        family_rows.append(variant_df)

    if not family_rows:
        return pd.DataFrame()
    return pd.concat(family_rows, ignore_index=True)


def compute_md_variant_pair_delta_table(
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    family_id: str,
    local_residue_sets: dict[str, set[int]],
) -> pd.DataFrame:
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    wildtype_df = summary_df.loc[summary_df["system"].eq(wildtype_system)].copy()
    wildtype_df = wildtype_df.rename(
        columns={
            "occupancy_mean": "wt_occupancy_mean",
            "pair_label": "wt_pair_label",
            "resid_a": "wt_resid_a",
            "resname_a": "wt_resname_a",
            "resid_b": "wt_resid_b",
            "resname_b": "wt_resname_b",
        }
    )

    family_rows = []
    for row in family_metadata_df.loc[~family_metadata_df["is_wildtype"]].itertuples(index=False):
        variant_df = summary_df.loc[summary_df["system"].eq(row.system)].copy()
        merged_df = variant_df.merge(
            wildtype_df[
                [
                    "pair_key",
                    "wt_pair_label",
                    "wt_resid_a",
                    "wt_resname_a",
                    "wt_resid_b",
                    "wt_resname_b",
                    "wt_occupancy_mean",
                ]
            ],
            on="pair_key",
            how="outer",
        )
        merged_df["pair_label"] = merged_df["pair_label"].combine_first(merged_df["wt_pair_label"])
        merged_df["resid_a"] = merged_df["resid_a"].combine_first(merged_df["wt_resid_a"]).astype(int)
        merged_df["resname_a"] = merged_df["resname_a"].combine_first(merged_df["wt_resname_a"])
        merged_df["resid_b"] = merged_df["resid_b"].combine_first(merged_df["wt_resid_b"]).astype(int)
        merged_df["resname_b"] = merged_df["resname_b"].combine_first(merged_df["wt_resname_b"])
        merged_df["occupancy_mean"] = merged_df["occupancy_mean"].fillna(0.0)
        merged_df["wt_occupancy_mean"] = merged_df["wt_occupancy_mean"].fillna(0.0)
        merged_df["delta_occupancy"] = (
            merged_df["occupancy_mean"] - merged_df["wt_occupancy_mean"]
        )
        merged_df["family"] = family_id
        merged_df["wt_system"] = wildtype_system
        merged_df["variant_system"] = row.system
        merged_df["variant_label"] = get_md_variant_display_label_from_row(row)

        mutation_resids = set(int(value) for value in row.mutation_resids_list)
        local_resids = local_residue_sets.get(str(row.system), set())
        merged_df["contains_mutation_pair"] = (
            merged_df["resid_a"].isin(mutation_resids)
            | merged_df["resid_b"].isin(mutation_resids)
        )
        merged_df["is_local_pair"] = (
            merged_df["resid_a"].isin(local_resids)
            | merged_df["resid_b"].isin(local_resids)
        )
        family_rows.append(merged_df)

    if not family_rows:
        return pd.DataFrame()
    return pd.concat(family_rows, ignore_index=True)


def compute_md_variant_frame_delta_table(
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    family_id: str,
) -> pd.DataFrame:
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    wildtype_df = summary_df.loc[summary_df["system"].eq(wildtype_system)].copy()
    wildtype_df = wildtype_df.rename(
        columns={
            "count_mean": "wt_count_mean",
            "count_std": "wt_count_std",
        }
    )
    wildtype_df = wildtype_df[["frame", "wt_count_mean", "wt_count_std"]]

    family_rows = []
    for row in family_metadata_df.loc[~family_metadata_df["is_wildtype"]].itertuples(index=False):
        variant_df = summary_df.loc[summary_df["system"].eq(row.system)].copy()
        variant_df = variant_df.merge(wildtype_df, on="frame", how="inner")
        variant_df["family"] = family_id
        variant_df["wt_system"] = wildtype_system
        variant_df["variant_system"] = row.system
        variant_df["variant_label"] = get_md_variant_display_label_from_row(row)
        variant_df["mutation_count"] = int(row.mutation_count)
        variant_df["delta_count"] = (
            variant_df["count_mean"] - variant_df["wt_count_mean"]
        )
        family_rows.append(variant_df)

    if not family_rows:
        return pd.DataFrame()
    return pd.concat(family_rows, ignore_index=True)


def compute_md_family_residue_interest_df(
    rmsf_delta_df: pd.DataFrame,
    family_id: str,
) -> pd.DataFrame:
    family_delta_df = rmsf_delta_df.loc[rmsf_delta_df["family"].eq(family_id)].copy()
    if family_delta_df.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "resid",
                "resname",
                "mean_delta_rmsf",
                "mean_abs_delta_rmsf",
                "max_abs_delta_rmsf",
                "mean_abs_standardized_delta_rmsf",
                "n_variants",
                "fraction_variants_lower_than_wt",
                "interest_rank",
            ]
        )

    pooled_sd = np.sqrt(
        np.square(family_delta_df["rmsf_std"].astype(float))
        + np.square(family_delta_df["wt_rmsf_std"].astype(float))
    )
    family_delta_df["standardized_delta_rmsf"] = np.where(
        pooled_sd > 0.0,
        family_delta_df["delta_rmsf"].astype(float) / pooled_sd,
        np.nan,
    )
    interest_df = (
        family_delta_df.groupby(["family", "resid", "resname"], as_index=False)
        .agg(
            mean_delta_rmsf=("delta_rmsf", "mean"),
            mean_abs_delta_rmsf=("delta_rmsf", lambda values: np.mean(np.abs(values))),
            max_abs_delta_rmsf=("delta_rmsf", lambda values: np.max(np.abs(values))),
            mean_abs_standardized_delta_rmsf=(
                "standardized_delta_rmsf",
                lambda values: np.nanmean(np.abs(values)),
            ),
            n_variants=("variant_system", "nunique"),
            fraction_variants_lower_than_wt=("delta_rmsf", lambda values: np.mean(np.asarray(values) < 0.0)),
        )
        .sort_values(
            [
                "mean_abs_delta_rmsf",
                "mean_abs_standardized_delta_rmsf",
                "max_abs_delta_rmsf",
                "resid",
            ],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    interest_df["interest_rank"] = np.arange(1, len(interest_df) + 1, dtype=int)
    return interest_df


def compute_md_family_pair_interest_df(
    pair_summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    family_id: str,
) -> pd.DataFrame:
    system_order = compute_md_variant_system_order(metadata_df, family_id)
    family_pair_df = pair_summary_df.loc[pair_summary_df["system"].isin(system_order)].copy()
    if family_pair_df.empty:
        return pd.DataFrame(
            columns=[
                "pair_label",
                "max_abs_delta_vs_wt",
                "mean_abs_delta_vs_wt",
                "mean_delta_vs_wt",
                "n_systems_present",
                "interest_rank",
            ]
        )

    pivot_df = family_pair_df.pivot(
        index="pair_label",
        columns="system",
        values="occupancy_mean",
    ).fillna(0.0)
    pivot_df = pivot_df.reindex(columns=system_order, fill_value=0.0)
    delta_to_wt_df = pivot_df.sub(pivot_df[system_order[0]], axis=0)

    interest_df = pd.DataFrame(
        {
            "pair_label": delta_to_wt_df.index,
            "max_abs_delta_vs_wt": delta_to_wt_df.abs().max(axis=1).to_numpy(dtype=float),
            "mean_abs_delta_vs_wt": delta_to_wt_df.abs().mean(axis=1).to_numpy(dtype=float),
            "mean_delta_vs_wt": delta_to_wt_df.mean(axis=1).to_numpy(dtype=float),
            "n_systems_present": (pivot_df > 0.0).sum(axis=1).to_numpy(dtype=int),
        }
    ).sort_values(
        ["max_abs_delta_vs_wt", "mean_abs_delta_vs_wt", "pair_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    interest_df["interest_rank"] = np.arange(1, len(interest_df) + 1, dtype=int)
    return interest_df


def select_md_variant_top_variable_pairs(
    family_id: str,
    metadata_df: pd.DataFrame,
    pair_summary_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    pair_interest_df = compute_md_family_pair_interest_df(
        pair_summary_df=pair_summary_df,
        metadata_df=metadata_df,
        family_id=family_id,
    )
    top_pair_labels = pair_interest_df.head(top_n)["pair_label"].tolist()
    return pair_summary_df.loc[pair_summary_df["pair_label"].isin(top_pair_labels)].copy()


def compute_md_variant_signal_table(
    metadata_df: pd.DataFrame,
    local_residue_sets: dict[str, set[int]],
    rmsf_delta_df: pd.DataFrame,
    sasa_delta_df: pd.DataFrame,
    ss_residue_delta_df: pd.DataFrame,
    ss_counts_summary_df: pd.DataFrame,
    pair_delta_by_metric: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metric_alias_map = {
        "residue_contact_pairs": "contact",
        "saltbridge_pairs": "saltbridge",
        "hbond_pairs": "hbond",
    }
    signal_rows = []

    for row in metadata_df.loc[~metadata_df["is_wildtype"]].itertuples(index=False):
        family_id = str(row.family)
        variant_system = str(row.system)
        wildtype_system = str(row.wt_system)
        local_resids = local_residue_sets.get(variant_system, set())
        mutation_resids = set(int(value) for value in row.mutation_resids_list)

        rmsf_variant_df = rmsf_delta_df.loc[
            rmsf_delta_df["variant_system"].eq(variant_system)
        ].copy()
        sasa_variant_df = sasa_delta_df.loc[
            sasa_delta_df["variant_system"].eq(variant_system)
        ].copy()
        ss_variant_df = ss_residue_delta_df.loc[
            ss_residue_delta_df["variant_system"].eq(variant_system)
        ].copy()
        ss_variant_summary = ss_counts_summary_df.loc[
            ss_counts_summary_df["system"].eq(variant_system)
        ].iloc[0]
        ss_wildtype_summary = ss_counts_summary_df.loc[
            ss_counts_summary_df["system"].eq(wildtype_system)
        ].iloc[0]

        rmsf_local = rmsf_variant_df.loc[
            rmsf_variant_df["resid"].isin(local_resids),
            "delta_rmsf",
        ]
        sasa_local = sasa_variant_df.loc[
            sasa_variant_df["resid"].isin(local_resids),
            "delta_mean_sasa",
        ]
        ss_local = ss_variant_df.loc[
            ss_variant_df["resid"].isin(local_resids),
            "delta_ordered_occupancy",
        ]
        rmsf_mutation = rmsf_variant_df.loc[
            rmsf_variant_df["resid"].isin(mutation_resids),
            "delta_rmsf",
        ]
        sasa_mutation = sasa_variant_df.loc[
            sasa_variant_df["resid"].isin(mutation_resids),
            "delta_mean_sasa",
        ]

        signal_row = {
            "family": family_id,
            "variant_system": variant_system,
            "variant_label": get_md_variant_display_label_from_row(row),
            "wt_system": wildtype_system,
            "mutation_count": int(row.mutation_count),
            "neighborhood_residue_count": len(local_resids),
            "median_tm_celsius": float(getattr(row, "median_tm_celsius", np.nan))
            if pd.notna(getattr(row, "median_tm_celsius", np.nan))
            else np.nan,
            "wildtype_median_tm_celsius": float(getattr(row, "wildtype_median_tm_celsius", np.nan))
            if pd.notna(getattr(row, "wildtype_median_tm_celsius", np.nan))
            else np.nan,
            "delta_tm_to_wildtype_celsius": float(getattr(row, "delta_tm_to_wildtype_celsius", np.nan))
            if pd.notna(getattr(row, "delta_tm_to_wildtype_celsius", np.nan))
            else np.nan,
            "delta_rmsf_global_mean": float(rmsf_variant_df["delta_rmsf"].mean()),
            "delta_rmsf_neighborhood_mean": float(rmsf_local.mean()) if len(rmsf_local) else np.nan,
            "delta_rmsf_mutation_mean": float(rmsf_mutation.mean()) if len(rmsf_mutation) else np.nan,
            "delta_sasa_global_mean": float(sasa_variant_df["delta_mean_sasa"].mean()),
            "delta_sasa_neighborhood_mean": float(sasa_local.mean()) if len(sasa_local) else np.nan,
            "delta_sasa_mutation_mean": float(sasa_mutation.mean()) if len(sasa_mutation) else np.nan,
            "delta_ordered_ss_global_mean": float(ss_variant_df["delta_ordered_occupancy"].mean()),
            "delta_ordered_ss_neighborhood_mean": float(ss_local.mean()) if len(ss_local) else np.nan,
            "delta_helix_fraction_global": float(
                ss_variant_summary["helix_fraction_mean"] - ss_wildtype_summary["helix_fraction_mean"]
            ),
            "delta_strand_fraction_global": float(
                ss_variant_summary["strand_fraction_mean"] - ss_wildtype_summary["strand_fraction_mean"]
            ),
            "delta_ordered_fraction_global": float(
                ss_variant_summary["ordered_fraction_mean"] - ss_wildtype_summary["ordered_fraction_mean"]
            ),
        }

        for metric_name, delta_df in pair_delta_by_metric.items():
            metric_base_name = metric_alias_map[metric_name]
            variant_pair_df = delta_df.loc[
                delta_df["variant_system"].eq(variant_system)
            ].copy()
            local_pair_df = variant_pair_df.loc[variant_pair_df["is_local_pair"]]

            signal_row[f"{metric_base_name}_gain_total"] = float(
                variant_pair_df["delta_occupancy"].clip(lower=0.0).sum()
            )
            signal_row[f"{metric_base_name}_loss_total"] = float(
                (-variant_pair_df["delta_occupancy"].clip(upper=0.0)).sum()
            )
            signal_row[f"{metric_base_name}_gain_local"] = float(
                local_pair_df["delta_occupancy"].clip(lower=0.0).sum()
            )
            signal_row[f"{metric_base_name}_loss_local"] = float(
                (-local_pair_df["delta_occupancy"].clip(upper=0.0)).sum()
            )
            total_gain = signal_row[f"{metric_base_name}_gain_total"]
            signal_row[f"{metric_base_name}_gain_local_fraction"] = (
                signal_row[f"{metric_base_name}_gain_local"] / total_gain
                if total_gain > 0
                else 0.0
            )

        signal_rows.append(signal_row)

    signal_df = pd.DataFrame(signal_rows)
    if signal_df.empty:
        return signal_df

    signal_df["rmsf_neighborhood_gain"] = -signal_df["delta_rmsf_neighborhood_mean"]
    signal_df["sasa_neighborhood_gain"] = -signal_df["delta_sasa_neighborhood_mean"]
    signal_df["ordered_ss_neighborhood_gain"] = signal_df["delta_ordered_ss_neighborhood_mean"]

    for column_name in MD_VARIANT_SIGNAL_COLUMNS:
        signal_df[f"{column_name}_rank"] = signal_df[column_name].rank(
            method="average",
            pct=True,
        )

    signal_df["composite_stabilizing_score"] = signal_df[
        [f"{column_name}_rank" for column_name in MD_VARIANT_SIGNAL_COLUMNS]
    ].mean(axis=1)
    signal_df["global_rank"] = signal_df["composite_stabilizing_score"].rank(
        method="dense",
        ascending=False,
    )
    signal_df["family_rank"] = signal_df.groupby("family")[
        "composite_stabilizing_score"
    ].rank(
        method="dense",
        ascending=False,
    )
    return signal_df.sort_values(
        ["global_rank", "family", "variant_system"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_md_variant_family_pattern_summary(
    variant_signal_df: pd.DataFrame,
) -> pd.DataFrame:
    if variant_signal_df.empty:
        return pd.DataFrame()

    family_rows = []
    for family_id, family_df in variant_signal_df.groupby("family"):
        family_rows.append(
            {
                "family": family_id,
                "n_variants": int(len(family_df)),
                "fraction_lower_local_rmsf": float((family_df["delta_rmsf_neighborhood_mean"] < 0.0).mean()),
                "fraction_lower_local_sasa": float((family_df["delta_sasa_neighborhood_mean"] < 0.0).mean()),
                "fraction_higher_local_ss": float((family_df["delta_ordered_ss_neighborhood_mean"] > 0.0).mean()),
                "fraction_contact_gain": float((family_df["contact_gain_total"] > 0.0).mean()),
                "fraction_saltbridge_gain": float((family_df["saltbridge_gain_total"] > 0.0).mean()),
                "fraction_hbond_gain": float((family_df["hbond_gain_total"] > 0.0).mean()),
                "median_composite_score": float(family_df["composite_stabilizing_score"].median()),
                "median_contact_local_fraction": float(family_df["contact_gain_local_fraction"].median()),
                "median_hbond_local_fraction": float(family_df["hbond_gain_local_fraction"].median()),
            }
        )

    family_rows.append(
        {
            "family": "ALL",
            "n_variants": int(len(variant_signal_df)),
            "fraction_lower_local_rmsf": float((variant_signal_df["delta_rmsf_neighborhood_mean"] < 0.0).mean()),
            "fraction_lower_local_sasa": float((variant_signal_df["delta_sasa_neighborhood_mean"] < 0.0).mean()),
            "fraction_higher_local_ss": float((variant_signal_df["delta_ordered_ss_neighborhood_mean"] > 0.0).mean()),
            "fraction_contact_gain": float((variant_signal_df["contact_gain_total"] > 0.0).mean()),
            "fraction_saltbridge_gain": float((variant_signal_df["saltbridge_gain_total"] > 0.0).mean()),
            "fraction_hbond_gain": float((variant_signal_df["hbond_gain_total"] > 0.0).mean()),
            "median_composite_score": float(variant_signal_df["composite_stabilizing_score"].median()),
            "median_contact_local_fraction": float(variant_signal_df["contact_gain_local_fraction"].median()),
            "median_hbond_local_fraction": float(variant_signal_df["hbond_gain_local_fraction"].median()),
        }
    )

    return pd.DataFrame(family_rows)


def _compute_md_variant_group_zscores(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").astype(float)
    std_value = float(numeric_values.std(ddof=0))
    if not np.isfinite(std_value) or np.isclose(std_value, 0.0):
        return pd.Series(np.zeros(len(numeric_values), dtype=float), index=values.index)
    return (numeric_values - float(numeric_values.mean())) / std_value


def summarize_md_variant_pair_deltas_by_residue(
    pair_delta_df: pd.DataFrame,
) -> pd.DataFrame:
    if pair_delta_df.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "variant_system",
                "variant_label",
                "resid",
                "resname",
                "score_raw",
                "gain_raw",
                "loss_raw",
                "n_pairs",
                "any_local_pair",
                "any_mutation_pair",
            ]
        )

    endpoint_tables = []
    for suffix in ("a", "b"):
        endpoint_tables.append(
            pair_delta_df[
                [
                    "family",
                    "variant_system",
                    "variant_label",
                    f"resid_{suffix}",
                    f"resname_{suffix}",
                    "pair_key",
                    "delta_occupancy",
                    "is_local_pair",
                    "contains_mutation_pair",
                ]
            ].rename(
                columns={
                    f"resid_{suffix}": "resid",
                    f"resname_{suffix}": "resname",
                }
            )
        )

    residue_delta_df = pd.concat(endpoint_tables, ignore_index=True)
    residue_delta_df["resid"] = pd.to_numeric(
        residue_delta_df["resid"],
        errors="coerce",
    ).astype("Int64")
    residue_delta_df = residue_delta_df.dropna(subset=["resid"]).copy()
    residue_delta_df["resid"] = residue_delta_df["resid"].astype(int)
    residue_delta_df["delta_occupancy"] = pd.to_numeric(
        residue_delta_df["delta_occupancy"],
        errors="coerce",
    ).fillna(0.0)
    residue_delta_df["gain_component"] = residue_delta_df["delta_occupancy"].clip(lower=0.0)
    residue_delta_df["loss_component"] = residue_delta_df["delta_occupancy"].clip(upper=0.0)

    return (
        residue_delta_df.groupby(
            ["family", "variant_system", "variant_label", "resid", "resname"],
            as_index=False,
        )
        .agg(
            score_raw=("delta_occupancy", "sum"),
            gain_raw=("gain_component", "sum"),
            loss_raw=("loss_component", "sum"),
            n_pairs=("pair_key", "nunique"),
            any_local_pair=("is_local_pair", "max"),
            any_mutation_pair=("contains_mutation_pair", "max"),
        )
        .sort_values(["family", "variant_system", "resid"])
        .reset_index(drop=True)
    )


def build_md_variant_molstar_residue_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    family_id: str,
    metric_name: str,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    metric_key = str(metric_name).strip().lower()
    metric_alias_map = {
        "rmsf": "rmsf",
        "hhbond": "hbond_pairs",
        "hhbonds": "hbond_pairs",
        "hbond": "hbond_pairs",
        "hbonds": "hbond_pairs",
        "hbond_pairs": "hbond_pairs",
        "saltbridge": "saltbridge_pairs",
        "saltbridges": "saltbridge_pairs",
        "saltbridge_pairs": "saltbridge_pairs",
    }
    resolved_metric_name = metric_alias_map.get(metric_key, metric_key)

    if resolved_metric_name == "rmsf":
        score_df = md_variant_bundle.rmsf_delta_df.loc[
            md_variant_bundle.rmsf_delta_df["family"].eq(family_id),
            [
                "family",
                "variant_system",
                "variant_label",
                "resid",
                "resname",
                "delta_rmsf",
                "is_mutation_site",
            ],
        ].rename(
            columns={
                "delta_rmsf": "score_raw",
                "is_mutation_site": "any_mutation_pair",
            }
        )
        score_df["any_local_pair"] = [
            int(resid) in md_variant_bundle.local_residue_sets.get(str(variant_system), set())
            for variant_system, resid in zip(
                score_df["variant_system"],
                score_df["resid"],
            )
        ]
    elif resolved_metric_name in {"hbond_pairs", "saltbridge_pairs"}:
        pair_delta_df = md_variant_bundle.pair_delta_by_metric[resolved_metric_name].loc[
            md_variant_bundle.pair_delta_by_metric[resolved_metric_name]["family"].eq(family_id)
        ].copy()
        score_df = summarize_md_variant_pair_deltas_by_residue(pair_delta_df)
    else:
        raise KeyError(
            "Unsupported Mol* residue metric. Expected one of "
            "'rmsf', 'hbond_pairs', or 'saltbridge_pairs'."
        )

    if score_df.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "variant_system",
                "variant_label",
                "metric_name",
                "metric_label",
                "resid",
                "resname",
                "score_raw",
                "score_z",
                "abs_score_z",
                "passes_threshold",
                "direction",
                "z_threshold",
                "bfactor_magnitude",
            ]
        )

    score_df["metric_name"] = resolved_metric_name
    metric_label_map = {
        "rmsf": "RMSF",
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }
    score_df["metric_label"] = metric_label_map[resolved_metric_name]
    score_df["score_raw"] = pd.to_numeric(score_df["score_raw"], errors="coerce").fillna(0.0)
    score_df["score_z"] = (
        score_df.groupby("variant_system", group_keys=False)["score_raw"]
        .apply(_compute_md_variant_group_zscores)
        .astype(float)
    )
    score_df["abs_score_z"] = score_df["score_z"].abs()
    score_df["passes_threshold"] = score_df["abs_score_z"] >= float(z_threshold)
    score_df["direction"] = np.where(
        score_df["score_z"] >= float(z_threshold),
        "positive",
        np.where(score_df["score_z"] <= -float(z_threshold), "negative", "neutral"),
    )
    score_df["z_threshold"] = float(z_threshold)
    score_df["bfactor_magnitude"] = np.where(
        score_df["passes_threshold"],
        np.clip(score_df["abs_score_z"] * 20.0, 0.0, 99.99),
        0.0,
    )
    return score_df.sort_values(["variant_system", "resid"]).reset_index(drop=True)


def build_md_variant_global_rmsd_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    metric_dir: Path,
) -> pd.DataFrame:
    rmsd_raw_df = read_md_variant_metric_table(
        metric_dir=metric_dir,
        metric_name="rmsd",
        metric_window_suffix=md_variant_bundle.metric_window_suffix,
    )
    if rmsd_raw_df.empty:
        return pd.DataFrame()

    value_column = "ca_rmsd" if "ca_rmsd" in rmsd_raw_df.columns else "rmsd"
    if value_column not in rmsd_raw_df.columns:
        raise KeyError(
            "Could not find an RMSD value column in the MD RMSD table. "
            f"Available columns: {sorted(rmsd_raw_df.columns)}"
        )

    rmsd_raw_df[value_column] = pd.to_numeric(rmsd_raw_df[value_column], errors="coerce")
    rmsd_raw_df = rmsd_raw_df.dropna(subset=[value_column]).copy()

    per_replicate_df = (
        rmsd_raw_df.groupby(["system", "replicate", "replicate_index"], as_index=False)
        .agg(global_rmsd_mean=(value_column, "mean"))
    )
    system_summary_df = (
        per_replicate_df.groupby("system", as_index=False)
        .agg(
            n_replicates=("replicate", "nunique"),
            global_rmsd_mean=("global_rmsd_mean", "mean"),
            global_rmsd_std=("global_rmsd_mean", "std"),
        )
    )
    system_summary_df["global_rmsd_std"] = system_summary_df["global_rmsd_std"].fillna(0.0)

    metadata_columns = [
        column_name
        for column_name in [
            "system",
            "family",
            "is_wildtype",
            "short_label",
            "tm_display_label",
            "simple_label",
            "simple_tm_display_label",
        ]
        if column_name in md_variant_bundle.metadata_df.columns
    ]
    metadata_df = md_variant_bundle.metadata_df[metadata_columns].drop_duplicates()
    summary_df = system_summary_df.merge(
        metadata_df,
        on="system",
        how="inner",
        validate="one_to_one",
    )

    family_tables = []
    for family_id in md_variant_bundle.family_ids:
        family_df = summary_df.loc[summary_df["family"].eq(family_id)].copy()
        if family_df.empty:
            continue

        wildtype_match_df = family_df.loc[family_df["is_wildtype"]]
        if wildtype_match_df.empty:
            continue
        wildtype_row = wildtype_match_df.iloc[0]

        variant_df = family_df.loc[~family_df["is_wildtype"]].copy()
        if variant_df.empty:
            continue

        variant_df["wt_system"] = str(wildtype_row["system"])
        variant_df["wt_global_rmsd_mean"] = float(wildtype_row["global_rmsd_mean"])
        variant_df["delta_global_rmsd"] = (
            variant_df["global_rmsd_mean"] - variant_df["wt_global_rmsd_mean"]
        )
        variant_df["variant_label"] = variant_df.apply(
            get_md_variant_display_label_from_row,
            axis=1,
        )
        variant_df["score_z"] = _compute_md_variant_group_zscores(
            variant_df["delta_global_rmsd"]
        ).astype(float)
        family_tables.append(variant_df)

    if not family_tables:
        return pd.DataFrame()

    global_rmsd_summary_df = pd.concat(family_tables, ignore_index=True)
    global_rmsd_summary_df["metric_name"] = "rmsd"
    global_rmsd_summary_df["metric_label"] = "RMSD"
    global_rmsd_summary_df["abs_score_z"] = global_rmsd_summary_df["score_z"].abs()
    return global_rmsd_summary_df.sort_values(
        ["family", "system"]
    ).reset_index(drop=True)


def _write_md_variant_bfactor_pdb(
    template_pdb_path: Path,
    residue_bfactor_map: dict[int, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with template_pdb_path.open() as source_handle, output_path.open("w") as target_handle:
        for line in source_handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
                residue_number = int(line[22:26])
                bfactor_value = float(residue_bfactor_map.get(residue_number, 0.0))
                target_handle.write(f"{line[:60]}{bfactor_value:6.2f}{line[66:]}")
            else:
                target_handle.write(line)


def _write_transformed_pdb(
    template_pdb_path: Path,
    output_path: Path,
    rotation_matrix: np.ndarray | None = None,
    mobile_center: np.ndarray | None = None,
    reference_center: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rotation_matrix is None or mobile_center is None or reference_center is None:
        output_path.write_text(template_pdb_path.read_text())
        return

    rotation_matrix = np.asarray(rotation_matrix, dtype=float)
    mobile_center = np.asarray(mobile_center, dtype=float)
    reference_center = np.asarray(reference_center, dtype=float)

    with template_pdb_path.open() as source_handle, output_path.open("w") as target_handle:
        for line in source_handle:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
                try:
                    atom_xyz = np.asarray(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    )
                except ValueError:
                    target_handle.write(line)
                    continue

                aligned_xyz = apply_kabsch_transform(
                    xyz=atom_xyz[np.newaxis, :],
                    rotation_matrix=rotation_matrix,
                    mobile_center=mobile_center,
                    reference_center=reference_center,
                )[0]
                target_handle.write(
                    f"{line[:30]}"
                    f"{aligned_xyz[0]:8.3f}{aligned_xyz[1]:8.3f}{aligned_xyz[2]:8.3f}"
                    f"{line[54:]}"
                )
            else:
                target_handle.write(line)


def _write_combined_overlay_pdb(
    structure_entries: list[dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atom_serial_number = 1
    with output_path.open("w") as target_handle:
        for structure_entry in structure_entries:
            pdb_path = Path(structure_entry["pdb_path"])
            chain_id = str(structure_entry["chain_id"])[:1] or "A"
            last_residue_name = "UNK"
            last_residue_number = "   1"
            last_insertion_code = " "

            with pdb_path.open() as source_handle:
                for line in source_handle:
                    if line.startswith(("ATOM", "HETATM")):
                        last_residue_name = line[17:20].strip() or "UNK"
                        last_residue_number = line[22:26]
                        last_insertion_code = line[26:27] if len(line) >= 27 else " "
                        target_handle.write(
                            f"{line[:6]}{atom_serial_number:5d}{line[11:21]}{chain_id}{line[22:]}"
                        )
                        atom_serial_number += 1

            target_handle.write(
                f"TER   {atom_serial_number:5d}      {last_residue_name:>3} {chain_id}{last_residue_number}{last_insertion_code}\n"
            )
            atom_serial_number += 1

        target_handle.write("END\n")


def _write_simple_molstar_html(
    pdb_path: Path,
    html_path: Path,
    title: str,
    subtitle: str,
    viewer_height_px: int = 560,
    footer_note: str | None = None,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_text = Path(pdb_path).read_text()
    resolved_footer_note = (
        footer_note
        or "This page loads one aligned combined PDB directly into the standard Mol* viewer."
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css" />
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #101418;
    }}
    .wrap {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }}
    .header, .footer {{
      padding: 14px 18px;
      background: #ffffff;
      box-shadow: 0 1px 0 rgba(16, 20, 24, 0.08);
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .header p, .footer p {{
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
      color: #46515c;
    }}
    #app {{
      position: relative;
      width: 100%;
      height: {int(viewer_height_px)}px;
    }}
    #app :is(.msp-plugin, .msp-layout-standard, .msp-layout-expanded) {{
      min-height: {int(viewer_height_px)}px;
    }}
    #status {{
      margin: 12px 18px 0 18px;
      padding: 12px 14px;
      border-radius: 10px;
      background: #eef5ff;
      border: 1px solid #c8dcff;
      color: #1c3f76;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    #status[data-state="hidden"] {{
      display: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
    </div>
    <div id="status">Loading Mol* viewer...</div>
    <div id="app"></div>
    <div class="footer">
      <p>{html.escape(resolved_footer_note)}</p>
    </div>
  </div>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"></script>
  <script type="text/javascript">
    const pdbText = {json.dumps(pdb_text)};
    const statusEl = document.getElementById('status');
    const backgroundColor = 0xf5f6f8;
    const loadLabel = {json.dumps(title)};
    const MolstarViewer = window.molstar?.Viewer;

    function setStatus(message) {{
      if (!statusEl) return;
      if (!message) {{
        statusEl.textContent = '';
        statusEl.dataset.state = 'hidden';
        return;
      }}
      statusEl.textContent = message;
      delete statusEl.dataset.state;
    }}

    async function initialiseViewer() {{
      if (!MolstarViewer?.create) {{
        setStatus('Mol* failed to load from the CDN, so the viewer could not be created.');
        return;
      }}

      try {{
        const viewer = await MolstarViewer.create('app', {{
          layoutIsExpanded: true,
          layoutShowControls: true,
          layoutShowRemoteState: false,
          layoutShowSequence: true,
          layoutShowLog: false,
          layoutShowLeftPanel: true,
          viewportShowExpand: true,
          viewportShowSelectionMode: false,
          viewportShowAnimation: false,
          viewportBackgroundColor: backgroundColor,
        }});
        await viewer.loadStructureFromData(pdbText, 'pdb', {{
          dataLabel: loadLabel,
        }});
        setStatus('');
      }} catch (error) {{
        console.error('Mol* superposition viewer initialization failed.', error);
        setStatus(`Mol* superposition viewer initialization failed: ${{error?.message ?? error}}`);
      }}
    }}

    initialiseViewer();
  </script>
</body>
</html>
"""
    html_path.write_text(html_text)


def _build_md_variant_residue_color_map(
    residue_score_map: dict[int, float],
    z_threshold: float,
    score_limit: float | None = None,
    color_bins: int = 41,
) -> tuple[dict[int, str], float]:
    if not residue_score_map:
        return {}, max(float(z_threshold), 1.0)

    score_values = np.asarray(list(residue_score_map.values()), dtype=float)
    score_values = score_values[np.isfinite(score_values)]
    if not len(score_values):
        return {}, max(float(z_threshold), 1.0)

    resolved_limit = (
        float(np.nanmax(np.abs(score_values)))
        if score_limit is None
        else float(score_limit)
    )
    resolved_limit = max(resolved_limit, float(z_threshold), 1e-6)
    norm = TwoSlopeNorm(vmin=-resolved_limit, vcenter=0.0, vmax=resolved_limit)
    bin_edges = np.linspace(-resolved_limit, resolved_limit, max(int(color_bins), 3), dtype=float)
    color_cache: dict[float, str] = {}
    residue_color_map: dict[int, str] = {}

    for residue_number, score_value in residue_score_map.items():
        score_value = float(score_value)
        if not np.isfinite(score_value) or abs(score_value) < float(z_threshold):
            continue

        clipped_value = float(np.clip(score_value, -resolved_limit, resolved_limit))
        bin_index = int(np.clip(np.digitize([clipped_value], bin_edges, right=True)[0], 1, len(bin_edges) - 1))
        binned_value = float(bin_edges[bin_index])
        if binned_value not in color_cache:
            color_cache[binned_value] = to_hex(
                DEFAULT_RMSF_DELTA_CMAP(norm(binned_value)),
                keep_alpha=False,
            )
        residue_color_map[int(residue_number)] = color_cache[binned_value]

    return residue_color_map, resolved_limit


def _write_md_variant_molstar_html(
    pdb_path: Path,
    html_path: Path,
    title: str,
    subtitle: str,
    residue_color_map: dict[int, str],
    all_residue_numbers: list[int],
    score_limit: float,
    viewer_height_px: int = 560,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pdb_text = pdb_path.read_text()
    grouped_residue_colors: dict[str, list[int]] = {}
    for residue_number, color_hex in residue_color_map.items():
        grouped_residue_colors.setdefault(str(color_hex), []).append(int(residue_number))
    grouped_residue_colors = {
        color_hex: sorted(residue_numbers)
        for color_hex, residue_numbers in grouped_residue_colors.items()
    }
    colored_residue_numbers = sorted({int(residue_number) for residue_number in residue_color_map})
    neutral_residue_numbers = sorted(
        int(residue_number)
        for residue_number in all_residue_numbers
        if int(residue_number) not in residue_color_map
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css" />
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #101418;
    }}
    .wrap {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }}
    .header, .footer {{
      padding: 14px 18px;
      background: #ffffff;
      box-shadow: 0 1px 0 rgba(16, 20, 24, 0.08);
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .header p, .footer p {{
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
      color: #46515c;
    }}
    #app {{
      position: relative;
      width: 100%;
      height: {int(viewer_height_px)}px;
    }}
    #app :is(.msp-plugin, .msp-layout-standard, .msp-layout-expanded) {{
      min-height: {int(viewer_height_px)}px;
    }}
    .legend {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
      margin: 0 18px 12px 18px;
      font-size: 12px;
      color: #46515c;
    }}
    .legend-bar {{
      height: 12px;
      border-radius: 999px;
      background: linear-gradient(90deg, #2166ac 0%, #f7f7f7 50%, #b2182b 100%);
      border: 1px solid rgba(16, 20, 24, 0.12);
    }}
    #status {{
      margin: 12px 18px 0 18px;
      padding: 12px 14px;
      border-radius: 10px;
      background: #fff3cd;
      border: 1px solid #f1d58a;
      color: #5f4400;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    #status[data-state="loading"] {{
      background: #eef5ff;
      border-color: #c8dcff;
      color: #1c3f76;
    }}
    #status[data-state="hidden"] {{
      display: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
    </div>
    <div id="status" data-state="loading">Loading Mol* viewer...</div>
    <div id="app"></div>
    <div class="legend">
      <span>Lower than WT</span>
      <div class="legend-bar" aria-hidden="true"></div>
      <span>Higher than WT</span>
    </div>
    <div class="footer">
      <p>Color intensity saturates at |z| = {float(score_limit):0.2f}. Residues below the export threshold remain gray.</p>
    </div>
  </div>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"></script>
  <script type="text/javascript">
    const pdbText = {json.dumps(pdb_text)};
    const groupedResidueColors = {json.dumps(grouped_residue_colors, sort_keys=True)};
    const neutralResidueNumbers = {json.dumps(neutral_residue_numbers)};
    const statusEl = document.getElementById('status');
    const neutralColor = '#d9dde3';
    const backgroundColor = 0xf5f6f8;
    const loadLabel = {json.dumps(title)};
    const MolstarLib = window.molstar?.lib;

    function setStatus(message, state = 'error') {{
      if (!statusEl) return;
      if (!message) {{
        statusEl.textContent = '';
        statusEl.dataset.state = 'hidden';
        return;
      }}
      statusEl.textContent = message;
      statusEl.dataset.state = state;
    }}

    function colorHexToMolstarInt(colorHex) {{
      return Number.parseInt(String(colorHex).replace('#', ''), 16);
    }}

    function residuesToLoci(structure, residueNumbers) {{
      if (!Array.isArray(residueNumbers) || !residueNumbers.length) return null;
      const root = structure.root ?? structure;
      let loci = MolstarLib.structure.StructureElement.Loci.fromSchema(root, {{
        items: {{
          auth_seq_id: residueNumbers,
        }},
      }});
      if (!loci?.elements?.length) {{
        loci = MolstarLib.structure.StructureElement.Loci.fromSchema(root, {{
          items: {{
            label_seq_id: residueNumbers,
          }},
        }});
      }}
      return loci?.elements?.length ? loci : null;
    }}

    async function applySignedResidueOverpaint(viewer) {{
      const structures = viewer.plugin.managers.structure.hierarchy.current.structures ?? [];
      if (!structures.length) return;

      const layerDefs = [];
      if (neutralResidueNumbers.length) {{
        layerDefs.push({{
          residueNumbers: neutralResidueNumbers,
          color: colorHexToMolstarInt(neutralColor),
        }});
      }}
      for (const [colorHex, residueNumbers] of Object.entries(groupedResidueColors)) {{
        if (!Array.isArray(residueNumbers) || !residueNumbers.length) continue;
        layerDefs.push({{
          residueNumbers: residueNumbers,
          color: colorHexToMolstarInt(colorHex),
        }});
      }}
      if (!layerDefs.length) return;

      const update = viewer.plugin.state.data.build();
      for (const structureRef of structures) {{
        for (const representationRef of structureRef.representations) {{
          const structure = representationRef.cell.obj?.data?.sourceData;
          if (!structure) continue;

          const bundleLayers = [];
          for (const layerDef of layerDefs) {{
            const loci = residuesToLoci(structure, layerDef.residueNumbers);
            if (!loci) continue;
            bundleLayers.push({{
              bundle: MolstarLib.structure.StructureElement.Bundle.fromLoci(loci),
              color: layerDef.color,
              clear: false,
            }});
          }}
          if (!bundleLayers.length) continue;

          update.to(representationRef.cell.transform.ref).apply(
            MolstarLib.plugin.StateTransforms.Representation.OverpaintStructureRepresentation3DFromBundle,
            {{
              layers: bundleLayers,
            }},
            {{ tags: 'md-variant-signed-overpaint' }},
          );
        }}
      }}

      await update.commit({{ doNotUpdateCurrent: true }});
    }}

    async function initialiseViewer() {{
      if (!window.molstar?.Viewer?.create || !MolstarLib?.structure?.StructureElement || !MolstarLib?.plugin?.StateTransforms) {{
        setStatus(
          'Mol* loaded incompletely from the CDN. The standard viewer APIs for signed residue coloring are unavailable in this page.',
        );
        return;
      }}

      try {{
        const viewer = await molstar.Viewer.create('app', {{
          layoutIsExpanded: true,
          layoutShowControls: true,
          layoutShowRemoteState: false,
          layoutShowSequence: true,
          layoutShowLog: false,
          layoutShowLeftPanel: true,
          viewportShowExpand: true,
          viewportShowSelectionMode: false,
          viewportShowAnimation: false,
          viewportBackgroundColor: backgroundColor,
        }});
        await viewer.loadStructureFromData(pdbText, 'pdb', {{
          dataLabel: loadLabel,
        }});
        await applySignedResidueOverpaint(viewer);
        setStatus('');
      }} catch (error) {{
        console.error('Mol* viewer initialization failed.', error);
        setStatus(`Mol* viewer initialization failed: ${{error?.message ?? error}}`);
      }}
    }}

    initialiseViewer();
  </script>
</body>
</html>
"""
    html_path.write_text(html_text)


def _write_wildtype_overlay_molstar_html(
    pdb_path: Path,
    html_path: Path,
    title: str,
    subtitle: str,
    structure_entries: list[dict[str, object]],
    viewer_height_px: int = 560,
    footer_note: str | None = None,
) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)

    pdb_text = Path(pdb_path).read_text()
    chain_payloads = []
    legend_items = []
    for structure_entry in structure_entries:
        residue_color_map = {
            int(residue_number): str(color_hex)
            for residue_number, color_hex in (
                structure_entry.get("residue_color_map") or {}
            ).items()
        }
        grouped_residue_colors: dict[str, list[int]] = {}
        for residue_number, color_hex in residue_color_map.items():
            grouped_residue_colors.setdefault(color_hex, []).append(int(residue_number))
        grouped_residue_colors = {
            color_hex: sorted(residue_numbers)
            for color_hex, residue_numbers in grouped_residue_colors.items()
        }
        all_residue_numbers = [
            int(residue_number)
            for residue_number in structure_entry.get("all_residue_numbers", [])
        ]
        neutral_residue_numbers = [
            residue_number
            for residue_number in all_residue_numbers
            if residue_number not in residue_color_map
        ]

        chain_payloads.append(
            {
                "chainId": str(structure_entry["chain_id"]),
                "label": str(structure_entry["label"]),
                "defaultColor": str(structure_entry["default_color"]),
                "neutralColor": str(structure_entry.get("neutral_color", "#d9dde3")),
                "allResidueNumbers": all_residue_numbers,
                "neutralResidueNumbers": neutral_residue_numbers,
                "groupedResidueColors": grouped_residue_colors,
            }
        )
        legend_items.append(
            (
                str(structure_entry["label"]),
                str(structure_entry["default_color"]),
            )
        )

    legend_html = "".join(
        (
            '<div class="structure-chip">'
            f'<span class="swatch" style="background:{html.escape(color_hex)};"></span>'
            f"<span>{html.escape(label)}</span>"
            "</div>"
        )
        for label, color_hex in legend_items
    )
    resolved_footer_note = (
        footer_note
        or "The current colors only distinguish the three aligned WT structures."
    )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.css" />
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #101418;
    }}
    .wrap {{
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      min-height: 100vh;
    }}
    .header, .footer {{
      padding: 14px 18px;
      background: #ffffff;
      box-shadow: 0 1px 0 rgba(16, 20, 24, 0.08);
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 16px;
    }}
    .header p, .footer p {{
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
      color: #46515c;
    }}
    .structure-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      padding: 10px 18px 12px 18px;
      background: #ffffff;
      border-top: 1px solid rgba(16, 20, 24, 0.08);
    }}
    .structure-chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: #27313b;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(16, 20, 24, 0.18);
      flex: 0 0 auto;
    }}
    #app {{
      position: relative;
      width: 100%;
      height: {int(viewer_height_px)}px;
    }}
    #app :is(.msp-plugin, .msp-layout-standard, .msp-layout-expanded) {{
      min-height: {int(viewer_height_px)}px;
    }}
    #status {{
      margin: 12px 18px 0 18px;
      padding: 12px 14px;
      border-radius: 10px;
      background: #eef5ff;
      border: 1px solid #c8dcff;
      color: #1c3f76;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    #status[data-state="hidden"] {{
      display: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(subtitle)}</p>
    </div>
    <div class="structure-legend">{legend_html}</div>
    <div id="status">Loading aligned wildtype structures...</div>
    <div id="app"></div>
    <div class="footer">
      <p>{html.escape(resolved_footer_note)}</p>
    </div>
  </div>
  <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/molstar@5.10.1/build/viewer/molstar.js"></script>
  <script type="text/javascript">
    const pdbText = {json.dumps(pdb_text)};
    const chainPayloads = {json.dumps(chain_payloads, sort_keys=True)};
    const statusEl = document.getElementById('status');
    const backgroundColor = 0xf5f6f8;
    const loadLabel = {json.dumps(title)};
    const MolstarLib = window.molstar?.lib;

    function setStatus(message) {{
      if (!statusEl) return;
      if (!message) {{
        statusEl.textContent = '';
        statusEl.dataset.state = 'hidden';
        return;
      }}
      statusEl.textContent = message;
      delete statusEl.dataset.state;
    }}

    function colorHexToMolstarInt(colorHex) {{
      return Number.parseInt(String(colorHex).replace('#', ''), 16);
    }}

    function buildSelectionLoci(structure, chainId, residueNumbers) {{
      const root = structure.root ?? structure;
      const schemaCandidates = [];
      if (Array.isArray(residueNumbers) && residueNumbers.length) {{
        schemaCandidates.push({{ auth_asym_id: chainId, auth_seq_id: residueNumbers }});
        schemaCandidates.push({{ label_asym_id: chainId, auth_seq_id: residueNumbers }});
        schemaCandidates.push({{ auth_asym_id: chainId, label_seq_id: residueNumbers }});
        schemaCandidates.push({{ label_asym_id: chainId, label_seq_id: residueNumbers }});
      }} else {{
        schemaCandidates.push({{ auth_asym_id: chainId }});
        schemaCandidates.push({{ label_asym_id: chainId }});
      }}
      for (const items of schemaCandidates) {{
        const loci = MolstarLib.structure.StructureElement.Loci.fromSchema(root, {{ items }});
        if (loci?.elements?.length) {{
          return loci;
        }}
      }}
      return null;
    }}

    function buildLayerDefs(payload) {{
      const groupedResidueColors = payload.groupedResidueColors ?? {{}};
      const hasResidueColors = Object.keys(groupedResidueColors).length > 0;
      const layerDefs = [];
      if (!hasResidueColors) {{
        if (payload.chainId) {{
          layerDefs.push({{
            chainId: payload.chainId,
            residueNumbers: null,
            color: colorHexToMolstarInt(payload.defaultColor),
          }});
        }}
        return layerDefs;
      }}

      if (Array.isArray(payload.neutralResidueNumbers) && payload.neutralResidueNumbers.length) {{
        layerDefs.push({{
          chainId: payload.chainId,
          residueNumbers: payload.neutralResidueNumbers,
          color: colorHexToMolstarInt(payload.neutralColor),
        }});
      }}
      for (const [colorHex, residueNumbers] of Object.entries(groupedResidueColors)) {{
        if (!Array.isArray(residueNumbers) || !residueNumbers.length) continue;
        layerDefs.push({{
          chainId: payload.chainId,
          residueNumbers: residueNumbers,
          color: colorHexToMolstarInt(colorHex),
        }});
      }}
      return layerDefs;
    }}

    async function applyStructureOverpaint(viewer) {{
      const structures = viewer.plugin.managers.structure.hierarchy.current.structures ?? [];
      if (!structures.length) return;

      const structureRef = structures[0];
      const update = viewer.plugin.state.data.build();
      for (let index = 0; index < chainPayloads.length; index += 1) {{
        const payload = chainPayloads[index];
        const layerDefs = buildLayerDefs(payload);
        if (!layerDefs.length) continue;

        for (const representationRef of structureRef.representations) {{
          const structure = representationRef.cell.obj?.data?.sourceData;
          if (!structure) continue;

          const bundleLayers = [];
          for (const layerDef of layerDefs) {{
            const loci = buildSelectionLoci(
              structure,
              layerDef.chainId,
              layerDef.residueNumbers,
            );
            if (!loci) continue;
            bundleLayers.push({{
              bundle: MolstarLib.structure.StructureElement.Bundle.fromLoci(loci),
              color: layerDef.color,
              clear: false,
            }});
          }}
          if (!bundleLayers.length) continue;

          update.to(representationRef.cell.transform.ref).apply(
            MolstarLib.plugin.StateTransforms.Representation.OverpaintStructureRepresentation3DFromBundle,
            {{
              layers: bundleLayers,
            }},
            {{ tags: `wt-overlay-${{index}}` }},
          );
        }}
      }}

      await update.commit({{ doNotUpdateCurrent: true }});
    }}

    async function initialiseViewer() {{
      if (!window.molstar?.Viewer?.create || !MolstarLib?.structure?.StructureElement || !MolstarLib?.plugin?.StateTransforms) {{
        setStatus('Mol* loaded incompletely from the CDN. The standard viewer APIs are unavailable in this page.');
        return;
      }}

      try {{
        const viewer = await molstar.Viewer.create('app', {{
          layoutIsExpanded: true,
          layoutShowControls: true,
          layoutShowRemoteState: false,
          layoutShowSequence: true,
          layoutShowLog: false,
          layoutShowLeftPanel: true,
          viewportShowExpand: true,
          viewportShowSelectionMode: false,
          viewportShowAnimation: false,
          viewportBackgroundColor: backgroundColor,
        }});
        await viewer.loadStructureFromData(pdbText, 'pdb', {{
          dataLabel: loadLabel,
        }});
        await applyStructureOverpaint(viewer);
        setStatus('');
      }} catch (error) {{
        console.error('Mol* WT overlay initialization failed.', error);
        setStatus(`Mol* WT overlay initialization failed: ${{error?.message ?? error}}`);
      }}
    }}

    initialiseViewer();
  </script>
</body>
</html>
"""
    html_path.write_text(html_text)


def export_wildtype_overlay_molstar_views(
    structure_family_df: pd.DataFrame,
    output_dir: Path,
    metrics: Iterable[str] = ("rmsf", "residue_contacts", "saltbridge_pairs"),
    viewer_height_px: int = 560,
    reference_family_id: str | None = None,
    structure_color_map: dict[str, str] | None = None,
) -> WildtypeMolstarOverlayExportBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wildtype_structure_df = (
        structure_family_df.loc[
            structure_family_df["is_wildtype"],
            ["family_id", "sample_name", "pdb_path"],
        ]
        .sort_values("family_id")
        .reset_index(drop=True)
    )
    if wildtype_structure_df.empty:
        return WildtypeMolstarOverlayExportBundle(
            viewer_index_df=pd.DataFrame(),
            alignment_summary_df=pd.DataFrame(),
        )

    resolved_reference_family_id = (
        str(reference_family_id)
        if reference_family_id is not None
        else str(wildtype_structure_df.iloc[0]["family_id"])
    )
    reference_match_df = wildtype_structure_df.loc[
        wildtype_structure_df["family_id"].eq(resolved_reference_family_id)
    ]
    if reference_match_df.empty:
        raise KeyError(
            "Requested WT overlay reference family was not found among the wildtype structures: "
            f"{resolved_reference_family_id}"
        )
    reference_row = reference_match_df.iloc[0]
    reference_trace = parse_ca_trace(Path(reference_row["pdb_path"]))

    metric_alias_map = {
        "rmsf": "rmsf",
        "contact": "residue_contacts",
        "contacts": "residue_contacts",
        "residue_contacts": "residue_contacts",
        "saltbridge": "saltbridge_pairs",
        "saltbridges": "saltbridge_pairs",
        "salt_bridge": "saltbridge_pairs",
        "salt_bridges": "saltbridge_pairs",
        "saltbridge_pairs": "saltbridge_pairs",
    }
    metric_label_map = {
        "rmsf": "RMSF",
        "residue_contacts": "Residue contacts",
        "saltbridge_pairs": "Salt bridges",
    }
    resolved_metrics = []
    for metric_name in metrics:
        metric_key = str(metric_name).strip().lower()
        resolved_metric_name = metric_alias_map.get(metric_key, metric_key)
        if resolved_metric_name not in metric_label_map:
            raise KeyError(
                "Unsupported WT Mol* overlay metric. Expected one of "
                "'rmsf', 'residue_contacts', or 'saltbridge_pairs'."
            )
        resolved_metrics.append(resolved_metric_name)

    default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#17becf"]
    structure_color_map = dict(structure_color_map or {})
    available_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    aligned_structure_rows = []
    structure_entries = []
    aligned_structure_dir = output_dir / "aligned_wildtypes"
    aligned_structure_dir.mkdir(parents=True, exist_ok=True)
    trace_cache = {
        str(row.family_id): parse_ca_trace(Path(row.pdb_path))
        for row in wildtype_structure_df.itertuples(index=False)
    }

    for structure_index, row in enumerate(wildtype_structure_df.itertuples(index=False)):
        family_id = str(row.family_id)
        structure_label = f"{family_id} WT"
        chain_id = available_chain_ids[structure_index % len(available_chain_ids)]
        aligned_pdb_path = aligned_structure_dir / f"{family_id}_aligned_to_{resolved_reference_family_id}.pdb"
        residue_numbers = (
            trace_cache[family_id]["residue_number"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        default_color = structure_color_map.get(
            family_id,
            default_palette[structure_index % len(default_palette)],
        )

        if family_id == resolved_reference_family_id:
            _write_transformed_pdb(
                template_pdb_path=Path(row.pdb_path),
                output_path=aligned_pdb_path,
            )
            alignment_row = {
                "family_id": family_id,
                "sample_name": str(row.sample_name),
                "reference_family_id": resolved_reference_family_id,
                "alignment_method": "reference_identity",
                "aligned_length": int(len(reference_trace)),
                "rmsd_angstrom": 0.0,
                "seq_identity_aligned": 1.0,
                "tm_score_to_reference": 1.0,
                "aligned_fraction_to_reference": 1.0,
            }
        else:
            alignment_result = compute_ca_superposition_alignment(
                structure_a_trace=reference_trace,
                structure_b_trace=trace_cache[family_id],
            )
            _write_transformed_pdb(
                template_pdb_path=Path(row.pdb_path),
                output_path=aligned_pdb_path,
                rotation_matrix=np.asarray(alignment_result["rotation_matrix"], dtype=float),
                mobile_center=np.asarray(alignment_result["mobile_center"], dtype=float),
                reference_center=np.asarray(alignment_result["reference_center"], dtype=float),
            )
            alignment_row = {
                "family_id": family_id,
                "sample_name": str(row.sample_name),
                "reference_family_id": resolved_reference_family_id,
                "alignment_method": "ca_superposition",
                "aligned_length": int(alignment_result["aligned_length"]),
                "rmsd_angstrom": float(alignment_result["rmsd_angstrom"]),
                "seq_identity_aligned": float(alignment_result["seq_identity_aligned"]),
                "tm_score_to_reference": float(alignment_result["tm_score_to_structure_a"]),
                "aligned_fraction_to_reference": float(
                    alignment_result["aligned_fraction_to_structure_a"]
                ),
            }

        alignment_row["default_color"] = default_color
        alignment_row["chain_id"] = chain_id
        alignment_row["aligned_pdb_path"] = aligned_pdb_path
        aligned_structure_rows.append(alignment_row)
        structure_entries.append(
            {
                "family_id": family_id,
                "chain_id": chain_id,
                "label": structure_label,
                "pdb_path": aligned_pdb_path,
                "default_color": default_color,
                "all_residue_numbers": residue_numbers,
                "residue_color_map": {},
            }
        )

    alignment_summary_df = pd.DataFrame(aligned_structure_rows)
    if not alignment_summary_df.empty:
        alignment_summary_df.to_csv(output_dir / "wildtype_alignment_summary.csv", index=False)

    combined_structure_dir = output_dir / "combined_wildtypes"
    combined_structure_dir.mkdir(parents=True, exist_ok=True)
    combined_pdb_path = (
        combined_structure_dir / f"wildtype_overlay_aligned_to_{resolved_reference_family_id}.pdb"
    )
    _write_combined_overlay_pdb(
        structure_entries=structure_entries,
        output_path=combined_pdb_path,
    )

    viewer_rows = []
    ordered_family_ids = [str(entry["family_id"]) for entry in structure_entries]
    for metric_name in resolved_metrics:
        metric_label = metric_label_map[metric_name]
        metric_output_dir = output_dir / metric_name
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        html_output_path = metric_output_dir / f"wildtype_overlay_{metric_name}.html"
        footer_note = (
            f"All structures are aligned to {resolved_reference_family_id} on matched CA atoms. "
            f"These colors only distinguish the WT backbones; per-residue {metric_label.lower()} coloring can be layered on next."
        )
        _write_wildtype_overlay_molstar_html(
            pdb_path=combined_pdb_path,
            html_path=html_output_path,
            title=f"Wildtype Overlay: {metric_label}",
            subtitle=(
                f"{len(structure_entries)} wildtype structures aligned to {resolved_reference_family_id} "
                f"and superimposed for the {metric_label} view."
            ),
            structure_entries=structure_entries,
            viewer_height_px=viewer_height_px,
            footer_note=footer_note,
        )
        viewer_rows.append(
            {
                "metric_name": metric_name,
                "metric_label": metric_label,
                "reference_family_id": resolved_reference_family_id,
                "wildtype_count": int(len(structure_entries)),
                "family_order": " | ".join(ordered_family_ids),
                "pdb_path": combined_pdb_path,
                "html_path": html_output_path,
            }
        )

    viewer_index_df = pd.DataFrame(viewer_rows)
    if not viewer_index_df.empty:
        viewer_index_df.to_csv(output_dir / "wildtype_overlay_viewer_index.csv", index=False)

    return WildtypeMolstarOverlayExportBundle(
        viewer_index_df=viewer_index_df,
        alignment_summary_df=alignment_summary_df,
    )


def export_wildtype_superposition_molstar_view(
    structure_family_df: pd.DataFrame,
    output_dir: Path,
    viewer_height_px: int = 560,
    reference_family_id: str | None = None,
) -> WildtypeMolstarSuperpositionExport:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wildtype_structure_df = (
        structure_family_df.loc[
            structure_family_df["is_wildtype"],
            ["family_id", "sample_name", "pdb_path"],
        ]
        .sort_values("family_id")
        .reset_index(drop=True)
    )
    if wildtype_structure_df.empty:
        raise ValueError("No wildtype structures were found for WT Mol* superposition export.")

    resolved_reference_family_id = (
        str(reference_family_id)
        if reference_family_id is not None
        else str(wildtype_structure_df.iloc[0]["family_id"])
    )
    reference_match_df = wildtype_structure_df.loc[
        wildtype_structure_df["family_id"].eq(resolved_reference_family_id)
    ]
    if reference_match_df.empty:
        raise KeyError(
            "Requested WT superposition reference family was not found among the wildtype structures: "
            f"{resolved_reference_family_id}"
        )

    reference_row = reference_match_df.iloc[0]
    reference_trace = parse_ca_trace(Path(reference_row["pdb_path"]))
    available_chain_ids = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

    aligned_structure_rows = []
    structure_entries = []
    aligned_structure_dir = output_dir / "aligned_wildtypes"
    aligned_structure_dir.mkdir(parents=True, exist_ok=True)
    trace_cache = {
        str(row.family_id): parse_ca_trace(Path(row.pdb_path))
        for row in wildtype_structure_df.itertuples(index=False)
    }

    for structure_index, row in enumerate(wildtype_structure_df.itertuples(index=False)):
        family_id = str(row.family_id)
        chain_id = available_chain_ids[structure_index % len(available_chain_ids)]
        aligned_pdb_path = aligned_structure_dir / f"{family_id}_aligned_to_{resolved_reference_family_id}.pdb"
        residue_numbers = (
            trace_cache[family_id]["residue_number"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )

        if family_id == resolved_reference_family_id:
            _write_transformed_pdb(
                template_pdb_path=Path(row.pdb_path),
                output_path=aligned_pdb_path,
            )
            alignment_row = {
                "family_id": family_id,
                "sample_name": str(row.sample_name),
                "reference_family_id": resolved_reference_family_id,
                "alignment_method": "reference_identity",
                "aligned_length": int(len(reference_trace)),
                "rmsd_angstrom": 0.0,
                "seq_identity_aligned": 1.0,
                "tm_score_to_reference": 1.0,
                "aligned_fraction_to_reference": 1.0,
            }
        else:
            alignment_result = compute_ca_superposition_alignment(
                structure_a_trace=reference_trace,
                structure_b_trace=trace_cache[family_id],
            )
            _write_transformed_pdb(
                template_pdb_path=Path(row.pdb_path),
                output_path=aligned_pdb_path,
                rotation_matrix=np.asarray(alignment_result["rotation_matrix"], dtype=float),
                mobile_center=np.asarray(alignment_result["mobile_center"], dtype=float),
                reference_center=np.asarray(alignment_result["reference_center"], dtype=float),
            )
            alignment_row = {
                "family_id": family_id,
                "sample_name": str(row.sample_name),
                "reference_family_id": resolved_reference_family_id,
                "alignment_method": "ca_superposition",
                "aligned_length": int(alignment_result["aligned_length"]),
                "rmsd_angstrom": float(alignment_result["rmsd_angstrom"]),
                "seq_identity_aligned": float(alignment_result["seq_identity_aligned"]),
                "tm_score_to_reference": float(alignment_result["tm_score_to_structure_a"]),
                "aligned_fraction_to_reference": float(
                    alignment_result["aligned_fraction_to_structure_a"]
                ),
            }

        alignment_row["chain_id"] = chain_id
        alignment_row["aligned_pdb_path"] = aligned_pdb_path
        aligned_structure_rows.append(alignment_row)
        structure_entries.append(
            {
                "family_id": family_id,
                "chain_id": chain_id,
                "label": family_id,
                "pdb_path": aligned_pdb_path,
                "all_residue_numbers": residue_numbers,
            }
        )

    alignment_summary_df = pd.DataFrame(aligned_structure_rows)
    alignment_summary_df.to_csv(output_dir / "wildtype_superposition_alignment_summary.csv", index=False)

    combined_pdb_path = output_dir / f"wildtype_superposition_aligned_to_{resolved_reference_family_id}.pdb"
    _write_combined_overlay_pdb(
        structure_entries=structure_entries,
        output_path=combined_pdb_path,
    )

    html_path = output_dir / "wildtype_superposition.html"
    _write_simple_molstar_html(
        pdb_path=combined_pdb_path,
        html_path=html_path,
        title="Wildtype Structural Superposition",
        subtitle=(
            f"Three wildtype structures aligned to {resolved_reference_family_id} and loaded as one combined PDB."
        ),
        viewer_height_px=viewer_height_px,
        footer_note=(
            "This is a minimal restart of the WT Mol* workflow: one aligned combined PDB, one standard Mol* load."
        ),
    )

    return WildtypeMolstarSuperpositionExport(
        html_path=html_path,
        pdb_path=combined_pdb_path,
        alignment_summary_df=alignment_summary_df,
    )


def export_md_variant_molstar_views(
    structure_family_df: pd.DataFrame,
    md_variant_bundle: MDVariantAnalysisBundle,
    metric_dir: Path,
    output_dir: Path,
    metrics: Iterable[str] = ("rmsf", "rmsd", "hbond_pairs", "saltbridge_pairs"),
    z_threshold: float = 1.5,
    viewer_height_px: int = 560,
) -> MDVariantMolstarExportBundle:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wildtype_structure_df = structure_family_df.loc[
        structure_family_df["is_wildtype"]
    ].copy()
    wildtype_structure_lookup = {
        str(row.family_id): Path(row.pdb_path)
        for row in wildtype_structure_df.itertuples(index=False)
    }

    residue_score_tables = []
    metric_alias_map = {
        "rmsf": "rmsf",
        "rmsd": "rmsd",
        "hhbond": "hbond_pairs",
        "hhbonds": "hbond_pairs",
        "hbond": "hbond_pairs",
        "hbonds": "hbond_pairs",
        "hbond_pairs": "hbond_pairs",
        "saltbridge": "saltbridge_pairs",
        "saltbridges": "saltbridge_pairs",
        "saltbridge_pairs": "saltbridge_pairs",
    }
    metric_names = [
        metric_alias_map.get(str(metric_name).strip().lower(), str(metric_name).strip().lower())
        for metric_name in metrics
    ]

    for metric_name in metric_names:
        if metric_name == "rmsd":
            continue
        for family_id in md_variant_bundle.family_ids:
            residue_score_table = build_md_variant_molstar_residue_score_table(
                md_variant_bundle=md_variant_bundle,
                family_id=family_id,
                metric_name=metric_name,
                z_threshold=z_threshold,
            )
            if not residue_score_table.empty:
                residue_score_tables.append(residue_score_table)

    score_table_df = (
        pd.concat(residue_score_tables, ignore_index=True)
        if residue_score_tables
        else pd.DataFrame()
    )
    global_rmsd_summary_df = build_md_variant_global_rmsd_score_table(
        md_variant_bundle=md_variant_bundle,
        metric_dir=metric_dir,
    )

    if not score_table_df.empty:
        score_table_df.to_csv(output_dir / "molstar_residue_metric_scores.csv", index=False)
    if not global_rmsd_summary_df.empty:
        global_rmsd_summary_df.to_csv(
            output_dir / "molstar_global_rmsd_scores.csv",
            index=False,
        )

    metric_label_map = {
        "rmsf": "RMSF",
        "rmsd": "RMSD",
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }

    viewer_rows = []
    for family_id in md_variant_bundle.family_ids:
        wildtype_pdb_path = wildtype_structure_lookup.get(str(family_id))
        if wildtype_pdb_path is None:
            continue

        all_residue_numbers = (
            parse_ca_trace(wildtype_pdb_path)["residue_number"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        family_output_dir = output_dir / str(family_id)
        family_output_dir.mkdir(parents=True, exist_ok=True)

        family_metric_tables = {}
        if not score_table_df.empty:
            for metric_name in {"rmsf", "hbond_pairs", "saltbridge_pairs"}:
                family_metric_tables[metric_name] = score_table_df.loc[
                    score_table_df["family"].eq(family_id)
                    & score_table_df["metric_name"].eq(metric_name)
                ].copy()

        family_global_rmsd_df = global_rmsd_summary_df.loc[
            global_rmsd_summary_df["family"].eq(family_id)
        ].copy()

        variant_systems = compute_md_variant_order(md_variant_bundle.metadata_df, family_id)
        for variant_system in variant_systems:
            variant_metadata_df = md_variant_bundle.metadata_df.loc[
                md_variant_bundle.metadata_df["system"].eq(variant_system)
            ]
            if variant_metadata_df.empty:
                continue
            variant_label = get_md_variant_display_label(
                md_variant_bundle.metadata_df,
                variant_system,
            )

            for metric_name in metric_names:
                if metric_name == "rmsd":
                    metric_variant_df = family_global_rmsd_df.loc[
                        family_global_rmsd_df["system"].eq(variant_system)
                    ].copy()
                else:
                    metric_table_df = family_metric_tables.get(metric_name, pd.DataFrame())
                    if metric_table_df.empty or "variant_system" not in metric_table_df.columns:
                        continue
                    metric_variant_df = metric_table_df.loc[
                        metric_table_df["variant_system"].eq(variant_system)
                    ].copy()
                if metric_variant_df.empty:
                    continue

                metric_output_dir = family_output_dir / metric_name / str(variant_system)
                metric_output_dir.mkdir(parents=True, exist_ok=True)

                if metric_name == "rmsd":
                    score_row = metric_variant_df.iloc[0]
                    score_z = float(score_row["score_z"])
                    abs_score_z = float(score_row["abs_score_z"])
                    passes_threshold = abs_score_z >= float(z_threshold)
                    signed_score_value = float(np.clip(score_z, -99.99, 99.99)) if passes_threshold else 0.0
                    signed_residue_score_map = {
                        int(residue_number): signed_score_value
                        for residue_number in all_residue_numbers
                    }
                    positive_residue_count = (
                        len(all_residue_numbers)
                        if passes_threshold and score_z > 0
                        else 0
                    )
                    negative_residue_count = (
                        len(all_residue_numbers)
                        if passes_threshold and score_z < 0
                        else 0
                    )
                    metric_variant_df = pd.DataFrame(
                        {
                            "family": [family_id],
                            "variant_system": [variant_system],
                            "variant_label": [variant_label],
                            "metric_name": ["rmsd"],
                            "metric_label": ["RMSD"],
                            "residues_colored_positive": [positive_residue_count],
                            "residues_colored_negative": [negative_residue_count],
                            "delta_global_rmsd": [float(score_row["delta_global_rmsd"])],
                            "score_z": [score_z],
                            "abs_score_z": [abs_score_z],
                            "z_threshold": [float(z_threshold)],
                        }
                    )
                else:
                    signed_residue_score_map = {
                        int(row.resid): (
                            float(np.clip(float(row.score_z), -99.99, 99.99))
                            if bool(row.passes_threshold)
                            else 0.0
                        )
                        for row in metric_variant_df.itertuples(index=False)
                    }
                    positive_residue_count = int(
                        metric_variant_df["direction"].eq("positive").sum()
                    )
                    negative_residue_count = int(
                        metric_variant_df["direction"].eq("negative").sum()
                    )
                    metric_variant_df.to_csv(
                        metric_output_dir / f"{variant_system}_{metric_name}_residue_scores.csv",
                        index=False,
                    )

                score_limit = float(
                    max(
                        metric_variant_df["abs_score_z"].max()
                        if "abs_score_z" in metric_variant_df.columns
                        else 0.0,
                        float(z_threshold),
                    )
                )
                residue_color_map, score_limit = _build_md_variant_residue_color_map(
                    residue_score_map=signed_residue_score_map,
                    z_threshold=z_threshold,
                    score_limit=score_limit,
                )

                pdb_output_path = (
                    metric_output_dir
                    / f"{variant_system}_{metric_name}_signed_z{float(z_threshold):0.1f}.pdb"
                )
                html_output_path = (
                    metric_output_dir
                    / f"{variant_system}_{metric_name}_signed_z{float(z_threshold):0.1f}.html"
                )
                _write_md_variant_bfactor_pdb(
                    template_pdb_path=wildtype_pdb_path,
                    residue_bfactor_map=signed_residue_score_map,
                    output_path=pdb_output_path,
                )
                subtitle = (
                    f"{family_id} | {variant_label} | {metric_label_map[metric_name]} | "
                    f"Blue = lower than WT, red = higher than WT, gray = |z| < {float(z_threshold):0.1f}."
                )
                _write_md_variant_molstar_html(
                    pdb_path=pdb_output_path,
                    html_path=html_output_path,
                    title=f"{variant_label}: {metric_label_map[metric_name]} Signed WT Delta",
                    subtitle=subtitle,
                    residue_color_map=residue_color_map,
                    all_residue_numbers=all_residue_numbers,
                    score_limit=score_limit,
                    viewer_height_px=viewer_height_px,
                )

                viewer_rows.append(
                    {
                        "family": family_id,
                        "variant_system": variant_system,
                        "variant_label": variant_label,
                        "metric_name": metric_name,
                        "metric_label": metric_label_map[metric_name],
                        "direction": "signed",
                        "direction_label": "Signed WT delta",
                        "z_threshold": float(z_threshold),
                        "positive_residue_count": int(positive_residue_count),
                        "negative_residue_count": int(negative_residue_count),
                        "colored_residue_count": int(len(residue_color_map)),
                        "pdb_path": pdb_output_path,
                        "html_path": html_output_path,
                    }
                )

    viewer_index_df = pd.DataFrame(viewer_rows)
    if not viewer_index_df.empty:
        viewer_index_df.to_csv(output_dir / "molstar_viewer_index.csv", index=False)

    return MDVariantMolstarExportBundle(
        viewer_index_df=viewer_index_df,
        score_table_df=score_table_df,
        global_rmsd_summary_df=global_rmsd_summary_df,
    )


def _compress_residue_numbers_to_ranges(residue_numbers: Iterable[int]) -> str:
    ordered_residue_numbers = sorted(
        {
            int(residue_number)
            for residue_number in residue_numbers
            if pd.notna(residue_number)
        }
    )
    if not ordered_residue_numbers:
        return ""

    range_tokens = []
    start_residue = ordered_residue_numbers[0]
    previous_residue = ordered_residue_numbers[0]

    for residue_number in ordered_residue_numbers[1:]:
        if residue_number == previous_residue + 1:
            previous_residue = residue_number
            continue

        if start_residue == previous_residue:
            range_tokens.append(str(start_residue))
        else:
            range_tokens.append(f"{start_residue}-{previous_residue}")
        start_residue = residue_number
        previous_residue = residue_number

    if start_residue == previous_residue:
        range_tokens.append(str(start_residue))
    else:
        range_tokens.append(f"{start_residue}-{previous_residue}")
    return ",".join(range_tokens)


def _resolve_wildtype_chimerax_metric_name(metric_name: str) -> str:
    metric_alias_map = {
        "rmsf": "rmsf",
        "interaction_change": "interaction_change",
        "combined_interaction_change": "interaction_change",
        "interaction_changes": "interaction_change",
        "combined_interactions": "interaction_change",
        "hhbond": "hbond_pairs",
        "hhbonds": "hbond_pairs",
        "hbond": "hbond_pairs",
        "hbonds": "hbond_pairs",
        "hbond_pairs": "hbond_pairs",
        "saltbridge": "saltbridge_pairs",
        "saltbridges": "saltbridge_pairs",
        "saltbridge_pairs": "saltbridge_pairs",
    }
    resolved_metric_name = metric_alias_map.get(
        str(metric_name).strip().lower(),
        str(metric_name).strip().lower(),
    )
    if resolved_metric_name not in {"rmsf", "interaction_change", "hbond_pairs", "saltbridge_pairs"}:
        raise KeyError(
            "Unsupported WT ChimeraX metric. Expected one of "
            "'rmsf', 'interaction_change', 'hbond_pairs', or 'saltbridge_pairs'."
        )
    return resolved_metric_name


def _normalize_chimerax_model_id(model_id: str) -> str:
    model_id_text = str(model_id).strip()
    if not model_id_text:
        raise ValueError("ChimeraX model ids must not be empty.")
    return model_id_text if model_id_text.startswith("#") else f"#{model_id_text}"


def _normalize_signed_z_breaks(signed_z_breaks: Iterable[float]) -> tuple[float, ...]:
    normalized_breaks = tuple(
        sorted(
            {
                float(break_value)
                for break_value in signed_z_breaks
                if pd.notna(break_value) and float(break_value) > 0.0
            }
        )
    )
    if not normalized_breaks:
        raise ValueError("At least one positive signed z-score break is required.")
    return normalized_breaks


DEFAULT_CHIMERAX_INTERACTION_COLORS = {
    "saltbridge_gain": ("#D55E00",),
    "saltbridge_loss": ("#007A5E",),
    "hbond_gain": ("#0072B2",),
    "hbond_loss": ("#B88600",),
}


def _normalize_chimerax_interaction_colors(
    interaction_colors: dict[str, Iterable[str]] | None,
    n_intervals: int,
) -> dict[str, tuple[str, ...]]:
    required_channels = tuple(DEFAULT_CHIMERAX_INTERACTION_COLORS)
    supplied_colors = (
        DEFAULT_CHIMERAX_INTERACTION_COLORS
        if interaction_colors is None
        else interaction_colors
    )
    missing_channels = set(required_channels).difference(supplied_colors)
    extra_channels = set(supplied_colors).difference(required_channels)
    if missing_channels or extra_channels:
        raise KeyError(
            "ChimeraX interaction colors must define exactly "
            f"{list(required_channels)}; missing={sorted(missing_channels)}, "
            f"unexpected={sorted(extra_channels)}."
        )

    normalized_colors: dict[str, tuple[str, ...]] = {}
    for channel_name in required_channels:
        channel_colors = tuple(to_hex(color_value) for color_value in supplied_colors[channel_name])
        if len(channel_colors) != int(n_intervals):
            raise ValueError(
                f"ChimeraX interaction channel {channel_name!r} has "
                f"{len(channel_colors)} colors, but {int(n_intervals)} are required "
                "to match the z-score breaks."
            )
        normalized_colors[channel_name] = channel_colors
    return normalized_colors


def _resolve_wildtype_chimerax_target_spec_df(
    wildtype_alignment_df: pd.DataFrame,
    target_layout: str = "combined_model",
    combined_model_id: str = "#1",
    separate_model_id_by_family: dict[str, str] | None = None,
    separate_model_chain_id: str = "A",
) -> pd.DataFrame:
    resolved_target_layout = str(target_layout).strip().lower()
    if resolved_target_layout not in {"combined_model", "separate_models"}:
        raise KeyError(
            "Unsupported WT ChimeraX target layout. Expected one of "
            "'combined_model' or 'separate_models'."
        )

    required_columns = {"family_id", "chain_id", "aligned_pdb_path"}
    missing_columns = required_columns.difference(wildtype_alignment_df.columns)
    if missing_columns:
        raise KeyError(
            "WT alignment summary is missing required columns for ChimeraX targeting: "
            f"{sorted(missing_columns)}"
        )

    target_spec_df = (
        wildtype_alignment_df.loc[:, ["family_id", "chain_id", "aligned_pdb_path"]]
        .drop_duplicates(subset=["family_id"])
        .sort_values(["family_id"])
        .reset_index(drop=True)
    )
    if resolved_target_layout == "combined_model":
        target_spec_df["target_model_id"] = _normalize_chimerax_model_id(combined_model_id)
        target_spec_df["target_chain_id"] = target_spec_df["chain_id"].astype(str)
        return target_spec_df

    if separate_model_id_by_family is None:
        separate_model_id_by_family = {
            str(row.family_id): f"#{row_index + 1}"
            for row_index, row in enumerate(target_spec_df.itertuples(index=False))
        }

    normalized_model_id_by_family = {
        str(family_id): _normalize_chimerax_model_id(model_id)
        for family_id, model_id in separate_model_id_by_family.items()
    }
    missing_family_ids = sorted(
        set(target_spec_df["family_id"].astype(str)).difference(normalized_model_id_by_family)
    )
    if missing_family_ids:
        raise KeyError(
            "Separate-model ChimeraX targeting is missing model ids for WT families: "
            f"{missing_family_ids}"
        )

    target_chain_id = str(separate_model_chain_id).strip() or "A"
    target_spec_df["target_model_id"] = target_spec_df["family_id"].map(normalized_model_id_by_family)
    target_spec_df["target_chain_id"] = target_chain_id
    return target_spec_df


def _build_wildtype_chimerax_family_average_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    wildtype_alignment_df: pd.DataFrame | None,
    metric_name: str,
) -> pd.DataFrame:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)

    if wildtype_alignment_df is not None:
        alignment_columns = {"family_id", "chain_id"}
        missing_alignment_columns = alignment_columns.difference(wildtype_alignment_df.columns)
        if missing_alignment_columns:
            raise KeyError(
                "WT alignment summary is missing required columns for ChimeraX export: "
                f"{sorted(missing_alignment_columns)}"
            )

    metric_label_map = {
        "rmsf": "RMSF",
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }
    family_score_tables = []
    for family_id in md_variant_bundle.family_ids:
        family_score_df = build_md_variant_molstar_residue_score_table(
            md_variant_bundle=md_variant_bundle,
            family_id=family_id,
            metric_name=resolved_metric_name,
            z_threshold=1.5,
        )
        if family_score_df.empty:
            continue

        family_score_df = family_score_df.copy()
        family_score_df["score_raw"] = pd.to_numeric(
            family_score_df["score_raw"],
            errors="coerce",
        ).fillna(0.0)
        if "gain_raw" not in family_score_df.columns:
            family_score_df["gain_raw"] = family_score_df["score_raw"].clip(lower=0.0)
        else:
            family_score_df["gain_raw"] = pd.to_numeric(
                family_score_df["gain_raw"],
                errors="coerce",
            ).fillna(0.0)
        if "loss_raw" not in family_score_df.columns:
            family_score_df["loss_raw"] = family_score_df["score_raw"].clip(upper=0.0)
        else:
            family_score_df["loss_raw"] = pd.to_numeric(
                family_score_df["loss_raw"],
                errors="coerce",
            ).fillna(0.0)

        family_average_df = (
            family_score_df.groupby(["family", "resid", "resname"], as_index=False)
            .agg(
                n_variants=("variant_system", "nunique"),
                mean_score_raw=("score_raw", "mean"),
                mean_abs_score_raw=("score_raw", lambda values: np.mean(np.abs(values))),
                mean_gain_raw=("gain_raw", "mean"),
                mean_loss_raw=("loss_raw", "mean"),
                fraction_variants_positive=(
                    "score_raw",
                    lambda values: np.mean(np.asarray(values, dtype=float) > 0.0),
                ),
                fraction_variants_negative=(
                    "score_raw",
                    lambda values: np.mean(np.asarray(values, dtype=float) < 0.0),
                ),
            )
            .rename(columns={"family": "family_id"})
        )
        family_average_df["metric_name"] = resolved_metric_name
        family_average_df["metric_label"] = metric_label_map[resolved_metric_name]
        family_average_df["mean_score_z"] = (
            family_average_df.groupby("family_id", group_keys=False)["mean_score_raw"]
            .apply(_compute_md_variant_group_zscores)
            .astype(float)
        )
        family_score_tables.append(family_average_df)

    if not family_score_tables:
        return pd.DataFrame(
            columns=[
                "family_id",
                "chain_id",
                "metric_name",
                "metric_label",
                "resid",
                "resname",
                "n_variants",
                "mean_score_raw",
                "mean_score_z",
                "mean_abs_score_raw",
                "mean_gain_raw",
                "mean_loss_raw",
                "fraction_variants_positive",
                "fraction_variants_negative",
            ]
        )

    aggregate_df = pd.concat(family_score_tables, ignore_index=True)
    if wildtype_alignment_df is not None:
        alignment_lookup_df = wildtype_alignment_df.loc[
            :,
            ["family_id", "chain_id"],
        ].drop_duplicates(subset=["family_id"])
        aggregate_df = aggregate_df.merge(
            alignment_lookup_df,
            on="family_id",
            how="left",
        )
    return aggregate_df.sort_values(["family_id", "resid"]).reset_index(drop=True)


def _build_md_variant_residue_replicate_matrix(
    residue_replicate_df: pd.DataFrame,
    system_name: str,
    residue_numbers: Iterable[int],
    replicate_indices: Iterable[int],
) -> pd.DataFrame:
    residue_index = pd.Index(sorted({int(resid) for resid in residue_numbers}), name="resid")
    replicate_columns = [int(replicate_index) for replicate_index in replicate_indices]
    if not len(residue_index) or not len(replicate_columns):
        return pd.DataFrame(index=residue_index, columns=replicate_columns, dtype=float).fillna(0.0)

    system_df = residue_replicate_df.loc[
        residue_replicate_df["system"].eq(system_name)
    ].copy()
    if system_df.empty:
        return pd.DataFrame(
            0.0,
            index=residue_index,
            columns=replicate_columns,
        )

    system_df["replicate_index"] = pd.to_numeric(
        system_df["replicate_index"],
        errors="coerce",
    ).astype("Int64")
    system_df = system_df.dropna(subset=["replicate_index"]).copy()
    system_df["replicate_index"] = system_df["replicate_index"].astype(int)

    matrix_df = system_df.pivot_table(
        index="resid",
        columns="replicate_index",
        values="occupancy_sum",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix_df.index = pd.Index(matrix_df.index.astype(int), name="resid")
    matrix_df = matrix_df.reindex(index=residue_index, columns=replicate_columns, fill_value=0.0)
    return matrix_df.astype(float)


def _build_wildtype_chimerax_pair_majority_significance_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    metric_name: str,
    alpha: float = 0.05,
    majority_fraction_threshold: float = 0.5,
) -> pd.DataFrame:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)
    if resolved_metric_name not in {"hbond_pairs", "saltbridge_pairs"}:
        raise KeyError(
            "WT ChimeraX majority-significance coloring is only supported for "
            "'hbond_pairs' and 'saltbridge_pairs'."
        )

    residue_replicate_df = md_variant_bundle.pair_residue_replicate_by_metric.get(
        resolved_metric_name,
        pd.DataFrame(),
    ).copy()
    if residue_replicate_df.empty:
        return pd.DataFrame(
            columns=[
                "family_id",
                "metric_name",
                "metric_label",
                "resid",
                "resname",
                "n_variants",
                "n_variants_tested",
                "n_variants_significant_positive",
                "n_variants_significant_negative",
                "fraction_variants_significant_positive",
                "fraction_variants_significant_negative",
                "majority_fraction",
                "majority_direction",
                "mean_score_raw",
                "mean_gain_raw",
                "mean_loss_raw",
                "majority_fraction_threshold",
                "alpha_fdr",
            ]
        )

    metric_label_map = {
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }
    metadata_df = md_variant_bundle.metadata_df
    family_rows = []

    for family_id in md_variant_bundle.family_ids:
        family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
        if family_metadata_df.empty or not family_metadata_df["is_wildtype"].any():
            continue

        wildtype_system = family_metadata_df.loc[
            family_metadata_df["is_wildtype"],
            "system",
        ].iat[0]
        variant_systems = compute_md_variant_order(metadata_df, family_id)
        if not variant_systems:
            continue

        family_systems = [wildtype_system, *variant_systems]
        family_residue_df = residue_replicate_df.loc[
            residue_replicate_df["system"].isin(family_systems)
        ].copy()
        if family_residue_df.empty:
            continue

        residue_name_lookup = (
            family_residue_df.loc[:, ["resid", "resname"]]
            .dropna(subset=["resid"])
            .assign(resid=lambda df: pd.to_numeric(df["resid"], errors="coerce").astype("Int64"))
            .dropna(subset=["resid"])
            .assign(resid=lambda df: df["resid"].astype(int))
            .drop_duplicates(subset=["resid"])
            .sort_values(["resid"])
        )
        residue_numbers = residue_name_lookup["resid"].astype(int).tolist()
        if not residue_numbers:
            continue
        residue_name_map = dict(
            zip(
                residue_name_lookup["resid"].astype(int),
                residue_name_lookup["resname"].astype(str),
            )
        )

        wildtype_replicate_count = int(md_variant_bundle.replicate_count_by_system.get(wildtype_system, 0))
        if wildtype_replicate_count < 2:
            continue
        wildtype_replicate_indices = list(range(1, wildtype_replicate_count + 1))
        wildtype_matrix_df = _build_md_variant_residue_replicate_matrix(
            residue_replicate_df=family_residue_df,
            system_name=wildtype_system,
            residue_numbers=residue_numbers,
            replicate_indices=wildtype_replicate_indices,
        )

        variant_test_tables = []
        for variant_system in variant_systems:
            variant_replicate_count = int(
                md_variant_bundle.replicate_count_by_system.get(variant_system, 0)
            )
            if variant_replicate_count < 2:
                continue

            variant_replicate_indices = list(range(1, variant_replicate_count + 1))
            variant_matrix_df = _build_md_variant_residue_replicate_matrix(
                residue_replicate_df=family_residue_df,
                system_name=variant_system,
                residue_numbers=residue_numbers,
                replicate_indices=variant_replicate_indices,
            )
            variant_label = get_md_variant_display_label(
                metadata_df,
                variant_system,
            )

            test_rows = []
            for resid in residue_numbers:
                variant_values = variant_matrix_df.loc[resid].to_numpy(dtype=float)
                wildtype_values = wildtype_matrix_df.loc[resid].to_numpy(dtype=float)
                statistic, p_value = _compute_welch_ttest(
                    variant_values,
                    wildtype_values,
                )
                variant_mean = (
                    float(np.mean(variant_values))
                    if variant_values.size
                    else np.nan
                )
                wildtype_mean = (
                    float(np.mean(wildtype_values))
                    if wildtype_values.size
                    else np.nan
                )
                mean_difference = (
                    variant_mean - wildtype_mean
                    if np.isfinite(variant_mean) and np.isfinite(wildtype_mean)
                    else np.nan
                )
                test_rows.append(
                    {
                        "family_id": family_id,
                        "metric_name": resolved_metric_name,
                        "metric_label": metric_label_map[resolved_metric_name],
                        "variant_system": variant_system,
                        "variant_label": variant_label,
                        "wildtype_system": wildtype_system,
                        "resid": int(resid),
                        "resname": residue_name_map.get(int(resid), ""),
                        "n_variant_samples": int(variant_values.size),
                        "n_wildtype_samples": int(wildtype_values.size),
                        "variant_mean": variant_mean,
                        "wildtype_mean": wildtype_mean,
                        "mean_difference": mean_difference,
                        "test_statistic": statistic,
                        "p_value": p_value,
                    }
                )

            variant_test_df = pd.DataFrame(test_rows)
            if variant_test_df.empty:
                continue

            variant_test_df["p_value_fdr"] = np.nan
            variant_test_df["significant_fdr"] = False
            valid_mask = variant_test_df["p_value"].notna()
            if valid_mask.any():
                reject, p_value_fdr, _, _ = multipletests(
                    variant_test_df.loc[valid_mask, "p_value"].to_numpy(dtype=float),
                    alpha=float(alpha),
                    method="fdr_bh",
                )
                variant_test_df.loc[valid_mask, "p_value_fdr"] = p_value_fdr
                variant_test_df.loc[valid_mask, "significant_fdr"] = reject

            variant_test_df["direction_vs_wt"] = np.where(
                variant_test_df["mean_difference"] > 0.0,
                "positive",
                np.where(
                    variant_test_df["mean_difference"] < 0.0,
                    "negative",
                    "same",
                ),
            )
            variant_test_tables.append(variant_test_df)

        if not variant_test_tables:
            continue

        family_test_df = pd.concat(variant_test_tables, ignore_index=True)
        family_test_df["significant_positive"] = (
            family_test_df["significant_fdr"]
            & family_test_df["direction_vs_wt"].eq("positive")
        )
        family_test_df["significant_negative"] = (
            family_test_df["significant_fdr"]
            & family_test_df["direction_vs_wt"].eq("negative")
        )

        family_summary_df = (
            family_test_df.groupby(
                ["family_id", "metric_name", "metric_label", "resid", "resname"],
                as_index=False,
            )
            .agg(
                n_variants=("variant_system", "nunique"),
                n_variants_tested=("p_value", lambda values: int(np.isfinite(values).sum())),
                n_variants_significant_positive=("significant_positive", "sum"),
                n_variants_significant_negative=("significant_negative", "sum"),
                mean_score_raw=("mean_difference", "mean"),
                mean_gain_raw=(
                    "mean_difference",
                    lambda values: float(
                        np.mean(
                            np.asarray(values, dtype=float)[
                                np.asarray(values, dtype=float) > 0.0
                            ]
                        )
                    )
                    if np.any(np.asarray(values, dtype=float) > 0.0)
                    else 0.0,
                ),
                mean_loss_raw=(
                    "mean_difference",
                    lambda values: float(
                        np.mean(
                            np.asarray(values, dtype=float)[
                                np.asarray(values, dtype=float) < 0.0
                            ]
                        )
                    )
                    if np.any(np.asarray(values, dtype=float) < 0.0)
                    else 0.0,
                ),
            )
            .sort_values(["family_id", "resid"])
            .reset_index(drop=True)
        )
        family_summary_df["fraction_variants_significant_positive"] = np.where(
            family_summary_df["n_variants_tested"] > 0,
            family_summary_df["n_variants_significant_positive"]
            / family_summary_df["n_variants_tested"],
            0.0,
        )
        family_summary_df["fraction_variants_significant_negative"] = np.where(
            family_summary_df["n_variants_tested"] > 0,
            family_summary_df["n_variants_significant_negative"]
            / family_summary_df["n_variants_tested"],
            0.0,
        )
        family_summary_df["majority_fraction"] = family_summary_df[
            [
                "fraction_variants_significant_positive",
                "fraction_variants_significant_negative",
            ]
        ].max(axis=1)
        family_summary_df["majority_direction"] = np.where(
            (
                family_summary_df["fraction_variants_significant_positive"]
                > family_summary_df["fraction_variants_significant_negative"]
            )
            & (
                family_summary_df["fraction_variants_significant_positive"]
                > float(majority_fraction_threshold)
            ),
            "positive",
            np.where(
                (
                    family_summary_df["fraction_variants_significant_negative"]
                    > family_summary_df["fraction_variants_significant_positive"]
                )
                & (
                    family_summary_df["fraction_variants_significant_negative"]
                    > float(majority_fraction_threshold)
                ),
                "negative",
                "neutral",
            ),
        )
        family_summary_df["majority_fraction_threshold"] = float(majority_fraction_threshold)
        family_summary_df["alpha_fdr"] = float(alpha)
        family_rows.append(family_summary_df)

    if not family_rows:
        return pd.DataFrame()
    return pd.concat(family_rows, ignore_index=True).sort_values(
        ["family_id", "resid"]
    ).reset_index(drop=True)


def _build_md_variant_pair_replicate_matrix(
    pair_replicate_df: pd.DataFrame,
    system_name: str,
    pair_keys: Iterable[str],
    replicate_indices: Iterable[int],
) -> pd.DataFrame:
    pair_index = pd.Index([str(pair_key) for pair_key in pair_keys], name="pair_key")
    replicate_columns = [int(replicate_index) for replicate_index in replicate_indices]
    if not len(pair_index) or not len(replicate_columns):
        return pd.DataFrame(index=pair_index, columns=replicate_columns, dtype=float).fillna(0.0)

    system_df = pair_replicate_df.loc[
        pair_replicate_df["system"].eq(system_name)
    ].copy()
    if system_df.empty:
        return pd.DataFrame(
            0.0,
            index=pair_index,
            columns=replicate_columns,
        )

    system_df["replicate_index"] = pd.to_numeric(
        system_df["replicate_index"],
        errors="coerce",
    ).astype("Int64")
    system_df = system_df.dropna(subset=["replicate_index"]).copy()
    system_df["replicate_index"] = system_df["replicate_index"].astype(int)

    matrix_df = system_df.pivot_table(
        index="pair_key",
        columns="replicate_index",
        values="occupancy_mean",
        aggfunc="mean",
        fill_value=0.0,
    )
    matrix_df = matrix_df.reindex(index=pair_index, columns=replicate_columns, fill_value=0.0)
    return matrix_df.astype(float)


def _build_wildtype_chimerax_pair_gain_focus_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    metric_name: str,
    alpha: float = 0.05,
    return_variant_tests: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)
    if resolved_metric_name not in {"hbond_pairs", "saltbridge_pairs"}:
        raise KeyError(
            "WT ChimeraX interaction focus is only supported for "
            "'hbond_pairs' and 'saltbridge_pairs'."
        )

    pair_replicate_df = md_variant_bundle.pair_replicate_by_metric.get(
        resolved_metric_name,
        pd.DataFrame(),
    ).copy()
    if pair_replicate_df.empty:
        empty_summary_df = pd.DataFrame(
            columns=[
                "family_id",
                "metric_name",
                "metric_label",
                "pair_key",
                "pair_label",
                "resid_a",
                "resname_a",
                "resid_b",
                "resname_b",
                "n_variants",
                "n_variants_tested",
                "n_variants_positive",
                "n_variants_significant_positive",
                "n_variants_significant_negative",
                "fraction_variants_positive",
                "fraction_variants_significant_positive",
                "fraction_variants_significant_negative",
                "dominant_fraction",
                "dominant_direction",
                "mean_variant_mean",
                "mean_wildtype_mean",
                "mean_score_raw",
                "mean_gain_raw",
                "mean_loss_raw",
                "alpha_fdr",
                "positive_variant_labels",
                "significant_positive_variant_labels",
            ]
        )
        return (
            (empty_summary_df, pd.DataFrame())
            if return_variant_tests
            else empty_summary_df
        )

    metric_label_map = {
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }
    metadata_df = md_variant_bundle.metadata_df
    family_rows = []
    all_variant_test_tables = []

    for family_id in md_variant_bundle.family_ids:
        family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
        if family_metadata_df.empty or not family_metadata_df["is_wildtype"].any():
            continue

        wildtype_system = family_metadata_df.loc[
            family_metadata_df["is_wildtype"],
            "system",
        ].iat[0]
        variant_systems = compute_md_variant_order(metadata_df, family_id)
        if not variant_systems:
            continue

        family_systems = [wildtype_system, *variant_systems]
        family_pair_df = pair_replicate_df.loc[
            pair_replicate_df["system"].isin(family_systems)
        ].copy()
        if family_pair_df.empty:
            continue

        pair_lookup_df = (
            family_pair_df[
                [
                    "pair_key",
                    "pair_label",
                    "resid_a",
                    "resname_a",
                    "resid_b",
                    "resname_b",
                ]
            ]
            .drop_duplicates(subset=["pair_key"])
            .sort_values(["resid_a", "resid_b", "pair_label"])
            .reset_index(drop=True)
        )
        pair_keys = pair_lookup_df["pair_key"].astype(str).tolist()
        if not pair_keys:
            continue

        wildtype_replicate_count = int(md_variant_bundle.replicate_count_by_system.get(wildtype_system, 0))
        if wildtype_replicate_count < 2:
            continue
        wildtype_matrix_df = _build_md_variant_pair_replicate_matrix(
            pair_replicate_df=family_pair_df,
            system_name=wildtype_system,
            pair_keys=pair_keys,
            replicate_indices=range(1, wildtype_replicate_count + 1),
        )

        variant_test_tables = []
        for variant_system in variant_systems:
            variant_replicate_count = int(
                md_variant_bundle.replicate_count_by_system.get(variant_system, 0)
            )
            if variant_replicate_count < 2:
                continue

            variant_matrix_df = _build_md_variant_pair_replicate_matrix(
                pair_replicate_df=family_pair_df,
                system_name=variant_system,
                pair_keys=pair_keys,
                replicate_indices=range(1, variant_replicate_count + 1),
            )
            variant_label = get_md_variant_display_label(
                metadata_df,
                variant_system,
            )

            test_rows = []
            for pair_row in pair_lookup_df.itertuples(index=False):
                pair_key = str(pair_row.pair_key)
                variant_values = variant_matrix_df.loc[pair_key].to_numpy(dtype=float)
                wildtype_values = wildtype_matrix_df.loc[pair_key].to_numpy(dtype=float)
                statistic, p_value = _compute_welch_ttest(
                    variant_values,
                    wildtype_values,
                )
                variant_mean = (
                    float(np.mean(variant_values))
                    if variant_values.size
                    else np.nan
                )
                wildtype_mean = (
                    float(np.mean(wildtype_values))
                    if wildtype_values.size
                    else np.nan
                )
                mean_difference = (
                    variant_mean - wildtype_mean
                    if np.isfinite(variant_mean) and np.isfinite(wildtype_mean)
                    else np.nan
                )
                test_rows.append(
                    {
                        "family_id": family_id,
                        "metric_name": resolved_metric_name,
                        "metric_label": metric_label_map[resolved_metric_name],
                        "variant_system": variant_system,
                        "variant_label": variant_label,
                        "wildtype_system": wildtype_system,
                        "pair_key": pair_key,
                        "pair_label": str(pair_row.pair_label),
                        "resid_a": int(pair_row.resid_a),
                        "resname_a": str(pair_row.resname_a),
                        "resid_b": int(pair_row.resid_b),
                        "resname_b": str(pair_row.resname_b),
                        "n_variant_samples": int(variant_values.size),
                        "n_wildtype_samples": int(wildtype_values.size),
                        "variant_mean": variant_mean,
                        "wildtype_mean": wildtype_mean,
                        "mean_difference": mean_difference,
                        "test_statistic": statistic,
                        "p_value": p_value,
                    }
                )

            variant_test_df = pd.DataFrame(test_rows)
            if variant_test_df.empty:
                continue

            variant_test_df["p_value_fdr"] = np.nan
            variant_test_df["significant_fdr"] = False
            valid_mask = variant_test_df["p_value"].notna()
            if valid_mask.any():
                reject, p_value_fdr, _, _ = multipletests(
                    variant_test_df.loc[valid_mask, "p_value"].to_numpy(dtype=float),
                    alpha=float(alpha),
                    method="fdr_bh",
                )
                variant_test_df.loc[valid_mask, "p_value_fdr"] = p_value_fdr
                variant_test_df.loc[valid_mask, "significant_fdr"] = reject

            variant_test_df["direction_vs_wt"] = np.where(
                variant_test_df["mean_difference"] > 0.0,
                "positive",
                np.where(
                    variant_test_df["mean_difference"] < 0.0,
                    "negative",
                    "same",
                ),
            )
            variant_test_tables.append(variant_test_df)

        if not variant_test_tables:
            continue

        family_test_df = pd.concat(variant_test_tables, ignore_index=True)
        family_test_df["positive"] = family_test_df["direction_vs_wt"].eq("positive")
        family_test_df["significant_positive"] = (
            family_test_df["significant_fdr"]
            & family_test_df["direction_vs_wt"].eq("positive")
        )
        family_test_df["significant_negative"] = (
            family_test_df["significant_fdr"]
            & family_test_df["direction_vs_wt"].eq("negative")
        )
        all_variant_test_tables.append(family_test_df.copy())

        family_summary_df = (
            family_test_df.groupby(
                [
                    "family_id",
                    "metric_name",
                    "metric_label",
                    "pair_key",
                    "pair_label",
                    "resid_a",
                    "resname_a",
                    "resid_b",
                    "resname_b",
                ],
                as_index=False,
            )
            .agg(
                n_variants=("variant_system", "nunique"),
                n_variants_tested=("p_value", lambda values: int(np.isfinite(values).sum())),
                n_variants_positive=("positive", "sum"),
                n_variants_significant_positive=("significant_positive", "sum"),
                n_variants_significant_negative=("significant_negative", "sum"),
                mean_variant_mean=("variant_mean", "mean"),
                mean_wildtype_mean=("wildtype_mean", "mean"),
                mean_score_raw=("mean_difference", "mean"),
                mean_gain_raw=(
                    "mean_difference",
                    lambda values: float(
                        np.mean(
                            np.asarray(values, dtype=float)[
                                np.asarray(values, dtype=float) > 0.0
                            ]
                        )
                    )
                    if np.any(np.asarray(values, dtype=float) > 0.0)
                    else 0.0,
                ),
                mean_loss_raw=(
                    "mean_difference",
                    lambda values: float(
                        np.mean(
                            np.asarray(values, dtype=float)[
                                np.asarray(values, dtype=float) < 0.0
                            ]
                        )
                    )
                    if np.any(np.asarray(values, dtype=float) < 0.0)
                    else 0.0,
                ),
            )
            .sort_values(["family_id", "resid_a", "resid_b", "pair_label"])
            .reset_index(drop=True)
        )
        family_summary_df["fraction_variants_positive"] = np.where(
            family_summary_df["n_variants_tested"] > 0,
            family_summary_df["n_variants_positive"]
            / family_summary_df["n_variants_tested"],
            0.0,
        )
        family_summary_df["fraction_variants_significant_positive"] = np.where(
            family_summary_df["n_variants_tested"] > 0,
            family_summary_df["n_variants_significant_positive"]
            / family_summary_df["n_variants_tested"],
            0.0,
        )
        family_summary_df["fraction_variants_significant_negative"] = np.where(
            family_summary_df["n_variants_tested"] > 0,
            family_summary_df["n_variants_significant_negative"]
            / family_summary_df["n_variants_tested"],
            0.0,
        )
        family_summary_df["dominant_fraction"] = family_summary_df[
            [
                "fraction_variants_significant_positive",
                "fraction_variants_significant_negative",
            ]
        ].max(axis=1)
        family_summary_df["dominant_direction"] = np.where(
            family_summary_df["fraction_variants_significant_positive"]
            > family_summary_df["fraction_variants_significant_negative"],
            "positive",
            np.where(
                family_summary_df["fraction_variants_significant_negative"]
                > family_summary_df["fraction_variants_significant_positive"],
                "negative",
                "neutral",
            ),
        )
        family_summary_df["alpha_fdr"] = float(alpha)

        positive_label_map = (
            family_test_df.loc[family_test_df["positive"]]
            .groupby("pair_key")["variant_label"]
            .agg(lambda values: "; ".join(sorted(dict.fromkeys(str(value) for value in values))))
            .to_dict()
        )
        significant_positive_label_map = (
            family_test_df.loc[family_test_df["significant_positive"]]
            .groupby("pair_key")["variant_label"]
            .agg(lambda values: "; ".join(sorted(dict.fromkeys(str(value) for value in values))))
            .to_dict()
        )
        family_summary_df["positive_variant_labels"] = family_summary_df["pair_key"].map(
            positive_label_map
        ).fillna("")
        family_summary_df["significant_positive_variant_labels"] = family_summary_df[
            "pair_key"
        ].map(significant_positive_label_map).fillna("")
        family_rows.append(family_summary_df)

    if not family_rows:
        empty_summary_df = pd.DataFrame()
        empty_test_df = pd.concat(all_variant_test_tables, ignore_index=True) if all_variant_test_tables else pd.DataFrame()
        return (
            (empty_summary_df, empty_test_df)
            if return_variant_tests
            else empty_summary_df
        )
    summary_df = pd.concat(family_rows, ignore_index=True).sort_values(
        [
            "family_id",
            "fraction_variants_significant_positive",
            "n_variants_significant_positive",
            "mean_gain_raw",
            "pair_label",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    if not return_variant_tests:
        return summary_df
    variant_test_df = (
        pd.concat(all_variant_test_tables, ignore_index=True)
        if all_variant_test_tables
        else pd.DataFrame()
    )
    return summary_df, variant_test_df


def _build_wildtype_chimerax_interaction_change_score_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    interaction_change_z_threshold: float = 1.0,
    interaction_change_z_breaks: Iterable[float] | None = None,
    interaction_colors: dict[str, Iterable[str]] | None = None,
    neutral_color: str = "#d9d9d9",
) -> pd.DataFrame:
    resolved_z_breaks = _normalize_signed_z_breaks(
        (interaction_change_z_threshold,)
        if interaction_change_z_breaks is None
        else interaction_change_z_breaks
    )
    resolved_interaction_colors = _normalize_chimerax_interaction_colors(
        interaction_colors=interaction_colors,
        n_intervals=len(resolved_z_breaks),
    )
    metric_tables = {
        metric_name: _build_wildtype_chimerax_family_average_score_table(
            md_variant_bundle=md_variant_bundle,
            wildtype_alignment_df=None,
            metric_name=metric_name,
        )
        for metric_name in ("hbond_pairs", "saltbridge_pairs")
    }

    merged_df = None
    for metric_name, metric_table_df in metric_tables.items():
        rename_prefix = metric_name.replace("_pairs", "")
        metric_subset_df = metric_table_df[
            [
                "family_id",
                "resid",
                "resname",
                "n_variants",
                "mean_gain_raw",
                "mean_loss_raw",
            ]
        ].rename(
            columns={
                "n_variants": f"{rename_prefix}_n_variants",
                "mean_gain_raw": f"{rename_prefix}_gain_raw",
                "mean_loss_raw": f"{rename_prefix}_loss_raw",
            }
        )
        if merged_df is None:
            merged_df = metric_subset_df
        else:
            merged_df = merged_df.merge(
                metric_subset_df,
                on=["family_id", "resid", "resname"],
                how="outer",
            )

    if merged_df is None or merged_df.empty:
        return pd.DataFrame()

    merged_df = merged_df.fillna(0.0).copy()
    merged_df["n_variants"] = merged_df[
        ["hbond_n_variants", "saltbridge_n_variants"]
    ].max(axis=1).astype(int)
    merged_df["hbond_loss_magnitude_raw"] = (
        -pd.to_numeric(merged_df["hbond_loss_raw"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    merged_df["saltbridge_loss_magnitude_raw"] = (
        -pd.to_numeric(merged_df["saltbridge_loss_raw"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    merged_df["hbond_gain_raw"] = pd.to_numeric(
        merged_df["hbond_gain_raw"], errors="coerce"
    ).fillna(0.0)
    merged_df["saltbridge_gain_raw"] = pd.to_numeric(
        merged_df["saltbridge_gain_raw"], errors="coerce"
    ).fillna(0.0)

    channel_specs = [
        ("hbond_gain", "hbond_gain_raw", "Hydrogen-bond gain"),
        ("hbond_loss", "hbond_loss_magnitude_raw", "Hydrogen-bond loss"),
        ("saltbridge_gain", "saltbridge_gain_raw", "Salt-bridge gain"),
        ("saltbridge_loss", "saltbridge_loss_magnitude_raw", "Salt-bridge loss"),
    ]
    channel_order = [spec[0] for spec in channel_specs]
    channel_label_map = {channel_name: label for channel_name, _, label in channel_specs}

    for channel_name, raw_column_name, _ in channel_specs:
        merged_df[raw_column_name] = pd.to_numeric(
            merged_df[raw_column_name],
            errors="coerce",
        ).fillna(0.0)
        merged_df[f"{channel_name}_score_z"] = (
            merged_df.groupby("family_id", group_keys=False)[raw_column_name]
            .apply(_compute_md_variant_group_zscores)
            .astype(float)
        )

    z_score_columns = [f"{channel_name}_score_z" for channel_name in channel_order]
    z_score_matrix = merged_df[z_score_columns].to_numpy(dtype=float)
    dominant_channel_indices = np.argmax(z_score_matrix, axis=1)
    dominant_score_z = z_score_matrix[np.arange(len(merged_df)), dominant_channel_indices]
    dominant_channel_names = np.asarray(channel_order, dtype=object)[dominant_channel_indices]
    dominant_raw_values = np.asarray(
        [
            merged_df.iloc[row_index][
                next(
                    raw_column_name
                    for channel_name, raw_column_name, _ in channel_specs
                    if channel_name == dominant_channel_names[row_index]
                )
            ]
            for row_index in range(len(merged_df))
        ],
        dtype=float,
    )

    merged_df["dominant_channel"] = dominant_channel_names
    merged_df["dominant_label"] = [
        channel_label_map[channel_name]
        for channel_name in dominant_channel_names
    ]
    merged_df["dominant_score_z"] = dominant_score_z.astype(float)
    merged_df["dominant_score_raw"] = dominant_raw_values
    merged_df["passes_threshold"] = merged_df["dominant_score_z"] >= resolved_z_breaks[0]
    scale_limit = float(resolved_z_breaks[-1])

    merged_df["normalized_value"] = np.clip(
        merged_df["dominant_score_z"].astype(float) / scale_limit,
        0.0,
        1.0,
    )
    merged_df["color_hex"] = neutral_color
    merged_df["interaction_z_bin"] = -1
    merged_df["interaction_z_lower"] = np.nan
    merged_df["interaction_z_upper"] = np.nan
    merged_df["interaction_z_bin_label"] = f"z < {resolved_z_breaks[0]:0.3g}"
    merged_df["interaction_palette_index"] = pd.Series(pd.NA, index=merged_df.index, dtype="Int64")
    for channel_name, _, _ in channel_specs:
        channel_mask = merged_df["dominant_channel"].eq(channel_name) & merged_df["passes_threshold"]
        if not channel_mask.any():
            continue
        for break_index, break_value in enumerate(resolved_z_breaks):
            next_break_value = (
                resolved_z_breaks[break_index + 1]
                if break_index + 1 < len(resolved_z_breaks)
                else None
            )
            interval_mask = channel_mask & (merged_df["dominant_score_z"] >= break_value)
            if next_break_value is not None:
                interval_mask &= merged_df["dominant_score_z"] < next_break_value
                interval_label = f"{break_value:0.3g} <= z < {next_break_value:0.3g}"
            else:
                interval_label = f"z >= {break_value:0.3g}"
            merged_df.loc[interval_mask, "color_hex"] = resolved_interaction_colors[
                channel_name
            ][break_index]
            merged_df.loc[interval_mask, "interaction_z_bin"] = break_index
            merged_df.loc[interval_mask, "interaction_z_lower"] = break_value
            merged_df.loc[interval_mask, "interaction_z_upper"] = next_break_value
            merged_df.loc[interval_mask, "interaction_z_bin_label"] = interval_label
            merged_df.loc[interval_mask, "interaction_palette_index"] = break_index

    merged_df["metric_name"] = "interaction_change"
    merged_df["metric_label"] = "H-bond and salt-bridge change"
    merged_df["color_mode"] = "combined"
    merged_df["scale_limit"] = float(scale_limit)
    merged_df["mean_score_raw"] = np.where(
        merged_df["dominant_channel"].str.endswith("loss"),
        -merged_df["dominant_score_raw"].astype(float),
        merged_df["dominant_score_raw"].astype(float),
    )
    merged_df["mean_gain_raw"] = np.where(
        merged_df["dominant_channel"].str.endswith("gain"),
        merged_df["dominant_score_raw"].astype(float),
        0.0,
    )
    merged_df["mean_loss_raw"] = np.where(
        merged_df["dominant_channel"].str.endswith("loss"),
        -merged_df["dominant_score_raw"].astype(float),
        0.0,
    )
    merged_df["fraction_variants_positive"] = np.where(
        merged_df["dominant_channel"].str.endswith("gain"),
        1.0,
        0.0,
    )
    merged_df["fraction_variants_negative"] = np.where(
        merged_df["dominant_channel"].str.endswith("loss"),
        1.0,
        0.0,
    )
    merged_df["interaction_change_z_threshold"] = resolved_z_breaks[0]
    merged_df["interaction_change_z_breaks"] = repr(resolved_z_breaks)
    merged_df["component_color_hex"] = merged_df["color_hex"]
    return merged_df.sort_values(["family_id", "resid"]).reset_index(drop=True)


def build_wildtype_chimerax_command_bundle(
    md_variant_bundle: MDVariantAnalysisBundle,
    wildtype_alignment_df: pd.DataFrame,
    metric_name: str = "rmsf",
    color_mode: str = "signed",
    combined_pdb_path: Path | None = None,
    model_id: str = "#1",
    target_layout: str = "combined_model",
    separate_model_id_by_family: dict[str, str] | None = None,
    separate_model_chain_id: str = "A",
    interaction_change_z_threshold: float = 1.0,
    interaction_change_z_breaks: Iterable[float] | None = None,
    interaction_colors: dict[str, Iterable[str]] | None = None,
    significance_alpha: float = 0.05,
    majority_fraction_threshold: float = 0.5,
    emphasize_colored_residues: bool = False,
    emphasis_style: str = "ball",
    signed_score_mode: str = "zscore",
    signed_z_breaks: Iterable[float] = (1.0, 2.0, 3.0),
    n_color_bins: int = 9,
    neutral_color: str = "#d9d9d9",
) -> WildtypeChimeraXCommandBundle:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)
    resolved_color_mode = str(color_mode).strip().lower()
    resolved_signed_score_mode = str(signed_score_mode).strip().lower()
    if resolved_color_mode not in {
        "signed",
        "destabilizing",
        "stabilizing",
        "combined",
        "majority_significance",
    }:
        raise KeyError(
            "Unsupported WT ChimeraX color mode. Expected one of "
            "'signed', 'destabilizing', 'stabilizing', 'combined', or "
            "'majority_significance'."
        )
    if resolved_signed_score_mode not in {"raw", "zscore"}:
        raise KeyError(
            "Unsupported WT ChimeraX signed score mode. Expected one of "
            "'raw' or 'zscore'."
        )
    if resolved_metric_name == "interaction_change":
        if resolved_color_mode != "combined":
            raise KeyError(
                "WT ChimeraX metric 'interaction_change' requires "
                "color_mode='combined'."
            )
    elif resolved_color_mode == "combined":
        raise KeyError(
            "WT ChimeraX color_mode='combined' is only supported for "
            "metric_name='interaction_change'."
        )
    if resolved_color_mode == "majority_significance" and resolved_metric_name not in {
        "hbond_pairs",
        "saltbridge_pairs",
    }:
        raise KeyError(
            "WT ChimeraX color_mode='majority_significance' is only supported for "
            "'hbond_pairs' and 'saltbridge_pairs'."
        )
    resolved_emphasis_style = str(emphasis_style).strip().lower()
    if resolved_emphasis_style not in {"ball", "stick", "sphere"}:
        raise KeyError(
            "Unsupported WT ChimeraX emphasis style. Expected one of "
            "'ball', 'stick', or 'sphere'."
        )

    if resolved_metric_name == "interaction_change":
        resolved_interaction_z_breaks = _normalize_signed_z_breaks(
            (interaction_change_z_threshold,)
            if interaction_change_z_breaks is None
            else interaction_change_z_breaks
        )
        resolved_interaction_colors = _normalize_chimerax_interaction_colors(
            interaction_colors=interaction_colors,
            n_intervals=len(resolved_interaction_z_breaks),
        )
        residue_color_df = _build_wildtype_chimerax_interaction_change_score_table(
            md_variant_bundle=md_variant_bundle,
            interaction_change_z_threshold=interaction_change_z_threshold,
            interaction_change_z_breaks=interaction_change_z_breaks,
            interaction_colors=interaction_colors,
            neutral_color=neutral_color,
        )
    elif resolved_color_mode == "majority_significance":
        residue_color_df = _build_wildtype_chimerax_pair_majority_significance_score_table(
            md_variant_bundle=md_variant_bundle,
            metric_name=resolved_metric_name,
            alpha=significance_alpha,
            majority_fraction_threshold=majority_fraction_threshold,
        )
    else:
        residue_color_df = _build_wildtype_chimerax_family_average_score_table(
            md_variant_bundle=md_variant_bundle,
            wildtype_alignment_df=wildtype_alignment_df,
            metric_name=resolved_metric_name,
        )
    target_spec_df = _resolve_wildtype_chimerax_target_spec_df(
        wildtype_alignment_df=wildtype_alignment_df,
        target_layout=target_layout,
        combined_model_id=model_id,
        separate_model_id_by_family=separate_model_id_by_family,
        separate_model_chain_id=separate_model_chain_id,
    )
    if residue_color_df.empty:
        return WildtypeChimeraXCommandBundle(
            residue_color_df=pd.DataFrame(),
            apply_command="",
            open_command="",
        )

    residue_color_df = residue_color_df.copy()
    residue_color_df = residue_color_df.merge(
        target_spec_df[
            [
                "family_id",
                "aligned_pdb_path",
                "target_model_id",
                "target_chain_id",
            ]
        ],
        on="family_id",
        how="left",
        validate="many_to_one",
    )
    if resolved_metric_name == "interaction_change":
        scale_limit = float(residue_color_df["scale_limit"].iloc[0]) if not residue_color_df.empty else float(
            interaction_change_z_threshold
        )
    elif resolved_color_mode == "signed":
        residue_color_df["color_hex"] = neutral_color
        if resolved_signed_score_mode == "zscore":
            resolved_signed_z_breaks = _normalize_signed_z_breaks(signed_z_breaks)
            negative_bin_colors = ["#c2d3e4", "#578abf", "#2166ac"]
            positive_bin_colors = ["#e5bec3", "#d48690", "#b2182b"]
            if len(resolved_signed_z_breaks) != 3:
                negative_bin_colors = [
                    to_hex(DEFAULT_RMSF_DELTA_CMAP(value))
                    for value in np.linspace(0.38, 0.08, len(resolved_signed_z_breaks))
                ]
                positive_bin_colors = [
                    to_hex(DEFAULT_RMSF_DELTA_CMAP(value))
                    for value in np.linspace(0.62, 0.92, len(resolved_signed_z_breaks))
                ]
            residue_color_df["color_value"] = pd.to_numeric(
                residue_color_df["mean_score_z"],
                errors="coerce",
            ).fillna(0.0)
            residue_color_df["color_value_label"] = "Mean WT-relative delta z-score across variants"
            scale_limit = float(max(resolved_signed_z_breaks))
            residue_color_df["normalized_value"] = np.clip(
                residue_color_df["color_value"].astype(float) / scale_limit,
                -1.0,
                1.0,
            )
            residue_color_df["signed_abs_z"] = residue_color_df["color_value"].abs()
            residue_color_df["signed_z_bin_label"] = (
                f"|z| < {resolved_signed_z_breaks[0]:0.3g}"
            )
            for break_index, break_value in enumerate(resolved_signed_z_breaks):
                next_break_value = (
                    resolved_signed_z_breaks[break_index + 1]
                    if break_index + 1 < len(resolved_signed_z_breaks)
                    else None
                )
                if next_break_value is None:
                    bin_label = f"|z| >= {break_value:0.3g}"
                    bin_mask = residue_color_df["signed_abs_z"] >= float(break_value)
                else:
                    bin_label = f"{break_value:0.3g} <= |z| < {next_break_value:0.3g}"
                    bin_mask = (
                        residue_color_df["signed_abs_z"] >= float(break_value)
                    ) & (
                        residue_color_df["signed_abs_z"] < float(next_break_value)
                    )
                negative_mask = bin_mask & (residue_color_df["color_value"] < 0.0)
                positive_mask = bin_mask & (residue_color_df["color_value"] > 0.0)
                residue_color_df.loc[negative_mask, "color_hex"] = negative_bin_colors[
                    min(break_index, len(negative_bin_colors) - 1)
                ]
                residue_color_df.loc[positive_mask, "color_hex"] = positive_bin_colors[
                    min(break_index, len(positive_bin_colors) - 1)
                ]
                residue_color_df.loc[bin_mask, "signed_z_bin_label"] = bin_label
        else:
            residue_color_df["color_value"] = residue_color_df["mean_score_raw"].astype(float)
            residue_color_df["color_value_label"] = "Mean WT-relative delta across variants"
            nonzero_mask = residue_color_df["color_value"].abs() > 0.0
            nonzero_values = residue_color_df.loc[nonzero_mask, "color_value"].abs().to_numpy(dtype=float)
            scale_limit = float(np.nanquantile(nonzero_values, 0.95)) if len(nonzero_values) else 0.0
            if not np.isfinite(scale_limit) or scale_limit <= 0.0:
                scale_limit = float(nonzero_values.max()) if len(nonzero_values) else 1.0
            scale_limit = max(scale_limit, 1e-9)
            residue_color_df["normalized_value"] = np.clip(
                residue_color_df["color_value"].astype(float) / scale_limit,
                -1.0,
                1.0,
            )
            neutral_threshold = 1.0 / max(float(n_color_bins), 8.0)
            palette = [
                to_hex(DEFAULT_RMSF_DELTA_CMAP(value))
                for value in np.linspace(0.0, 1.0, max(int(n_color_bins), 3))
            ]
            colored_mask = residue_color_df["normalized_value"].abs() >= neutral_threshold
            if colored_mask.any():
                signed_fraction = (
                    residue_color_df.loc[colored_mask, "normalized_value"].to_numpy(dtype=float) + 1.0
                ) / 2.0
                color_indices = np.floor(signed_fraction * len(palette)).astype(int)
                color_indices = np.clip(color_indices, 0, len(palette) - 1)
                residue_color_df.loc[colored_mask, "color_hex"] = [
                    palette[int(index)]
                    for index in color_indices
                ]
    elif resolved_color_mode == "majority_significance":
        residue_color_df["color_value"] = pd.to_numeric(
            residue_color_df["majority_fraction"],
            errors="coerce",
        ).fillna(0.0)
        residue_color_df["color_value_label"] = (
            "Fraction of variants with FDR-significant WT-relative occupancy change "
            "in the dominant direction"
        )
        residue_color_df["normalized_value"] = np.clip(
            residue_color_df["color_value"].astype(float),
            0.0,
            1.0,
        )
        residue_color_df["color_hex"] = neutral_color
        residue_color_df["majority_fraction_bin_label"] = (
            "<= "
            + residue_color_df["majority_fraction_threshold"].astype(float).map(
                lambda value: f"{value:0.3g}"
            )
        )
        low_positive_color, mid_positive_color, high_positive_color = (
            "#e5bec3",
            "#d48690",
            "#b2182b",
        )
        low_negative_color, mid_negative_color, high_negative_color = (
            "#c2d3e4",
            "#578abf",
            "#2166ac",
        )
        lower_mask = (
            residue_color_df["majority_direction"].eq("negative")
            & (residue_color_df["majority_fraction"] > residue_color_df["majority_fraction_threshold"])
        )
        higher_mask = (
            residue_color_df["majority_direction"].eq("positive")
            & (residue_color_df["majority_fraction"] > residue_color_df["majority_fraction_threshold"])
        )
        threshold_values = residue_color_df["majority_fraction_threshold"].astype(float)
        universal_mask = residue_color_df["majority_fraction"] >= 0.999999
        strong_mask = residue_color_df["majority_fraction"] >= 0.75
        moderate_mask = residue_color_df["majority_fraction"] > threshold_values
        residue_color_df.loc[moderate_mask, "majority_fraction_bin_label"] = (
            f"> {float(residue_color_df['majority_fraction_threshold'].iloc[0]):0.3g} to < 0.75"
        )
        residue_color_df.loc[strong_mask, "majority_fraction_bin_label"] = "0.75 to < 1.0"
        residue_color_df.loc[universal_mask, "majority_fraction_bin_label"] = "1.0"
        residue_color_df.loc[higher_mask & moderate_mask, "color_hex"] = low_positive_color
        residue_color_df.loc[higher_mask & strong_mask, "color_hex"] = mid_positive_color
        residue_color_df.loc[higher_mask & universal_mask, "color_hex"] = high_positive_color
        residue_color_df.loc[lower_mask & moderate_mask, "color_hex"] = low_negative_color
        residue_color_df.loc[lower_mask & strong_mask, "color_hex"] = mid_negative_color
        residue_color_df.loc[lower_mask & universal_mask, "color_hex"] = high_negative_color
        scale_limit = 1.0
    else:
        if resolved_metric_name == "rmsf":
            if resolved_color_mode == "destabilizing":
                focus_values = residue_color_df["mean_score_raw"].clip(lower=0.0)
                focus_label = "Mean RMSF increase across variants"
            else:
                focus_values = (-residue_color_df["mean_score_raw"]).clip(lower=0.0)
                focus_label = "Mean RMSF decrease across variants"
        else:
            if resolved_color_mode == "destabilizing":
                focus_values = (-residue_color_df["mean_score_raw"]).clip(lower=0.0)
                focus_label = "Mean interaction loss across variants"
            else:
                focus_values = residue_color_df["mean_score_raw"].clip(lower=0.0)
                focus_label = "Mean interaction gain across variants"

        residue_color_df["color_value"] = focus_values.astype(float)
        residue_color_df["color_value_label"] = focus_label
        positive_values = residue_color_df.loc[
            residue_color_df["color_value"] > 0.0,
            "color_value",
        ].to_numpy(dtype=float)
        scale_limit = float(np.nanquantile(positive_values, 0.95)) if len(positive_values) else 0.0
        if not np.isfinite(scale_limit) or scale_limit <= 0.0:
            scale_limit = float(positive_values.max()) if len(positive_values) else 1.0
        scale_limit = max(scale_limit, 1e-9)
        residue_color_df["normalized_value"] = np.clip(
            residue_color_df["color_value"].astype(float) / scale_limit,
            0.0,
            1.0,
        )
        palette = [
            to_hex(color)
            for color in plt.get_cmap("Reds")(np.linspace(0.35, 0.95, max(int(n_color_bins), 3)))
        ]
        residue_color_df["color_hex"] = neutral_color
        colored_mask = residue_color_df["normalized_value"] > 0.0
        if colored_mask.any():
            positive_fraction = residue_color_df.loc[colored_mask, "normalized_value"].to_numpy(dtype=float)
            color_indices = np.ceil(positive_fraction * (len(palette) - 1)).astype(int)
            color_indices = np.clip(color_indices, 0, len(palette) - 1)
            residue_color_df.loc[colored_mask, "color_hex"] = [
                palette[int(index)]
                for index in color_indices
            ]

    residue_color_df["residue_spec"] = residue_color_df.apply(
        lambda row: f"{row['target_model_id']}/{row['target_chain_id']}:{int(row['resid'])}",
        axis=1,
    )
    residue_color_df["metric_name"] = resolved_metric_name
    residue_color_df["color_mode"] = resolved_color_mode
    residue_color_df["scale_limit"] = float(scale_limit)

    metric_label = str(residue_color_df["metric_label"].iloc[0])
    target_comment = " | ".join(
        (
            f"{row.family_id}={row.target_model_id}/chain {row.target_chain_id}"
            if str(target_layout).strip().lower() == "separate_models"
            else f"{row.family_id}={row.target_model_id}/chain {row.chain_id}"
        )
        for row in target_spec_df.itertuples(index=False)
    )

    color_command_lines = [
        f"# Figure 4 WT ChimeraX coloring | metric: {metric_label} | mode: {resolved_color_mode}",
        f"# Target layout: {str(target_layout).strip().lower()}",
        f"# Target mapping: {target_comment}",
        f"# Neutral color: {neutral_color}",
        f"# Scale cap: {float(scale_limit):0.6g}",
    ]
    if bool(emphasize_colored_residues):
        color_command_lines.append(
            f"# Emphasis overlay: colored residues will also be shown as atoms in {resolved_emphasis_style} style"
        )
    if resolved_metric_name == "interaction_change":
        color_command_lines.append(
            "# Combined interaction z-score intervals: neutral below "
            f"{resolved_interaction_z_breaks[0]:0.3g}"
        )
        interaction_label_map = {
            "saltbridge_gain": "Salt-bridge gain",
            "saltbridge_loss": "Salt-bridge loss",
            "hbond_gain": "H-bond gain",
            "hbond_loss": "H-bond loss",
        }
        for channel_name, channel_label in interaction_label_map.items():
            for break_index, break_value in enumerate(resolved_interaction_z_breaks):
                next_break_value = (
                    resolved_interaction_z_breaks[break_index + 1]
                    if break_index + 1 < len(resolved_interaction_z_breaks)
                    else None
                )
                range_label = (
                    f"{break_value:0.3g} <= z < {next_break_value:0.3g}"
                    if next_break_value is not None
                    else f"z >= {break_value:0.3g}"
                )
                color_command_lines.append(
                    f"#   {channel_label} | {range_label}: "
                    f"{resolved_interaction_colors[channel_name][break_index]}"
                )
    elif resolved_color_mode == "majority_significance":
        majority_fraction_threshold = float(
            residue_color_df["majority_fraction_threshold"].iloc[0]
        )
        alpha_value = float(residue_color_df["alpha_fdr"].iloc[0])
        color_command_lines.extend(
            [
                (
                    "# Majority-significance mode: each residue is colored only when "
                    f"> {majority_fraction_threshold:0.3g} of variants in the family "
                    "show an FDR-significant occupancy change vs WT in the same direction."
                ),
                f"# Variant-level test: Welch t-test on replicate occupancies | FDR alpha={alpha_value:0.3g}",
                "#   negative > threshold to < 0.75: #c2d3e4 | positive > threshold to < 0.75: #e5bec3",
                "#   negative 0.75 to < 1.0: #578abf | positive 0.75 to < 1.0: #d48690",
                "#   negative 1.0: #2166ac | positive 1.0: #b2182b",
            ]
        )
    elif resolved_color_mode == "signed" and resolved_signed_score_mode == "zscore":
        resolved_signed_z_breaks = _normalize_signed_z_breaks(signed_z_breaks)
        negative_bin_colors = ["#c2d3e4", "#578abf", "#2166ac"]
        positive_bin_colors = ["#e5bec3", "#d48690", "#b2182b"]
        if len(resolved_signed_z_breaks) != 3:
            negative_bin_colors = [
                to_hex(DEFAULT_RMSF_DELTA_CMAP(value))
                for value in np.linspace(0.38, 0.08, len(resolved_signed_z_breaks))
            ]
            positive_bin_colors = [
                to_hex(DEFAULT_RMSF_DELTA_CMAP(value))
                for value in np.linspace(0.62, 0.92, len(resolved_signed_z_breaks))
            ]
        color_command_lines.append(
            f"# Signed score mode: zscore | explicit |z| bins with neutral below {resolved_signed_z_breaks[0]:0.3g}"
        )
        for break_index, break_value in enumerate(resolved_signed_z_breaks):
            next_break_value = (
                resolved_signed_z_breaks[break_index + 1]
                if break_index + 1 < len(resolved_signed_z_breaks)
                else None
            )
            if next_break_value is None:
                range_label = f"|z| >= {break_value:0.3g}"
            else:
                range_label = f"{break_value:0.3g} <= |z| < {next_break_value:0.3g}"
            color_command_lines.append(
                f"#   negative {range_label}: {negative_bin_colors[min(break_index, len(negative_bin_colors) - 1)]} | "
                f"positive {range_label}: {positive_bin_colors[min(break_index, len(positive_bin_colors) - 1)]}"
            )
    if str(target_layout).strip().lower() == "combined_model":
        color_command_lines.append(f"color {_normalize_chimerax_model_id(model_id)} {neutral_color}")
    else:
        for row in target_spec_df.itertuples(index=False):
            color_command_lines.append(f"color {row.target_model_id} {neutral_color}")

    colored_residue_df = residue_color_df.loc[
        residue_color_df["color_hex"].ne(neutral_color)
    ].copy()
    if colored_residue_df.empty:
        color_command_lines.append(
            "# No residues exceeded the current coloring threshold; the model remains neutral."
        )
    else:
        grouped_color_df = (
            colored_residue_df.groupby(
                ["family_id", "target_model_id", "target_chain_id", "color_hex"],
                as_index=False,
            )
            .agg(
                residue_numbers=("resid", lambda values: sorted(set(int(value) for value in values))),
                max_normalized_value=("normalized_value", "max"),
            )
            .sort_values(["family_id", "target_model_id", "max_normalized_value"])
        )
        if bool(emphasize_colored_residues):
            emphasis_group_df = (
                colored_residue_df.groupby(
                    ["family_id", "target_model_id", "target_chain_id"],
                    as_index=False,
                )
                .agg(
                    residue_numbers=("resid", lambda values: sorted(set(int(value) for value in values))),
                )
                .sort_values(["family_id", "target_model_id"])
            )
            for row in emphasis_group_df.itertuples(index=False):
                residue_range_spec = _compress_residue_numbers_to_ranges(row.residue_numbers)
                if not residue_range_spec:
                    continue
                residue_spec = f"{row.target_model_id}/{row.target_chain_id}:{residue_range_spec}"
                color_command_lines.append(f"show {residue_spec} atoms")
                color_command_lines.append(f"style {residue_spec} {resolved_emphasis_style}")
        for row in grouped_color_df.itertuples(index=False):
            residue_range_spec = _compress_residue_numbers_to_ranges(row.residue_numbers)
            if not residue_range_spec:
                continue
            color_command_lines.append(
                f"color {row.target_model_id}/{row.target_chain_id}:{residue_range_spec} {row.color_hex}"
            )

    apply_command = "\n".join(color_command_lines)
    open_command_lines = list(color_command_lines)
    insert_index = 5
    if str(target_layout).strip().lower() == "combined_model":
        if combined_pdb_path is not None:
            open_command_lines.insert(
                insert_index,
                f"open {json.dumps(str(Path(combined_pdb_path).resolve()))}",
            )
    else:
        for row in reversed(list(target_spec_df.itertuples(index=False))):
            open_command_lines.insert(
                insert_index,
                f"open {json.dumps(str(Path(row.aligned_pdb_path).resolve()))}",
            )
    open_command = "\n".join(open_command_lines)

    return WildtypeChimeraXCommandBundle(
        residue_color_df=residue_color_df.sort_values(["family_id", "resid"]).reset_index(drop=True),
        apply_command=apply_command,
        open_command=open_command,
    )


def build_wildtype_chimerax_interaction_gain_focus_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    wildtype_alignment_df: pd.DataFrame,
    metric_name: str,
    combined_pdb_path: Path | None = None,
    model_id: str = "#1",
    target_layout: str = "combined_model",
    separate_model_id_by_family: dict[str, str] | None = None,
    separate_model_chain_id: str = "A",
    significance_alpha: float = 0.05,
    minimum_significant_fraction: float | None = 0.5,
    minimum_significant_variant_count: int | None = 2,
    selection_rule: str = "fraction_or_count",
    max_pairs_per_family: int = 3,
    emphasis_style: str = "stick",
    neutral_color: str = "#c1c1c1",
) -> pd.DataFrame:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)
    if resolved_metric_name not in {"hbond_pairs", "saltbridge_pairs"}:
        raise KeyError(
            "WT ChimeraX interaction focus is only supported for "
            "'hbond_pairs' and 'saltbridge_pairs'."
        )
    if minimum_significant_fraction is None and minimum_significant_variant_count is None:
        raise ValueError(
            "At least one interaction-focus selection threshold must be provided."
        )

    resolved_selection_rule = str(selection_rule).strip().lower()
    if resolved_selection_rule not in {
        "fraction_or_count",
        "fraction_and_count",
        "fraction_only",
        "count_only",
    }:
        raise ValueError(
            "Unsupported interaction-focus selection rule. Expected one of "
            "'fraction_or_count', 'fraction_and_count', 'fraction_only', or "
            "'count_only'."
        )

    resolved_emphasis_style = str(emphasis_style).strip().lower()
    if resolved_emphasis_style not in {"ball", "stick", "sphere"}:
        raise ValueError(
            "Unsupported WT ChimeraX interaction-focus emphasis style. Expected one of "
            "'ball', 'stick', or 'sphere'."
        )
    if int(max_pairs_per_family) <= 0:
        raise ValueError("max_pairs_per_family must be positive.")

    focus_df = _build_wildtype_chimerax_pair_gain_focus_score_table(
        md_variant_bundle=md_variant_bundle,
        metric_name=resolved_metric_name,
        alpha=significance_alpha,
    )
    if focus_df.empty:
        return focus_df

    target_spec_df = _resolve_wildtype_chimerax_target_spec_df(
        wildtype_alignment_df=wildtype_alignment_df,
        target_layout=target_layout,
        combined_model_id=model_id,
        separate_model_id_by_family=separate_model_id_by_family,
        separate_model_chain_id=separate_model_chain_id,
    )
    focus_df = focus_df.merge(
        target_spec_df[
            [
                "family_id",
                "aligned_pdb_path",
                "target_model_id",
                "target_chain_id",
            ]
        ],
        on="family_id",
        how="left",
        validate="many_to_one",
    )
    focus_df = focus_df.loc[focus_df["dominant_direction"].eq("positive")].copy()
    if focus_df.empty:
        return focus_df

    fraction_threshold = (
        float(minimum_significant_fraction)
        if minimum_significant_fraction is not None
        else np.nan
    )
    count_threshold = (
        int(minimum_significant_variant_count)
        if minimum_significant_variant_count is not None
        else None
    )
    fraction_mask = pd.Series(False, index=focus_df.index)
    count_mask = pd.Series(False, index=focus_df.index)
    if minimum_significant_fraction is not None:
        fraction_mask = focus_df["fraction_variants_significant_positive"] >= fraction_threshold
    if minimum_significant_variant_count is not None:
        count_mask = focus_df["n_variants_significant_positive"] >= count_threshold

    if resolved_selection_rule == "fraction_or_count":
        selected_mask = fraction_mask | count_mask
    elif resolved_selection_rule == "fraction_and_count":
        selected_mask = fraction_mask & count_mask
    elif resolved_selection_rule == "fraction_only":
        selected_mask = fraction_mask
    else:
        selected_mask = count_mask

    focus_df = focus_df.loc[selected_mask].copy()
    if focus_df.empty:
        return focus_df

    focus_df = focus_df.sort_values(
        [
            "family_id",
            "fraction_variants_significant_positive",
            "n_variants_significant_positive",
            "mean_gain_raw",
            "pair_label",
        ],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    focus_df["focus_rank"] = focus_df.groupby("family_id").cumcount() + 1
    focus_df = focus_df.loc[
        focus_df["focus_rank"] <= int(max_pairs_per_family)
    ].copy()
    if focus_df.empty:
        return focus_df

    focus_df["metric_name"] = resolved_metric_name
    focus_df["color_hex"] = {
        "hbond_pairs": "#1b9e77",
        "saltbridge_pairs": "#d95f02",
    }[resolved_metric_name]
    focus_df["selection_rule"] = resolved_selection_rule
    focus_df["minimum_significant_fraction"] = fraction_threshold
    focus_df["minimum_significant_variant_count"] = (
        int(count_threshold) if count_threshold is not None else pd.NA
    )
    focus_df["residue_spec_a"] = focus_df.apply(
        lambda row: f"{row['target_model_id']}/{row['target_chain_id']}:{int(row['resid_a'])}",
        axis=1,
    )
    focus_df["residue_spec_b"] = focus_df.apply(
        lambda row: f"{row['target_model_id']}/{row['target_chain_id']}:{int(row['resid_b'])}",
        axis=1,
    )
    focus_df["pair_residue_spec"] = focus_df.apply(
        lambda row: (
            f"{row['target_model_id']}/{row['target_chain_id']}:"
            f"{int(row['resid_a'])},{int(row['resid_b'])}"
        ),
        axis=1,
    )
    focus_df["command_slug"] = focus_df.apply(
        lambda row: (
            "focus_"
            + normalize_md_variant_metric_column_name(str(row["metric_name"]))
            + "_"
            + normalize_md_variant_metric_column_name(str(row["family_id"]))
            + f"_pair_{int(row['focus_rank']):02d}_"
            + normalize_md_variant_metric_column_name(str(row["pair_label"]))[:60]
        ),
        axis=1,
    )

    apply_commands = []
    open_commands = []
    metric_label = {
        "hbond_pairs": "Hydrogen bonds",
        "saltbridge_pairs": "Salt bridges",
    }[resolved_metric_name]
    for row in focus_df.itertuples(index=False):
        variant_note = str(row.significant_positive_variant_labels).strip()
        if not variant_note:
            variant_note = str(row.positive_variant_labels).strip()
        command_lines = [
            "# Figure 4 WT ChimeraX gained-interaction focus",
            f"# Metric: {metric_label}",
            f"# Family: {row.family_id}",
            f"# Pair: {row.pair_label}",
            (
                "# Selection: significant gain in the family | "
                f"rule={resolved_selection_rule} | "
                f"fraction={float(row.fraction_variants_significant_positive):0.3g} | "
                f"count={int(row.n_variants_significant_positive)}/{int(row.n_variants_tested)} | "
                f"FDR alpha={float(row.alpha_fdr):0.3g}"
            ),
            f"# Significant positive variants: {variant_note if variant_note else 'none listed'}",
            f"color {row.target_model_id} {neutral_color}",
            f"show {row.pair_residue_spec} atoms",
            f"style {row.pair_residue_spec} {resolved_emphasis_style}",
            f"color {row.pair_residue_spec} {row.color_hex}",
        ]
        apply_command = "\n".join(command_lines)
        open_command_lines = list(command_lines)
        insert_index = 6
        if str(target_layout).strip().lower() == "combined_model":
            if combined_pdb_path is not None:
                open_command_lines.insert(
                    insert_index,
                    f"open {json.dumps(str(Path(combined_pdb_path).resolve()))}",
                )
        else:
            open_command_lines.insert(
                insert_index,
                f"open {json.dumps(str(Path(row.aligned_pdb_path).resolve()))}",
            )
        apply_commands.append(apply_command)
        open_commands.append("\n".join(open_command_lines))

    focus_df["apply_command"] = apply_commands
    focus_df["open_command"] = open_commands
    return focus_df.sort_values(
        ["family_id", "focus_rank", "pair_label"]
    ).reset_index(drop=True)


def _resolve_variant_chimerax_structure_path(
    structure_root: Path,
    variant_system: str,
    metric_name: str,
) -> Path | None:
    structure_root = Path(structure_root)
    if not structure_root.exists():
        return None
    exact_candidates = list(structure_root.rglob(f"{variant_system}.pdb"))
    if exact_candidates:
        return sorted(exact_candidates, key=lambda path: path.as_posix())[0].resolve()
    candidates = list(structure_root.rglob(f"{variant_system}_*.pdb"))
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda path: (
            str(metric_name) not in path.name,
            "signed" not in path.name,
            "validation" in path.as_posix(),
            len(path.as_posix()),
            path.as_posix(),
        ),
    )[0].resolve()


def _read_pdb_residue_heavy_sidechain_atoms(
    pdb_path: Path,
    residue_number: int,
) -> tuple[str, list[tuple[str, np.ndarray]]]:
    backbone_atom_names = {"N", "CA", "C", "O", "OXT"}
    chain_id = "A"
    sidechain_atoms: list[tuple[str, np.ndarray]] = []
    ca_atom: tuple[str, np.ndarray] | None = None
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                line_residue_number = int(line[22:26])
            except ValueError:
                continue
            if line_residue_number != int(residue_number):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() or atom_name[:1].upper()
            if element == "H":
                continue
            atom_xyz = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
            chain_id = line[21].strip() or chain_id
            if atom_name == "CA":
                ca_atom = (atom_name, atom_xyz)
            if atom_name not in backbone_atom_names:
                sidechain_atoms.append((atom_name, atom_xyz))
    if not sidechain_atoms and ca_atom is not None:
        sidechain_atoms = [ca_atom]
    return chain_id, sidechain_atoms


def _find_closest_variant_interaction_atoms(
    pdb_path: Path,
    residue_number_a: int,
    residue_number_b: int,
) -> tuple[str, str, str, float] | None:
    chain_id_a, atoms_a = _read_pdb_residue_heavy_sidechain_atoms(
        pdb_path=pdb_path,
        residue_number=residue_number_a,
    )
    chain_id_b, atoms_b = _read_pdb_residue_heavy_sidechain_atoms(
        pdb_path=pdb_path,
        residue_number=residue_number_b,
    )
    if not atoms_a or not atoms_b:
        return None
    atom_name_a, atom_xyz_a, atom_name_b, atom_xyz_b = min(
        (
            (name_a, xyz_a, name_b, xyz_b)
            for name_a, xyz_a in atoms_a
            for name_b, xyz_b in atoms_b
        ),
        key=lambda values: float(np.linalg.norm(values[1] - values[3])),
    )
    distance_angstrom = float(np.linalg.norm(atom_xyz_a - atom_xyz_b))
    chain_id = chain_id_a if chain_id_a == chain_id_b else chain_id_a
    return chain_id, atom_name_a, atom_name_b, distance_angstrom


def build_variant_chimerax_interaction_closeup_table(
    md_variant_bundle: MDVariantAnalysisBundle,
    metric_name: str,
    variant_structure_root: Path,
    significance_alpha: float = 0.05,
    minimum_significant_fraction: float | None = 0.5,
    minimum_significant_variant_count: int | None = 2,
    selection_rule: str = "fraction_or_count",
    max_pairs_per_family: int = 2,
    substituted_interaction_color: str = "#CC79A7",
    original_interaction_color: str = "#0072B2",
    neutral_color: str = "#c1c1c1",
    backbone_transparency_percent: float = 45.0,
    show_dashed_interaction: bool = True,
    interaction_line_color: str = "#4d4d4d",
    interaction_line_dashes: int = 8,
    interaction_line_radius: float = 0.08,
    closeup_pad_fraction: float = 0.5,
    alignment_reference_pdb_path: Path | None = None,
    aligned_structure_output_dir: Path | None = None,
    model_id_start: int = 101,
) -> pd.DataFrame:
    resolved_metric_name = _resolve_wildtype_chimerax_metric_name(metric_name)
    if resolved_metric_name not in {"hbond_pairs", "saltbridge_pairs"}:
        raise KeyError(
            "Variant ChimeraX interaction close-ups are only supported for "
            "'hbond_pairs' and 'saltbridge_pairs'."
        )
    if not 0.0 <= float(backbone_transparency_percent) <= 100.0:
        raise ValueError("backbone_transparency_percent must be between 0 and 100.")
    if int(max_pairs_per_family) <= 0:
        raise ValueError("max_pairs_per_family must be positive.")

    summary_df, variant_test_df = _build_wildtype_chimerax_pair_gain_focus_score_table(
        md_variant_bundle=md_variant_bundle,
        metric_name=resolved_metric_name,
        alpha=significance_alpha,
        return_variant_tests=True,
    )
    if summary_df.empty or variant_test_df.empty:
        return pd.DataFrame()

    fraction_mask = pd.Series(False, index=summary_df.index)
    count_mask = pd.Series(False, index=summary_df.index)
    if minimum_significant_fraction is not None:
        fraction_mask = (
            summary_df["fraction_variants_significant_positive"]
            >= float(minimum_significant_fraction)
        )
    if minimum_significant_variant_count is not None:
        count_mask = (
            summary_df["n_variants_significant_positive"]
            >= int(minimum_significant_variant_count)
        )
    resolved_selection_rule = str(selection_rule).strip().lower()
    selection_masks = {
        "fraction_or_count": fraction_mask | count_mask,
        "fraction_and_count": fraction_mask & count_mask,
        "fraction_only": fraction_mask,
        "count_only": count_mask,
    }
    if resolved_selection_rule not in selection_masks:
        raise ValueError(
            "Unsupported selection_rule. Expected 'fraction_or_count', "
            "'fraction_and_count', 'fraction_only', or 'count_only'."
        )

    selected_pair_df = summary_df.loc[
        summary_df["dominant_direction"].eq("positive")
        & selection_masks[resolved_selection_rule]
    ].copy()
    selected_pair_df = selected_pair_df.sort_values(
        [
            "family_id",
            "fraction_variants_significant_positive",
            "n_variants_significant_positive",
            "mean_gain_raw",
            "pair_label",
        ],
        ascending=[True, False, False, False, True],
    )
    selected_pair_df["pair_family_rank"] = selected_pair_df.groupby("family_id").cumcount() + 1
    selected_pair_df = selected_pair_df.loc[
        selected_pair_df["pair_family_rank"] <= int(max_pairs_per_family)
    ].copy()
    if selected_pair_df.empty:
        return pd.DataFrame()

    candidate_df = variant_test_df.loc[variant_test_df["significant_positive"]].merge(
        selected_pair_df,
        on=[
            "family_id",
            "metric_name",
            "metric_label",
            "pair_key",
            "pair_label",
            "resid_a",
            "resname_a",
            "resid_b",
            "resname_b",
        ],
        how="inner",
        suffixes=("_variant", "_family"),
        validate="many_to_one",
    )
    if candidate_df.empty:
        return pd.DataFrame()

    structure_path_cache: dict[str, Path | None] = {}
    for variant_system in candidate_df["variant_system"].astype(str).unique():
        structure_path_cache[variant_system] = _resolve_variant_chimerax_structure_path(
            structure_root=variant_structure_root,
            variant_system=variant_system,
            metric_name=resolved_metric_name,
        )
    candidate_df["variant_structure_path"] = candidate_df["variant_system"].map(
        structure_path_cache
    )
    candidate_df = candidate_df.loc[candidate_df["variant_structure_path"].notna()].copy()
    if candidate_df.empty:
        return pd.DataFrame()

    candidate_df = candidate_df.sort_values(
        [
            "family_id",
            "pair_family_rank",
            "mean_difference",
            "variant_mean",
            "p_value_fdr",
            "variant_system",
        ],
        ascending=[True, True, False, False, True, True],
    )
    closeup_df = candidate_df.groupby(
        ["family_id", "pair_family_rank"],
        as_index=False,
        sort=False,
    ).head(1).copy()

    metadata_lookup_df = md_variant_bundle.metadata_df.set_index("system", drop=False)
    closeup_rows = []
    alignment_cache: dict[str, dict[str, object]] = {}
    reference_trace = (
        parse_ca_trace(Path(alignment_reference_pdb_path))
        if alignment_reference_pdb_path is not None
        else None
    )
    if reference_trace is not None and aligned_structure_output_dir is None:
        raise ValueError(
            "aligned_structure_output_dir is required when alignment_reference_pdb_path is set."
        )
    metric_label = {
        "hbond_pairs": "Hydrogen bond",
        "saltbridge_pairs": "Salt bridge",
    }[resolved_metric_name]
    for closeup_index, row in enumerate(closeup_df.itertuples(index=False)):
        metadata_row = metadata_lookup_df.loc[str(row.variant_system)]
        mutation_resids = set(int(value) for value in metadata_row["mutation_resids_list"])
        substituted_resids = sorted(
            {int(row.resid_a), int(row.resid_b)}.intersection(mutation_resids)
        )
        original_resids = sorted(
            {int(row.resid_a), int(row.resid_b)}.difference(mutation_resids)
        )
        source_structure_path = Path(row.variant_structure_path)
        structure_path = source_structure_path
        alignment_summary = {
            "alignment_reference_pdb_path": "",
            "alignment_method": "none",
            "alignment_aligned_length": pd.NA,
            "alignment_rmsd_angstrom": np.nan,
            "alignment_tm_score_to_reference": np.nan,
        }
        if reference_trace is not None:
            variant_system = str(row.variant_system)
            if variant_system not in alignment_cache:
                variant_trace = parse_ca_trace(source_structure_path)
                alignment_result = compute_ca_superposition_alignment(
                    structure_a_trace=reference_trace,
                    structure_b_trace=variant_trace,
                )
                aligned_output_dir = Path(aligned_structure_output_dir)
                aligned_path = (
                    aligned_output_dir
                    / f"{variant_system}_aligned_to_{Path(alignment_reference_pdb_path).stem}.pdb"
                )
                _write_transformed_pdb(
                    template_pdb_path=source_structure_path,
                    output_path=aligned_path,
                    rotation_matrix=np.asarray(alignment_result["rotation_matrix"], dtype=float),
                    mobile_center=np.asarray(alignment_result["mobile_center"], dtype=float),
                    reference_center=np.asarray(alignment_result["reference_center"], dtype=float),
                )
                alignment_cache[variant_system] = {
                    "aligned_path": aligned_path.resolve(),
                    "alignment_reference_pdb_path": str(
                        Path(alignment_reference_pdb_path).resolve()
                    ),
                    "alignment_method": "ca_superposition",
                    "alignment_aligned_length": int(alignment_result["aligned_length"]),
                    "alignment_rmsd_angstrom": float(alignment_result["rmsd_angstrom"]),
                    "alignment_tm_score_to_reference": float(
                        alignment_result["tm_score_to_structure_a"]
                    ),
                }
            cached_alignment = alignment_cache[variant_system]
            structure_path = Path(cached_alignment["aligned_path"])
            alignment_summary = {
                key: cached_alignment[key]
                for key in (
                    "alignment_reference_pdb_path",
                    "alignment_method",
                    "alignment_aligned_length",
                    "alignment_rmsd_angstrom",
                    "alignment_tm_score_to_reference",
                )
            }
        closest_atoms = _find_closest_variant_interaction_atoms(
            pdb_path=structure_path,
            residue_number_a=int(row.resid_a),
            residue_number_b=int(row.resid_b),
        )
        chain_id = closest_atoms[0] if closest_atoms is not None else "A"
        model_id = f"#{int(model_id_start) + closeup_index}"
        pair_spec = f"{model_id}/{chain_id}:{int(row.resid_a)},{int(row.resid_b)}"
        command_lines = [
            "# Figure 4 statistically enriched variant-interaction close-up",
            f"# Family: {row.family_id}",
            f"# Variant: {row.variant_label}",
            f"# Metric and pair: {metric_label} | {row.pair_label}",
            (
                "# Variant vs WT: "
                f"occupancy={float(row.variant_mean):0.4g} vs {float(row.wildtype_mean):0.4g} | "
                f"gain={float(row.mean_difference):0.4g} | "
                f"q={float(row.p_value_fdr):0.3g}"
            ),
            (
                "# Family frequency: "
                f"{int(row.n_variants_significant_positive)}/{int(row.n_variants_tested)} "
                f"({float(row.fraction_variants_significant_positive):0.3g}) significant gains"
            ),
            "# Dashed connector joins the closest heavy sidechain atoms in this static structure; it is illustrative, not an MD-frame contact measurement.",
            f"# Common-frame alignment: {alignment_summary['alignment_method']} | model id: {model_id}",
            f"open {json.dumps(str(structure_path.resolve()))} id {model_id}",
            f"hide {model_id} atoms",
            f"cartoon {model_id}",
            f"color {model_id} {to_hex(neutral_color)} target c",
            f"transparency {model_id} {float(backbone_transparency_percent):0.3g} target c",
            f"show {pair_spec} & sidechain atoms",
            f"style {pair_spec} & sidechain stick",
        ]
        if original_resids:
            original_spec = f"{model_id}/{chain_id}:" + ",".join(map(str, original_resids))
            command_lines.extend(
                [
                    f"color {original_spec} & sidechain {to_hex(original_interaction_color)} target a",
                    f"transparency {original_spec} & sidechain 0 target a",
                ]
            )
        if substituted_resids:
            substituted_spec = f"{model_id}/{chain_id}:" + ",".join(map(str, substituted_resids))
            command_lines.extend(
                [
                    f"color {substituted_spec} & sidechain {to_hex(substituted_interaction_color)} target a",
                    f"transparency {substituted_spec} & sidechain 0 target a",
                ]
            )

        interaction_atom_spec_a = ""
        interaction_atom_spec_b = ""
        interaction_distance_angstrom = np.nan
        if bool(show_dashed_interaction) and closest_atoms is not None:
            _, atom_name_a, atom_name_b, interaction_distance_angstrom = closest_atoms
            interaction_atom_spec_a = f"{model_id}/{chain_id}:{int(row.resid_a)}@{atom_name_a}"
            interaction_atom_spec_b = f"{model_id}/{chain_id}:{int(row.resid_b)}@{atom_name_b}"
            command_lines.append(
                f"distance {interaction_atom_spec_a} {interaction_atom_spec_b} "
                f"color {to_hex(interaction_line_color)} dashes {int(interaction_line_dashes)} "
                f"radius {float(interaction_line_radius):0.3g} symbol false"
            )
        command_lines.append(f"view {pair_spec} pad {float(closeup_pad_fraction):0.3g}")

        command_slug = (
            "variant_focus_"
            + normalize_md_variant_metric_column_name(resolved_metric_name)
            + "_"
            + normalize_md_variant_metric_column_name(str(row.family_id))
            + f"_pair_{int(row.pair_family_rank):02d}_"
            + normalize_md_variant_metric_column_name(str(row.variant_system))[:50]
        )
        closeup_rows.append(
            {
                **row._asdict(),
                "source_variant_structure_path": source_structure_path.resolve(),
                "aligned_variant_structure_path": structure_path.resolve(),
                "target_model_id": model_id,
                **alignment_summary,
                "mutation_labels": str(metadata_row.get("mutation_labels", "")),
                "substituted_interaction_resids": " ".join(map(str, substituted_resids)),
                "original_interaction_resids": " ".join(map(str, original_resids)),
                "chain_id": chain_id,
                "interaction_atom_spec_a": interaction_atom_spec_a,
                "interaction_atom_spec_b": interaction_atom_spec_b,
                "static_interaction_distance_angstrom": interaction_distance_angstrom,
                "command_slug": command_slug,
                "open_command": "\n".join(command_lines),
            }
        )
    return pd.DataFrame(closeup_rows).sort_values(
        ["family_id", "pair_family_rank", "metric_name"]
    ).reset_index(drop=True)


def build_md_variant_analysis_bundle(
    metric_dir: Path,
    mutation_metadata_path: Path,
    metric_window: str | int | None = "latest",
    tm_summary_df: pd.DataFrame | None = None,
    label_lookup_path: Path | None = None,
) -> MDVariantAnalysisBundle:
    required_metrics = [
        "rmsf",
        "residue_sasa",
        "secondary_structure_counts",
        "secondary_structure_residue",
        "mutation_neighborhood",
        *MD_VARIANT_INTERACTION_COUNT_METRICS,
        *MD_VARIANT_PAIR_METRICS,
    ]
    metric_window_suffix = resolve_md_variant_metric_window_suffix(
        metric_dir=metric_dir,
        required_metrics=required_metrics,
        metric_window=metric_window,
    )
    metric_window_start, metric_window_stop = resolve_md_variant_metric_window_bounds(
        metric_dir=metric_dir,
        metric_names=required_metrics,
        metric_window_suffix=metric_window_suffix,
    )

    metadata_df = load_md_variant_metadata(
        mutation_metadata_path=mutation_metadata_path,
    )
    metadata_df = annotate_md_variant_metadata_with_tm(
        metadata_df=metadata_df,
        tm_summary_df=tm_summary_df,
    )
    resolved_label_lookup_path = (
        label_lookup_path
        if label_lookup_path is not None
        else mutation_metadata_path.parent / "md_variant_label_lookup.csv"
    )
    if resolved_label_lookup_path.exists():
        metadata_df = annotate_md_variant_metadata_with_label_lookup(
            metadata_df=metadata_df,
            label_lookup_df=load_md_variant_label_lookup(resolved_label_lookup_path),
        )
    family_ids = order_md_variant_families(metadata_df)

    rmsf_raw_df = read_md_variant_metric_table(
        metric_dir, "rmsf", metric_window_suffix=metric_window_suffix
    )
    sasa_raw_df = read_md_variant_metric_table(
        metric_dir, "residue_sasa", metric_window_suffix=metric_window_suffix
    )
    ss_counts_raw_df = read_md_variant_metric_table(
        metric_dir,
        "secondary_structure_counts",
        metric_window_suffix=metric_window_suffix,
    )
    ss_residue_raw_df = read_md_variant_metric_table(
        metric_dir,
        "secondary_structure_residue",
        metric_window_suffix=metric_window_suffix,
    )
    mutation_neighborhood_df = read_md_variant_metric_table(
        metric_dir,
        "mutation_neighborhood",
        metric_window_suffix=metric_window_suffix,
    )
    interaction_count_raw_by_metric = {
        metric_name: read_md_variant_metric_table(
            metric_dir,
            metric_name,
            metric_window_suffix=metric_window_suffix,
        )
        for metric_name in MD_VARIANT_INTERACTION_COUNT_METRICS
    }
    rmsf_raw_df = backfill_md_variant_rmsf_resnames(
        rmsf_df=rmsf_raw_df,
        residue_reference_df=sasa_raw_df,
    )
    pair_raw_by_metric = {
        metric_name: prepare_md_variant_pair_metric(
            read_md_variant_metric_table(
                metric_dir,
                metric_name,
                metric_window_suffix=metric_window_suffix,
            ),
            metric_name=metric_name,
        )
        for metric_name in MD_VARIANT_PAIR_METRICS
    }
    replicate_count_by_system = compute_md_variant_replicate_counts(rmsf_raw_df)

    local_residue_sets = load_md_variant_local_residue_sets(
        metadata_df=metadata_df,
        rmsf_raw_df=rmsf_raw_df,
        mutation_neighborhood_df=mutation_neighborhood_df,
    )

    ss_residue_raw_df["ordered_occupancy"] = (
        ss_residue_raw_df["helix_occupancy"] + ss_residue_raw_df["strand_occupancy"]
    )

    rmsf_summary_df = aggregate_md_variant_residue_metric(rmsf_raw_df, ["rmsf"])
    sasa_summary_df = aggregate_md_variant_residue_metric(sasa_raw_df, ["mean_sasa"])
    global_sasa_replicate_df = compute_md_variant_global_sasa_replicate_table(sasa_raw_df).merge(
        metadata_df[
            [
                "system",
                "family",
                "is_wildtype",
                "mutation_count",
                "short_label",
                "tm_display_label",
            ]
        ],
        on="system",
        how="left",
        validate="many_to_one",
    )
    ss_residue_summary_df = aggregate_md_variant_residue_metric(
        ss_residue_raw_df,
        ["helix_occupancy", "strand_occupancy", "loop_occupancy", "ordered_occupancy"],
    )
    ss_counts_summary_df = aggregate_md_variant_secondary_structure_counts(ss_counts_raw_df)
    interaction_count_summary_by_metric = {
        metric_name: aggregate_md_variant_interaction_count_metric(
            metric_df,
            value_column=MD_VARIANT_INTERACTION_COUNT_COLUMN_MAP[metric_name],
        )
        for metric_name, metric_df in interaction_count_raw_by_metric.items()
    }
    interaction_count_replicate_by_metric = {
        metric_name: compute_md_variant_interaction_count_replicate_table(
            metric_df,
            value_column=MD_VARIANT_INTERACTION_COUNT_COLUMN_MAP[metric_name],
        ).merge(
            metadata_df[
                [
                    "system",
                    "family",
                    "is_wildtype",
                    "mutation_count",
                    "short_label",
                    "tm_display_label",
                ]
            ],
            on="system",
            how="left",
            validate="many_to_one",
        )
        for metric_name, metric_df in interaction_count_raw_by_metric.items()
    }
    interaction_count_block_by_metric = {
        metric_name: compute_md_variant_interaction_count_block_table(
            metric_df,
            value_column=MD_VARIANT_INTERACTION_COUNT_COLUMN_MAP[metric_name],
        ).merge(
            metadata_df[
                [
                    "system",
                    "family",
                    "is_wildtype",
                    "mutation_count",
                    "short_label",
                    "tm_display_label",
                ]
            ],
            on="system",
            how="left",
            validate="many_to_one",
        )
        for metric_name, metric_df in interaction_count_raw_by_metric.items()
    }
    pair_summary_by_metric = {
        metric_name: aggregate_md_variant_pair_metric(
            metric_df,
            replicate_count_by_system=replicate_count_by_system,
            )
        for metric_name, metric_df in pair_raw_by_metric.items()
    }
    pair_replicate_by_metric = {
        metric_name: compute_md_variant_pair_replicate_table(metric_df).merge(
            metadata_df[
                [
                    "system",
                    "family",
                    "is_wildtype",
                    "mutation_count",
                    "short_label",
                    "tm_display_label",
                ]
            ],
            on="system",
            how="left",
            validate="many_to_one",
        )
        for metric_name, metric_df in pair_raw_by_metric.items()
    }
    pair_residue_replicate_by_metric = {
        metric_name: compute_md_variant_pair_residue_replicate_table(metric_df).merge(
            metadata_df[
                [
                    "system",
                    "family",
                    "is_wildtype",
                    "mutation_count",
                    "short_label",
                    "tm_display_label",
                ]
            ],
            on="system",
            how="left",
            validate="many_to_one",
        )
        for metric_name, metric_df in pair_raw_by_metric.items()
    }

    rmsf_delta_tables = []
    sasa_delta_tables = []
    ss_delta_tables = []
    interaction_count_delta_tables_by_metric = {
        metric_name: []
        for metric_name in MD_VARIANT_INTERACTION_COUNT_METRICS
    }
    pair_delta_tables_by_metric = {
        metric_name: []
        for metric_name in MD_VARIANT_PAIR_METRICS
    }
    for family_id in family_ids:
        rmsf_delta_tables.append(
            compute_md_variant_residue_delta_table(
                summary_df=rmsf_summary_df,
                metadata_df=metadata_df,
                family_id=family_id,
                value_columns=["rmsf"],
            )
        )
        sasa_delta_tables.append(
            compute_md_variant_residue_delta_table(
                summary_df=sasa_summary_df,
                metadata_df=metadata_df,
                family_id=family_id,
                value_columns=["mean_sasa"],
            )
        )
        ss_delta_tables.append(
            compute_md_variant_residue_delta_table(
                summary_df=ss_residue_summary_df,
                metadata_df=metadata_df,
                family_id=family_id,
                value_columns=[
                    "helix_occupancy",
                    "strand_occupancy",
                    "loop_occupancy",
                    "ordered_occupancy",
                ],
            )
        )
        for metric_name, interaction_count_summary_df in interaction_count_summary_by_metric.items():
            interaction_count_delta_tables_by_metric[metric_name].append(
                compute_md_variant_frame_delta_table(
                    summary_df=interaction_count_summary_df,
                    metadata_df=metadata_df,
                    family_id=family_id,
                )
            )
        for metric_name, pair_summary_df in pair_summary_by_metric.items():
            pair_delta_tables_by_metric[metric_name].append(
                compute_md_variant_pair_delta_table(
                    summary_df=pair_summary_df,
                    metadata_df=metadata_df,
                    family_id=family_id,
                    local_residue_sets=local_residue_sets,
                )
            )

    rmsf_delta_df = pd.concat(rmsf_delta_tables, ignore_index=True)
    sasa_delta_df = pd.concat(sasa_delta_tables, ignore_index=True)
    ss_residue_delta_df = pd.concat(ss_delta_tables, ignore_index=True)
    interaction_count_delta_by_metric = {
        metric_name: pd.concat(metric_tables, ignore_index=True)
        for metric_name, metric_tables in interaction_count_delta_tables_by_metric.items()
    }
    pair_delta_by_metric = {
        metric_name: pd.concat(metric_tables, ignore_index=True)
        for metric_name, metric_tables in pair_delta_tables_by_metric.items()
    }

    variant_signal_df = compute_md_variant_signal_table(
        metadata_df=metadata_df,
        local_residue_sets=local_residue_sets,
        rmsf_delta_df=rmsf_delta_df,
        sasa_delta_df=sasa_delta_df,
        ss_residue_delta_df=ss_residue_delta_df,
        ss_counts_summary_df=ss_counts_summary_df,
        pair_delta_by_metric=pair_delta_by_metric,
    )
    family_pattern_summary_df = build_md_variant_family_pattern_summary(
        variant_signal_df=variant_signal_df
    )

    return MDVariantAnalysisBundle(
        metric_window_suffix=metric_window_suffix,
        metric_window_start=metric_window_start,
        metric_window_stop=metric_window_stop,
        metadata_df=metadata_df,
        replicate_count_by_system=replicate_count_by_system,
        local_residue_sets=local_residue_sets,
        family_ids=family_ids,
        rmsf_summary_df=rmsf_summary_df,
        sasa_summary_df=sasa_summary_df,
        ss_counts_summary_df=ss_counts_summary_df,
        ss_residue_summary_df=ss_residue_summary_df,
        interaction_count_summary_by_metric=interaction_count_summary_by_metric,
        interaction_count_replicate_by_metric=interaction_count_replicate_by_metric,
        interaction_count_block_by_metric=interaction_count_block_by_metric,
        global_sasa_replicate_df=global_sasa_replicate_df,
        pair_replicate_by_metric=pair_replicate_by_metric,
        pair_residue_replicate_by_metric=pair_residue_replicate_by_metric,
        pair_summary_by_metric=pair_summary_by_metric,
        rmsf_delta_df=rmsf_delta_df,
        sasa_delta_df=sasa_delta_df,
        ss_residue_delta_df=ss_residue_delta_df,
        interaction_count_delta_by_metric=interaction_count_delta_by_metric,
        pair_delta_by_metric=pair_delta_by_metric,
        variant_signal_df=variant_signal_df,
        family_pattern_summary_df=family_pattern_summary_df,
    )


def build_md_variant_local_signal_plot_table(
    variant_signal_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    family_id: str,
) -> pd.DataFrame:
    family_df = variant_signal_df.loc[variant_signal_df["family"].eq(family_id)].copy()
    family_df["variant_system"] = pd.Categorical(
        family_df["variant_system"],
        categories=compute_md_variant_order(metadata_df, family_id),
        ordered=True,
    )
    table_df = family_df[
        [
            "variant_system",
            "variant_label",
            "delta_rmsf_neighborhood_mean",
            "delta_sasa_neighborhood_mean",
            "delta_ordered_ss_neighborhood_mean",
            "contact_gain_local",
            "saltbridge_gain_local",
            "hbond_gain_local",
            "contact_gain_local_fraction",
            "hbond_gain_local_fraction",
        ]
    ].copy()
    table_df = table_df.sort_values("variant_system")
    zscore_columns = [
        column_name
        for column_name in table_df.columns
        if column_name not in {"variant_system", "variant_label"}
    ]
    for column_name in zscore_columns:
        values = table_df[column_name].astype(float)
        if float(values.std(ddof=0)) > 0:
            table_df[column_name] = (values - values.mean()) / values.std(ddof=0)
        else:
            table_df[column_name] = 0.0
    return table_df


def plot_md_family_rmsf_panels(
    family_id: str,
    metadata_df: pd.DataFrame,
    rmsf_summary_df: pd.DataFrame,
):
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[family_metadata_df["is_wildtype"], "system"].iat[0]
    wildtype_df = rmsf_summary_df.loc[rmsf_summary_df["system"].eq(wildtype_system)].sort_values("resid").copy()
    variant_order = compute_md_variant_order(metadata_df, family_id)
    color_map = build_md_variant_family_color_map(metadata_df, family_id)

    n_variants = len(variant_order)
    n_columns = 2 if n_variants > 1 else 1
    n_rows = int(np.ceil(n_variants / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, max(4.0, 3.5 * n_rows + 1.1)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    figure.set_constrained_layout_pads(hspace=0.08, h_pad=0.08, w_pad=0.04)
    axes = np.atleast_1d(axes).ravel()

    y_max = max(
        float((wildtype_df["rmsf_mean"] + wildtype_df["rmsf_std"]).max()),
        float(
            rmsf_summary_df.loc[
                rmsf_summary_df["system"].isin(variant_order),
                "rmsf_mean",
            ].max()
        ),
    )
    residue_ticks = choose_residue_ticks(wildtype_df["resid"].to_numpy(dtype=int))

    for axis, variant_system in zip(axes, variant_order):
        variant_df = rmsf_summary_df.loc[
            rmsf_summary_df["system"].eq(variant_system)
        ].sort_values("resid").copy()
        variant_metadata = family_metadata_df.loc[
            family_metadata_df["system"].eq(variant_system)
        ].iloc[0]
        mutation_resids = list(variant_metadata["mutation_resids_list"])

        axis.fill_between(
            wildtype_df["resid"],
            wildtype_df["rmsf_mean"] - wildtype_df["rmsf_std"],
            wildtype_df["rmsf_mean"] + wildtype_df["rmsf_std"],
            color="#bdbdbd",
            alpha=0.35,
            label="WT replicate SD",
        )
        axis.plot(
            wildtype_df["resid"],
            wildtype_df["rmsf_mean"],
            color="#222222",
            linewidth=2.0,
            label="WT mean",
        )
        axis.plot(
            variant_df["resid"],
            variant_df["rmsf_mean"],
            color=color_map[variant_system],
            linewidth=1.8,
            label="Variant mean",
        )
        axis.fill_between(
            variant_df["resid"],
            variant_df["rmsf_mean"] - variant_df["rmsf_std"],
            variant_df["rmsf_mean"] + variant_df["rmsf_std"],
            color=color_map[variant_system],
            alpha=0.15,
        )

        if mutation_resids:
            axis.scatter(
                mutation_resids,
                np.full(len(mutation_resids), -0.06 * y_max),
                marker="v",
                s=18,
                color=color_map[variant_system],
                edgecolors="none",
                clip_on=False,
                zorder=4,
            )

        axis.set_ylim(-0.08 * y_max, 1.05 * y_max)
        axis.set_xticks(residue_ticks)
        axis.set_title(
            build_md_family_rmsf_panel_title(variant_metadata),
            fontsize=9,
            pad=10,
        )
        axis.grid(alpha=0.18, linewidth=0.5)

    for axis in axes[n_variants:]:
        axis.set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color="#222222", linewidth=2.0, label="WT mean"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.35, label="WT replicate SD"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.8, label="Variant mean"),
        plt.Line2D([0], [0], marker="v", linestyle="None", color="#1f77b4", markersize=6, label="Mutation site"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        f"{family_id}: per-variant RMSF vs WT\nNegative local shifts in the companion heatmap indicate reduced flexibility relative to WT.",
        fontsize=13,
        y=1.05,
    )
    figure.supxlabel("Residue number")
    figure.supylabel("RMSF")
    return figure


def plot_md_family_delta_heatmap(
    family_id: str,
    metadata_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    value_column: str,
    title: str,
    colorbar_label: str,
    vlim: float | None = None,
    figure_width_mm: float = 381.0,
    figure_row_height_mm: float = 10.7,
    figure_base_height_mm: float = 45.7,
    colorbar_height_mm: float = 28.0,
    colorbar_width_mm: float = 4.0,
    colorbar_pad_mm: float = 4.0,
    y_tick_fontfamily: str = "DejaVu Sans Mono",
    y_tick_extra_columns: int = 1,
    show_y_tick_tm_metadata: bool = True,
    title_size_pt: float = 5.5,
    label_size_pt: float = 5.5,
    tick_size_pt: float = 5.5,
):
    variant_order = compute_md_variant_order(metadata_df, family_id)
    family_delta_df = delta_df.loc[delta_df["family"].eq(family_id)].copy()
    family_delta_df["variant_system"] = pd.Categorical(
        family_delta_df["variant_system"],
        categories=variant_order,
        ordered=True,
    )
    family_delta_df = family_delta_df.sort_values(["variant_system", "resid"])
    pivot_df = family_delta_df.pivot(
        index="variant_system",
        columns="resid",
        values=value_column,
    ).loc[variant_order]

    residue_numbers = pivot_df.columns.to_numpy(dtype=int)
    tick_values = choose_residue_ticks(residue_numbers)
    tick_positions = [
        int(np.where(residue_numbers == tick_value)[0][0])
        for tick_value in tick_values
        if tick_value in residue_numbers
    ]
    label_getter = (
        get_md_variant_display_label
        if show_y_tick_tm_metadata
        else get_md_variant_base_label
    )
    row_labels = [label_getter(metadata_df, system_name) for system_name in pivot_df.index]
    formatted_row_labels, row_label_columns = format_fixed_width_monospace_labels(
        row_labels,
        extra_columns=y_tick_extra_columns,
    )
    max_abs_value = float(np.nanmax(np.abs(pivot_df.to_numpy(dtype=float))))
    heatmap_limit = vlim if vlim is not None else max(max_abs_value, 1e-6)
    heatmap_limit = 4
    figure_width_in, figure_row_height_in, figure_base_height_in = mm_to_inches(
        figure_width_mm,
        figure_row_height_mm,
        figure_base_height_mm,
    )
    figure_height_in = figure_row_height_in * len(variant_order) + figure_base_height_in
    colorbar_width_in, colorbar_height_in, colorbar_pad_in = mm_to_inches(
        colorbar_width_mm,
        colorbar_height_mm,
        colorbar_pad_mm,
    )
    estimated_label_width_in = estimate_monospace_label_width_inches(
        row_label_columns,
        tick_size_pt,
    )
    left_margin = min(max((estimated_label_width_in + 0.35) / figure_width_in, 0.18), 0.42)
    right_margin = max(left_margin + 0.20, 1.0 - ((colorbar_width_in + colorbar_pad_in + 0.35) / figure_width_in))
    figure, axis = plt.subplots(
        figsize=(figure_width_in, figure_height_in),
        constrained_layout=False,
    )
    figure.subplots_adjust(left=left_margin, right=right_margin, top=0.83, bottom=0.16)
    image = axis.imshow(
        pivot_df.to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-heatmap_limit, vcenter=0.0, vmax=heatmap_limit),
    )
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(
        formatted_row_labels,
        fontfamily=y_tick_fontfamily,
        fontsize=tick_size_pt,
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_values, fontsize=tick_size_pt)
    axis.set_xlabel("Residue number", fontsize=label_size_pt)
    axis.set_ylabel("Variant", fontsize=label_size_pt)
    axis.set_title(title, fontsize=title_size_pt, pad=10)
    axis.tick_params(axis="y", pad=6)

    for row_index, variant_system in enumerate(variant_order):
        mutation_resids = metadata_df.loc[
            metadata_df["system"].eq(variant_system),
            "mutation_resids_list",
        ].iat[0]
        mutation_positions = [
            int(np.where(residue_numbers == residue_number)[0][0])
            for residue_number in mutation_resids
            if residue_number in residue_numbers
        ]
        if mutation_positions:
            axis.scatter(
                mutation_positions,
                np.full(len(mutation_positions), row_index),
                marker="s",
                s=12,
                color="black",
                linewidths=0,
            )

    axis_position = axis.get_position()
    cbar_height_fraction = min(colorbar_height_in / figure_height_in, axis_position.height)
    cbar_width_fraction = colorbar_width_in / figure_width_in
    cbar_pad_fraction = colorbar_pad_in / figure_width_in
    cbar_x0 = axis_position.x1 + cbar_pad_fraction
    cbar_y0 = axis_position.y0 + max((axis_position.height - cbar_height_fraction) / 2.0, 0.0)
    cbar_axis = figure.add_axes(
        [
            cbar_x0,
            cbar_y0,
            cbar_width_fraction,
            cbar_height_fraction,
        ]
    )
    colorbar = figure.colorbar(image, cax=cbar_axis)
    colorbar.set_label(colorbar_label, fontsize=label_size_pt)
    colorbar.ax.tick_params(labelsize=tick_size_pt)
    return figure


def plot_md_family_pair_heatmaps(
    family_id: str,
    metadata_df: pd.DataFrame,
    pair_summary_by_metric: dict[str, pd.DataFrame],
):
    system_order = compute_md_variant_system_order(metadata_df, family_id)
    figure, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    for axis, metric_name in zip(axes, MD_VARIANT_PAIR_METRICS):
        selected_df = select_md_variant_top_variable_pairs(
            family_id=family_id,
            metadata_df=metadata_df,
            pair_summary_df=pair_summary_by_metric[metric_name],
            top_n=25,
        )
        pivot_df = selected_df.pivot(
            index="pair_label",
            columns="system",
            values="occupancy_mean",
        ).fillna(0.0)
        pivot_df = pivot_df.reindex(columns=system_order)
        row_order = (
            pivot_df.sub(pivot_df[system_order[0]], axis=0)
            .abs()
            .max(axis=1)
            .sort_values(ascending=False)
            .index.tolist()
        )
        pivot_df = pivot_df.loc[row_order]
        column_labels = [
            get_md_variant_display_label(metadata_df, system_name)
            for system_name in pivot_df.columns
        ]

        image = axis.imshow(
            pivot_df.to_numpy(dtype=float),
            aspect="auto",
            cmap=MD_VARIANT_OCCUPANCY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_xticks(np.arange(len(column_labels)))
        axis.set_xticklabels(column_labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(pivot_df.index)))
        axis.set_yticklabels(
            [wrap_md_variant_label(label, width=18) for label in pivot_df.index],
            fontsize=8,
        )
        axis.set_title(
            {
                "residue_contact_pairs": "Residue contacts",
                "saltbridge_pairs": "Salt bridges",
                "hbond_pairs": "Hydrogen bonds",
            }[metric_name]
        )
        axis.set_xlabel("System")
        if axis is axes[0]:
            axis.set_ylabel("Top variable residue pairs")

    colorbar = figure.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
    colorbar.set_label("Mean occupancy across replicates")
    figure.suptitle(
        f"{family_id}: occupancy heatmaps for the most WT-divergent interaction pairs\n"
        "Rows are ranked by the largest absolute occupancy shift to WT across variants in this family.",
        fontsize=13,
    )
    return figure


def plot_md_family_interaction_count_trajectories(
    family_id: str,
    metadata_df: pd.DataFrame,
    interaction_count_summary_by_metric: dict[str, pd.DataFrame],
    interaction_count_delta_by_metric: dict[str, pd.DataFrame],
):
    variant_order = compute_md_variant_order(metadata_df, family_id)
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    color_map = build_md_variant_family_color_map(metadata_df, family_id)

    figure, axes = plt.subplots(
        len(MD_VARIANT_INTERACTION_COUNT_METRICS),
        1,
        figsize=(14, 8.8),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    legend_handles = [
        plt.Line2D([0], [0], color="#222222", linewidth=1.2, label="WT mean (0 delta)"),
        plt.Rectangle((0, 0), 1, 1, color="#bdbdbd", alpha=0.3, label="WT replicate SD"),
    ]

    for axis_index, (axis, metric_name) in enumerate(
        zip(axes, MD_VARIANT_INTERACTION_COUNT_METRICS)
    ):
        summary_df = interaction_count_summary_by_metric[metric_name]
        delta_df = interaction_count_delta_by_metric[metric_name]
        wt_df = summary_df.loc[summary_df["system"].eq(wildtype_system)].sort_values("frame").copy()
        family_delta_df = delta_df.loc[delta_df["family"].eq(family_id)].copy()

        axis.fill_between(
            wt_df["time_ns"],
            -wt_df["count_std"],
            wt_df["count_std"],
            color="#bdbdbd",
            alpha=0.3,
            linewidth=0.0,
        )
        axis.axhline(0.0, color="#222222", linewidth=1.2)

        max_abs_delta = float(np.nanmax(np.abs(family_delta_df["delta_count"]))) if not family_delta_df.empty else 0.0
        max_abs_wt_std = float(np.nanmax(wt_df["count_std"])) if not wt_df.empty else 0.0
        y_limit = max(max_abs_delta, max_abs_wt_std, 1e-6) * 1.08

        for variant_system in variant_order:
            variant_df = family_delta_df.loc[
                family_delta_df["variant_system"].eq(variant_system)
            ].sort_values("frame").copy()
            if variant_df.empty:
                continue

            variant_label = metadata_df.loc[
                metadata_df["system"].eq(variant_system),
                "short_label",
            ].iat[0]
            axis.plot(
                variant_df["time_ns"],
                variant_df["delta_count"],
                color=color_map[variant_system],
                linewidth=1.3,
                alpha=0.95,
                label=variant_label if axis_index == 0 else None,
            )
            if axis_index == 0:
                legend_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=color_map[variant_system],
                        linewidth=1.3,
                        label=variant_label,
                    )
                )

        axis.set_ylim(-y_limit, y_limit)
        axis.set_ylabel("Delta count")
        axis.set_title(
            f"{MD_VARIANT_INTERACTION_COUNT_LABEL_MAP[metric_name]}\n"
            "Variant mean count - WT mean count per frame",
            fontsize=11,
        )
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)

    axes[-1].set_xlabel("Trajectory time (ns)")
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        f"{family_id}: interaction-count trajectories relative to WT\n"
        "Zero marks the WT mean trajectory at each timepoint; the gray band shows WT replicate SD.",
        fontsize=13,
        y=1.05,
    )
    return figure


def plot_md_family_absolute_metric_boxplots(
    family_id: str,
    metadata_df: pd.DataFrame,
    interaction_count_summary_by_metric: dict[str, pd.DataFrame],
    interaction_count_block_by_metric: dict[str, pd.DataFrame],
    interaction_count_replicate_by_metric: dict[str, pd.DataFrame],
    global_sasa_replicate_df: pd.DataFrame,
    significance_df: pd.DataFrame | None = None,
):
    system_order = compute_md_variant_system_order(metadata_df, family_id)
    family_metadata_df = metadata_df.loc[metadata_df["family"].eq(family_id)].copy()
    wildtype_system = family_metadata_df.loc[
        family_metadata_df["is_wildtype"],
        "system",
    ].iat[0]
    color_map = build_md_variant_family_color_map(metadata_df, family_id)
    if significance_df is None:
        significance_df = compute_md_variant_absolute_metric_significance_df(
            family_id=family_id,
            metadata_df=metadata_df,
            interaction_count_block_by_metric=interaction_count_block_by_metric,
            interaction_count_replicate_by_metric=interaction_count_replicate_by_metric,
            global_sasa_replicate_df=global_sasa_replicate_df,
        )

    positions = np.arange(1, len(system_order) + 1)
    system_labels = [
        "WT"
        if system_name == wildtype_system
        else get_md_variant_base_label_from_row(
            metadata_df.loc[metadata_df["system"].eq(system_name)].iloc[0]
        )
        for system_name in system_order
    ]

    figure_height = max(4.8, 0.42 * len(system_order) + 2.6)
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(18, figure_height),
        sharey=False,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes).ravel()

    plot_specs = [
        (
            "residue_contacts",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["residue_contacts"],
            interaction_count_summary_by_metric["residue_contacts"],
            "count_mean",
            "Replicate-mean count",
        ),
        (
            "hbond_counts",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["hbond_counts"],
            interaction_count_summary_by_metric["hbond_counts"],
            "count_mean",
            "Replicate-mean count",
        ),
        (
            "saltbridge",
            MD_VARIANT_INTERACTION_COUNT_LABEL_MAP["saltbridge"],
            interaction_count_summary_by_metric["saltbridge"],
            "count_mean",
            "Replicate-mean count",
        ),
        (
            "global_sasa",
            "Global SASA",
            global_sasa_replicate_df.loc[
                global_sasa_replicate_df["family"].eq(family_id)
            ].copy(),
            "global_mean_sasa",
            "Replicate-mean SASA",
        ),
    ]

    for axis_index, (axis, metric_key, title, metric_df, value_column, xlabel) in enumerate(
        (axis, *spec) for axis, spec in zip(axes, plot_specs)
    ):
        values_by_system = []
        for system_name in system_order:
            system_values = (
                metric_df.loc[metric_df["system"].eq(system_name), value_column]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            if system_values.size == 0:
                system_values = np.array([np.nan], dtype=float)
            values_by_system.append(system_values)

        boxplot = axis.boxplot(
            values_by_system,
            positions=positions,
            vert=False,
            patch_artist=True,
            widths=0.68,
            tick_labels=system_labels,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.35},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
            boxprops={"linewidth": 1.0},
        )

        for box_patch, system_name in zip(boxplot["boxes"], system_order):
            if system_name == wildtype_system:
                box_patch.set_facecolor("#d9d9d9")
                box_patch.set_edgecolor("#222222")
            else:
                box_patch.set_facecolor(color_map[system_name])
                box_patch.set_edgecolor("#222222")
            box_patch.set_alpha(0.82)

        if axis_index == 0:
            axis.tick_params(axis="y", labelsize=9)
            axis.set_ylabel("System")
        else:
            axis.tick_params(axis="y", labelleft=False)
        axis.invert_yaxis()
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.22, linewidth=0.55)

        metric_significance_df = significance_df.loc[
            significance_df["metric_key"].eq(metric_key)
            & significance_df["stars"].astype(str).ne("")
        ].copy()
        if not metric_significance_df.empty:
            x_left, x_right = axis.get_xlim()
            x_span = x_right - x_left
            edge_padding = 0.03 * x_span if np.isfinite(x_span) and x_span > 0 else 0.35
            star_by_system = metric_significance_df.set_index("system")["stars"].to_dict()
            for box_index, (position, system_name, system_values) in enumerate(
                zip(positions, system_order, values_by_system)
            ):
                stars = star_by_system.get(system_name, "")
                finite_values = system_values[np.isfinite(system_values)]
                if not stars or finite_values.size == 0:
                    continue
                whisker_indices = (2 * box_index, 2 * box_index + 1)
                visible_right_edge = max(
                    float(np.nanmax(boxplot["whiskers"][whisker_index].get_xdata()))
                    for whisker_index in whisker_indices
                )
                star_anchor_x = min(visible_right_edge, x_right - edge_padding)
                anchor_at_edge = np.isclose(star_anchor_x, x_right - edge_padding)
                axis.annotate(
                    stars,
                    xy=(star_anchor_x, position),
                    xytext=((-3, 0) if anchor_at_edge else (3, 0)),
                    textcoords="offset points",
                    va="center",
                    ha=("right" if anchor_at_edge else "left"),
                    fontsize=12,
                    fontweight="bold",
                    color="#111111",
                    clip_on=True,
                )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.suptitle(
        f"{family_id}: absolute interaction-count and SASA distributions\n"
        "Counts show framewise replicate-mean distributions in their native units; "
        "SASA shows residue-summed replicate means in absolute units. "
        "Count stars mark distribution-shift tests on 100-frame block means; "
        "SASA stars mark Welch tests on replicate means, all after BH correction "
        "(* < 0.05, ** < 0.01, *** < 0.001).",
        fontsize=12.5,
        y=0.98,
    )
    return figure


def plot_md_variant_pair_delta_bars(
    family_id: str,
    variant_system: str,
    metadata_df: pd.DataFrame,
    pair_delta_by_metric: dict[str, pd.DataFrame],
):
    variant_label = get_md_variant_display_label(metadata_df, variant_system)
    figure, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

    for axis, metric_name in zip(axes, MD_VARIANT_PAIR_METRICS):
        variant_df = pair_delta_by_metric[metric_name].loc[
            pair_delta_by_metric[metric_name]["variant_system"].eq(variant_system)
        ].copy()
        variant_df = variant_df.sort_values("delta_occupancy")
        selected_df = pd.concat(
            [variant_df.head(5), variant_df.tail(5)],
            ignore_index=True,
        ).sort_values("delta_occupancy")
        if selected_df.empty:
            axis.set_visible(False)
            continue

        bar_colors = np.where(
            selected_df["delta_occupancy"] >= 0.0,
            "#1b9e77",
            "#d95f02",
        )
        edge_colors = np.where(
            selected_df["is_local_pair"],
            "#111111",
            "#ffffff",
        )
        axis.barh(
            np.arange(len(selected_df)),
            selected_df["delta_occupancy"],
            color=bar_colors,
            edgecolor=edge_colors,
            linewidth=1.0,
        )
        axis.axvline(0.0, color="#333333", linewidth=1.0)
        axis.set_yticks(np.arange(len(selected_df)))
        axis.set_yticklabels(
            [wrap_md_variant_label(label, width=24) for label in selected_df["pair_label"]],
            fontsize=9,
        )
        axis.set_xlabel("Variant mean occupancy - WT mean occupancy")
        axis.set_title(
            {
                "residue_contact_pairs": "Residue contacts",
                "saltbridge_pairs": "Salt bridges",
                "hbond_pairs": "Hydrogen bonds",
            }[metric_name]
        )
        axis.grid(axis="x", alpha=0.18, linewidth=0.5)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#1b9e77", label="Gained occupancy"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d95f02", label="Lost occupancy"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#111111", label="Touches local mutation neighborhood"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        f"{family_id} / {variant_label}: top gained and lost interactions vs WT\n"
        "Outlined bars mark pairs that include mutation-neighborhood residues.",
        fontsize=13,
        y=1.05,
    )
    return figure


def plot_md_family_local_signal_heatmap(
    family_id: str,
    metadata_df: pd.DataFrame,
    variant_signal_df: pd.DataFrame,
):
    plot_df = build_md_variant_local_signal_plot_table(
        variant_signal_df=variant_signal_df,
        metadata_df=metadata_df,
        family_id=family_id,
    )
    heatmap_columns = [
        "delta_rmsf_neighborhood_mean",
        "delta_sasa_neighborhood_mean",
        "delta_ordered_ss_neighborhood_mean",
        "contact_gain_local",
        "saltbridge_gain_local",
        "hbond_gain_local",
        "contact_gain_local_fraction",
        "hbond_gain_local_fraction",
    ]
    label_map = {
        "delta_rmsf_neighborhood_mean": "Neighborhood RMSF delta",
        "delta_sasa_neighborhood_mean": "Neighborhood SASA delta",
        "delta_ordered_ss_neighborhood_mean": "Neighborhood SS delta",
        "contact_gain_local": "Local contact gain",
        "saltbridge_gain_local": "Local salt-bridge gain",
        "hbond_gain_local": "Local H-bond gain",
        "contact_gain_local_fraction": "Local contact fraction",
        "hbond_gain_local_fraction": "Local H-bond fraction",
    }

    figure, axis = plt.subplots(
        figsize=(12, max(3.2, 0.42 * len(plot_df) + 1.8)),
        constrained_layout=True,
    )
    image = axis.imshow(
        plot_df[heatmap_columns].to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5),
    )
    axis.set_xticks(np.arange(len(heatmap_columns)))
    axis.set_xticklabels(
        [label_map[column_name] for column_name in heatmap_columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(plot_df)))
    axis.set_yticklabels(plot_df["variant_label"])
    axis.set_title(
        f"{family_id}: local mutation-neighborhood summary\n"
        "Each column is standardized within the family to highlight relative local shifts between variants.",
        fontsize=13,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02)
    colorbar.set_label("Within-family z-score")
    return figure


def plot_md_family_secondary_structure_summary(
    family_id: str,
    metadata_df: pd.DataFrame,
    ss_counts_summary_df: pd.DataFrame,
    ss_residue_delta_df: pd.DataFrame,
):
    system_order = compute_md_variant_system_order(metadata_df, family_id)
    system_summary_df = ss_counts_summary_df.loc[
        ss_counts_summary_df["system"].isin(system_order)
    ].copy()
    system_summary_df["system"] = pd.Categorical(
        system_summary_df["system"],
        categories=system_order,
        ordered=True,
    )
    system_summary_df = system_summary_df.sort_values("system")
    system_labels = [
        get_md_variant_display_label(metadata_df, system_name)
        for system_name in system_summary_df["system"]
    ]

    variant_order = compute_md_variant_order(metadata_df, family_id)
    family_ss_delta_df = ss_residue_delta_df.loc[
        ss_residue_delta_df["family"].eq(family_id)
    ].copy()
    family_ss_delta_df["variant_system"] = pd.Categorical(
        family_ss_delta_df["variant_system"],
        categories=variant_order,
        ordered=True,
    )
    family_ss_delta_df = family_ss_delta_df.sort_values(["variant_system", "resid"])
    helix_pivot = family_ss_delta_df.pivot(
        index="variant_system",
        columns="resid",
        values="delta_helix_occupancy",
    ).loc[variant_order]
    strand_pivot = family_ss_delta_df.pivot(
        index="variant_system",
        columns="resid",
        values="delta_strand_occupancy",
    ).loc[variant_order]

    residue_numbers = helix_pivot.columns.to_numpy(dtype=int)
    tick_values = choose_residue_ticks(residue_numbers)
    tick_positions = [
        int(np.where(residue_numbers == tick_value)[0][0])
        for tick_value in tick_values
        if tick_value in residue_numbers
    ]
    row_labels = [
        get_md_variant_display_label(metadata_df, system_name)
        for system_name in variant_order
    ]

    combined_delta_values = np.concatenate(
        [
            helix_pivot.to_numpy(dtype=float).ravel(),
            strand_pivot.to_numpy(dtype=float).ravel(),
        ]
    )
    finite_delta_values = combined_delta_values[np.isfinite(combined_delta_values)]
    max_abs_delta = float(np.max(np.abs(finite_delta_values))) if len(finite_delta_values) else 1e-3
    max_abs_delta = max(max_abs_delta, 1e-3)

    figure = plt.figure(figsize=(17, 12), constrained_layout=True)
    grid_spec = figure.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.0])
    axis_bar = figure.add_subplot(grid_spec[0, 0])
    axis_helix = figure.add_subplot(grid_spec[1, 0])
    axis_strand = figure.add_subplot(grid_spec[2, 0])

    x_positions = np.arange(len(system_summary_df))
    axis_bar.bar(x_positions, system_summary_df["helix_fraction_mean"], color="#ef8a62", label="Helix")
    axis_bar.bar(
        x_positions,
        system_summary_df["strand_fraction_mean"],
        bottom=system_summary_df["helix_fraction_mean"],
        color="#67a9cf",
        label="Strand",
    )
    axis_bar.bar(
        x_positions,
        system_summary_df["loop_fraction_mean"],
        bottom=system_summary_df["helix_fraction_mean"] + system_summary_df["strand_fraction_mean"],
        color="#d9d9d9",
        label="Loop",
    )
    axis_bar.set_xticks(x_positions)
    axis_bar.set_xticklabels(system_labels, rotation=45, ha="right")
    axis_bar.set_ylabel("Mean fraction")
    axis_bar.set_title("Global secondary-structure composition per system")
    axis_bar.legend(frameon=False, ncol=3, loc="upper right")
    axis_bar.grid(axis="y", alpha=0.18, linewidth=0.5)

    helix_image = axis_helix.imshow(
        helix_pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta),
    )
    axis_helix.set_yticks(np.arange(len(row_labels)))
    axis_helix.set_yticklabels(row_labels)
    axis_helix.set_xticks(tick_positions)
    axis_helix.set_xticklabels(tick_values)
    axis_helix.set_ylabel("Variant")
    axis_helix.set_title("Per-residue helix occupancy delta (variant - WT)")

    strand_image = axis_strand.imshow(
        strand_pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta),
    )
    axis_strand.set_yticks(np.arange(len(row_labels)))
    axis_strand.set_yticklabels(row_labels)
    axis_strand.set_xticks(tick_positions)
    axis_strand.set_xticklabels(tick_values)
    axis_strand.set_xlabel("Residue number")
    axis_strand.set_ylabel("Variant")
    axis_strand.set_title("Per-residue strand occupancy delta (variant - WT)")

    colorbar = figure.colorbar(strand_image, ax=[axis_helix, axis_strand], fraction=0.02, pad=0.01)
    colorbar.set_label("Occupancy delta")
    figure.suptitle(
        f"{family_id}: secondary-structure retention summary\n"
        "Positive helix or strand deltas indicate residues that retain ordered secondary structure more often than WT.",
        fontsize=14,
    )
    return figure


def plot_md_variant_signal_dashboard(
    variant_signal_df: pd.DataFrame,
):
    dashboard_columns = MD_VARIANT_SIGNAL_COLUMNS + [
        "delta_ordered_fraction_global",
        "contact_gain_local_fraction",
        "saltbridge_gain_local_fraction",
        "hbond_gain_local_fraction",
    ]
    label_map = {
        "rmsf_neighborhood_gain": "Lower local RMSF",
        "contact_gain_total": "Contact gain",
        "saltbridge_gain_total": "Salt-bridge gain",
        "hbond_gain_total": "H-bond gain",
        "sasa_neighborhood_gain": "Lower local SASA",
        "ordered_ss_neighborhood_gain": "Higher local SS",
        "delta_ordered_fraction_global": "Higher global SS",
        "contact_gain_local_fraction": "Contact gain local",
        "saltbridge_gain_local_fraction": "Salt-bridge gain local",
        "hbond_gain_local_fraction": "H-bond gain local",
    }
    dashboard_df = variant_signal_df[
        ["family", "variant_label", "composite_stabilizing_score", *dashboard_columns]
    ].copy()
    dashboard_df = dashboard_df.sort_values(
        ["composite_stabilizing_score", "family"],
        ascending=[False, True],
    ).reset_index(drop=True)

    standardized_df = dashboard_df[dashboard_columns].copy()
    for column_name in dashboard_columns:
        values = standardized_df[column_name].astype(float)
        if float(values.std(ddof=0)) > 0:
            standardized_df[column_name] = (values - values.mean()) / values.std(ddof=0)
        else:
            standardized_df[column_name] = 0.0

    figure, axis = plt.subplots(
        figsize=(13, max(4.2, 0.42 * len(dashboard_df) + 2.1)),
        constrained_layout=True,
    )
    image = axis.imshow(
        standardized_df.to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_DELTA_CMAP,
        norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=2.5),
    )
    axis.set_xticks(np.arange(len(dashboard_columns)))
    axis.set_xticklabels(
        [label_map[column_name] for column_name in dashboard_columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(dashboard_df)))
    axis.set_yticklabels(
        [f"{row.family} / {row.variant_label}" for row in dashboard_df.itertuples()],
        fontsize=9,
    )
    axis.set_title(
        "Variant stabilizing-signal dashboard\n"
        "Colors are z-scores across variants, so red means a stronger WT-relative stabilizing signature in that column.",
        fontsize=13,
    )

    previous_family = None
    for row_index, family_id in enumerate(dashboard_df["family"]):
        if previous_family is not None and family_id != previous_family:
            axis.axhline(row_index - 0.5, color="#222222", linewidth=1.0)
        previous_family = family_id

    colorbar = figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02)
    colorbar.set_label("Across-variant z-score")
    return figure


def plot_md_variant_family_pattern_heatmap(
    family_pattern_summary_df: pd.DataFrame,
):
    plot_df = family_pattern_summary_df.loc[
        family_pattern_summary_df["family"].ne("ALL")
    ].copy()
    heatmap_columns = [
        "fraction_lower_local_rmsf",
        "fraction_lower_local_sasa",
        "fraction_higher_local_ss",
        "fraction_contact_gain",
        "fraction_saltbridge_gain",
        "fraction_hbond_gain",
        "median_contact_local_fraction",
        "median_hbond_local_fraction",
    ]
    label_map = {
        "fraction_lower_local_rmsf": "Lower local RMSF",
        "fraction_lower_local_sasa": "Lower local SASA",
        "fraction_higher_local_ss": "Higher local SS",
        "fraction_contact_gain": "Any contact gain",
        "fraction_saltbridge_gain": "Any salt-bridge gain",
        "fraction_hbond_gain": "Any H-bond gain",
        "median_contact_local_fraction": "Contact gain local",
        "median_hbond_local_fraction": "H-bond gain local",
    }

    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = axis.imshow(
        plot_df[heatmap_columns].to_numpy(dtype=float),
        aspect="auto",
        cmap=MD_VARIANT_OCCUPANCY_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(np.arange(len(heatmap_columns)))
    axis.set_xticklabels(
        [label_map[column_name] for column_name in heatmap_columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(plot_df)))
    axis.set_yticklabels(plot_df["family"])
    axis.set_title(
        "Shared family-level patterns\n"
        "Each value is the fraction of variants in a family showing the named stabilizing trend.",
        fontsize=13,
    )

    for row_index in range(len(plot_df)):
        for column_index, column_name in enumerate(heatmap_columns):
            value = float(plot_df.iloc[row_index][column_name])
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=9,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02)
    colorbar.set_label("Fraction / median local fraction")
    return figure
