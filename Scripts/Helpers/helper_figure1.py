from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


_NUMBER_PATTERN = re.compile(r"(\d+)")


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = _NUMBER_PATTERN.split(path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def merge_emapper_annotations(
    input_dir: str | Path,
    pattern: str,
    output_name: str | Path,
) -> Path:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern), key=_natural_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No EggNOG annotation files found in {input_dir} for pattern {pattern!r}"
        )

    output_path = Path(output_name)
    if not output_path.is_absolute():
        output_path = input_dir / output_path

    header = None
    with files[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#query"):
                header = line.rstrip("\n")
                break

    if header is None:
        raise ValueError(f"Missing #query header in {files[0]}")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"## Combined from {len(files)} files matching {pattern}\n")
        handle.write(f"{header}\n")

        for file_path in files:
            with file_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.startswith("#"):
                        handle.write(line)

    return output_path


def merge_ogt_annotations(input_dir: str | Path = "Data/Figure1") -> Path:
    return merge_emapper_annotations(
        input_dir=input_dir,
        pattern="part_*.emapper.annotations",
        output_name="ogt.emapper.annotations",
    )


def merge_uniprot_annotations(
    input_dir: str | Path = "Data/Figure1/EggNOG_annotations",
) -> Path:
    return merge_emapper_annotations(
        input_dir=input_dir,
        pattern="file_*.emapper.annotations",
        output_name="../Uniprot.emapper.annotations",
    )


def merge_swissprot_annotations(
    input_dir: str | Path = "Data/Figure1/Swiss_eggnog",
) -> Path:
    return merge_emapper_annotations(
        input_dir=input_dir,
        pattern="part_*.emapper.annotations",
        output_name="SiwssProt.emapper.annotations",
    )



####################################################################################
########################### Computing Thermophilic COGs ############################
####################################################################################

def extract_COG(x, idx):
    if x == "-":
        return x
    list_COGs = x.split(",")
    if len(list_COGs) < idx+1:
        return list_COGs[-1].split("@")[0]
        
    COG = list_COGs[idx].split("@")[0]
    return COG

ec_col = "EC"
def calc_num_EC_tempRange(df, min_temp=60, key="Temperature_y"):
    
    df_tmp = df
    ecs = (df_tmp[ec_col].dropna().astype(str).str.split(";").explode().str.strip())
    ecs = ecs[ecs.ne("")]  # remove empty strings
    ecs = ecs[ecs.str.match(r"^\d+(\.\d+|\.-){3}$", na=False)]
    unique_ec4 = pd.Index(ecs.unique())
    n_unique_ec4 = len(unique_ec4)

    print("Unique full ECs (x1.x2.x3.x4):", n_unique_ec4)

    df_tmp = df_tmp.loc[df_tmp[key]>=min_temp]
    ecs = (df_tmp[ec_col].dropna().astype(str).str.split(";").explode().str.strip())
    ecs = ecs[ecs.ne("")]  # remove empty strings
    ecs = ecs[ecs.str.match(r"^\d+(\.\d+|\.-){3}$", na=False)]
    unique_ec4_mintemp = pd.Index(ecs.unique())
    n_unique_ec4_mintemp = len(unique_ec4_mintemp)

    print(r"Unique full ECs with temperature above 60 $ \degree C$ (x1.x2.x3.x4):", n_unique_ec4_mintemp)
    
    print(fr"Fraction of ECs with sequence above 60 $ \degree C$: {n_unique_ec4_mintemp/n_unique_ec4:.2f}" )
    return n_unique_ec4, n_unique_ec4_mintemp

def calc_num_COG_tempRange(df, min_temp=60, key="Temperature_y", COG_col="root_COG"):
    df_tmp = df
    ecs = (df_tmp[COG_col].dropna())
    ecs = ecs[ecs.ne("")]  # remove empty strings
    
    unique_COG = pd.Index(ecs.unique())
    n_unique_COG = len(unique_COG)

    print("Unique COGs:", n_unique_COG)

    df_tmp = df_tmp.loc[df_tmp[key]>=min_temp]
    ecs = (df_tmp[COG_col].dropna())
    ecs = ecs[ecs.ne("")]  # remove empty strings
    unique_COG_mintemp = pd.Index(ecs.unique())
    n_unique_COG_mintemp = len(unique_COG_mintemp)

    print(r"Unique full COGs with temperature above 60 $ \degree C$:", n_unique_COG_mintemp)
    
    print(fr"Fraction of COGs with sequence above 60 $ \degree C$: {n_unique_COG_mintemp/n_unique_COG:.2f}" )
    return n_unique_COG, n_unique_COG_mintemp
