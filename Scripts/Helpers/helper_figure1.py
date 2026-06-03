from __future__ import annotations

from pathlib import Path
import re


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


def merge_uniprot_annotations(input_dir: str | Path = "Data/Figure1") -> Path:
    return merge_emapper_annotations(
        input_dir=input_dir,
        pattern="file_*.emapper.annotations",
        output_name="Uniprot.emapper.annotations",
    )


def merge_swissprot_annotations(
    input_dir: str | Path = "Data/Figure1/Swiss_eggnog",
) -> Path:
    return merge_emapper_annotations(
        input_dir=input_dir,
        pattern="part_*.emapper.annotations",
        output_name="SiwssProt.emapper.annotations",
    )
