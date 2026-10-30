"""Build the audit-backed two-acquisition CAMO-FS temporal representation.

The output labels EARLY, TRANSITION, and LATE are operational names for the
July 5 snapshot, July 14-minus-July 5 change, and July 14 snapshot. They are
not phenological stage assignments.

Run from the project root with:
    python src/camo_fs/preprocessing/build_temporal_dataset.py

The raw ZIP is read in place; its member is not extracted. July 5 and July 14
snapshots are stored as float32 after finite-range checks. TRANSITION is
computed from the parsed float64 source values and stored as float64 to retain
the exact result of that subtraction. Labels retain their source integer dtype.
Requires pandas and pyarrow (pyarrow is listed in requirements.txt).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ZIP = PROJECT_ROOT / "data/raw/crop+mapping+using+fused+optical+radar+data+set.zip"
AUDIT_JSON = PROJECT_ROOT / "docs/data_audit.json"
PARQUET_OUTPUT = PROJECT_ROOT / "data/processed/camo_temporal_261.parquet"
METADATA_OUTPUT = PROJECT_ROOT / "data/processed/camo_temporal_feature_metadata.csv"
ARCHIVE_MEMBER = "WinnipegDataset.txt"
DATE_EARLY = "2012-07-05"
DATE_LATE = "2012-07-14"
REPRESENTATIONS = ("EARLY", "TRANSITION", "LATE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mapping(audit_path: Path) -> list[dict[str, object]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    mapping = audit.get("official_same_measurement_pairs")
    if not isinstance(mapping, list) or len(mapping) != 87:
        raise ValueError("Audit JSON must contain exactly 87 verified date-pair records.")
    if audit.get("rows") != 325834:
        raise ValueError("Audit JSON row count does not match the Step 0 verified count.")
    if audit.get("predictor_count") != 174:
        raise ValueError("Audit JSON must identify 174 source predictors.")
    for index, pair in enumerate(mapping, start=1):
        required = {
            "first_date", "first_name", "second_date", "second_name", "sensor",
            "same_measurement_name",
        }
        if not required.issubset(pair):
            raise ValueError(f"Audit mapping record {index} lacks required fields.")
        if pair["same_measurement_name"] is not True:
            raise ValueError(f"Audit mapping record {index} is not verified as a same-name pair.")
        for key in ("first_date", "second_date", "first_name", "second_name", "sensor"):
            if not isinstance(pair[key], str):
                raise ValueError(f"Audit mapping record {index} has invalid {key}.")
        if pair["first_date"] == pair["second_date"]:
            raise ValueError(f"Audit mapping record {index} maps a column to itself.")
    first_columns = [str(pair["first_date"]) for pair in mapping]
    second_columns = [str(pair["second_date"]) for pair in mapping]
    if len(set(first_columns)) != 87 or len(set(second_columns)) != 87:
        raise ValueError("Audit mapping contains duplicate source columns.")
    if set(first_columns) & set(second_columns):
        raise ValueError("Audit mapping reuses a column across acquisition dates.")
    return mapping


def measurement_base(pair: dict[str, object]) -> str:
    first_name = str(pair["first_name"])
    second_name = str(pair["second_name"])
    sensor = str(pair["sensor"]).lower()
    suffixes = {
        "radar": ("_Rad05July", "_Rad14July"),
        "optical": ("_Opt05July", "_Opt14July"),
    }
    if sensor not in suffixes:
        raise ValueError(f"Unexpected sensor/modality in audit mapping: {sensor}")
    first_suffix, second_suffix = suffixes[sensor]
    if not first_name.endswith(first_suffix) or not second_name.endswith(second_suffix):
        raise ValueError(f"UCI date suffixes do not match metadata for {first_name}/{second_name}.")
    first_base = first_name[: -len(first_suffix)]
    second_base = second_name[: -len(second_suffix)]
    if first_base != second_base:
        raise ValueError(f"UCI base measurement names differ: {first_base}/{second_base}.")
    return first_base


def build_metadata(mapping: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for representation in REPRESENTATIONS:
        for channel_number, pair in enumerate(mapping, start=1):
            channel_id = f"CH{channel_number:03d}"
            measurement = measurement_base(pair)
            sensor = str(pair["sensor"]).lower()
            feature_name = f"{representation}_{channel_id}_{measurement}"
            if representation == "EARLY":
                transformation = "identity: July 5 measurement"
            elif representation == "TRANSITION":
                transformation = "July 14 measurement minus July 5 measurement"
            else:
                transformation = "identity: July 14 measurement"
            rows.append(
                {
                    "feature_name": feature_name,
                    "representation": representation,
                    "channel_id": channel_id,
                    "source_july5_column": str(pair["first_date"]),
                    "source_july14_column": str(pair["second_date"]),
                    "sensor": sensor,
                    "measurement_name": measurement,
                    "date_early": DATE_EARLY,
                    "date_late": DATE_LATE,
                    "transformation": transformation,
                }
            )
    return pd.DataFrame(rows)


def load_raw_data(raw_zip: Path) -> tuple[pd.DataFrame, str]:
    digest_before = sha256_file(raw_zip)
    with zipfile.ZipFile(raw_zip, "r") as archive:
        if ARCHIVE_MEMBER not in archive.namelist():
            raise FileNotFoundError(f"{ARCHIVE_MEMBER} is not present in {raw_zip.name}.")
        with archive.open(ARCHIVE_MEMBER, "r") as raw_stream:
            raw = pd.read_csv(raw_stream)
    return raw, digest_before


def validate_and_write(
    raw: pd.DataFrame,
    mapping: list[dict[str, object]],
    metadata: pd.DataFrame,
    raw_zip: Path,
    expected_sha256: str,
) -> dict[str, object]:
    expected_raw_columns = {"label"}
    expected_raw_columns.update(str(pair["first_date"]) for pair in mapping)
    expected_raw_columns.update(str(pair["second_date"]) for pair in mapping)
    if len(raw) != 325834:
        raise ValueError(f"Expected 325834 source rows; found {len(raw)}.")
    if raw.columns[0] != "label" or len(raw.columns) != 175:
        raise ValueError("Raw table must contain label plus 174 predictors.")
    if not expected_raw_columns.issubset(raw.columns):
        raise ValueError("At least one audit-mapped source column is absent from the raw data.")
    if raw["label"].isna().any() or not pd.api.types.is_integer_dtype(raw["label"].dtype):
        raise ValueError("Source label must be non-missing and integer typed.")

    first_columns = [str(pair["first_date"]) for pair in mapping]
    second_columns = [str(pair["second_date"]) for pair in mapping]
    july5 = raw[first_columns].to_numpy(dtype=np.float64, copy=True)
    july14 = raw[second_columns].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(july5).all() or not np.isfinite(july14).all():
        raise ValueError("Mapped source measurements contain NaN or infinity.")

    # Snapshot downcasting is used only after confirming float32 remains finite.
    early32 = july5.astype(np.float32)
    late32 = july14.astype(np.float32)
    if not np.isfinite(early32).all() or not np.isfinite(late32).all():
        raise ValueError("A snapshot value overflows float32; refusing unsafe downcast.")
    # Keep date differences in float64 so they equal the subtraction of parsed
    # raw source values, rather than differences after rounding both operands.
    transition64 = np.subtract(july14, july5, dtype=np.float64)
    if not np.isfinite(transition64).all():
        raise ValueError("Transition calculation produced NaN or infinity.")

    expected_feature_names = metadata["feature_name"].tolist()
    if len(metadata) != 261 or len(set(expected_feature_names)) != 261:
        raise ValueError("Metadata must contain 261 distinct derived feature records.")

    values: dict[str, object] = {"label": raw["label"].to_numpy(copy=True)}
    for rep, matrix in zip(REPRESENTATIONS, (early32, transition64, late32)):
        names = metadata.loc[metadata["representation"] == rep, "feature_name"].tolist()
        if len(names) != 87:
            raise ValueError(f"Expected 87 feature names for {rep}; found {len(names)}.")
        for idx, name in enumerate(names):
            values[name] = matrix[:, idx]
    derived = pd.DataFrame(values)

    # Verify the intended feature block layout and transformations before disk IO.
    expected_order = ["label"] + expected_feature_names
    if derived.columns.tolist() != expected_order:
        raise ValueError("Derived output columns do not follow metadata representation order.")
    if derived.shape != (325834, 262):
        raise ValueError(f"Unexpected output shape: {derived.shape}.")
    if not np.array_equal(derived["label"].to_numpy(), raw["label"].to_numpy()):
        raise ValueError("Labels changed during construction.")
    transition_names = metadata.loc[
        metadata["representation"] == "TRANSITION", "feature_name"
    ].tolist()
    if not np.array_equal(derived[transition_names].to_numpy(), transition64):
        raise ValueError("Transition values are not exactly July 14 minus July 5.")
    if derived[expected_feature_names].isna().to_numpy().any():
        raise ValueError("Derived features contain missing values.")
    if not np.isfinite(derived[expected_feature_names].to_numpy()).all():
        raise ValueError("Derived features contain NaN or infinity.")
    duplicate_count = int(derived.duplicated(keep="first").sum())
    if duplicate_count:
        raise ValueError(f"Output contains {duplicate_count} duplicate rows.")

    PARQUET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PARQUET_OUTPUT.parent.mkdir(exist_ok=True)
    parquet_fd, parquet_tmp_name = tempfile.mkstemp(
        prefix=".camo_temporal_261_", suffix=".parquet", dir=PARQUET_OUTPUT.parent
    )
    os.close(parquet_fd)
    metadata_fd, metadata_tmp_name = tempfile.mkstemp(
        prefix=".camo_temporal_feature_metadata_", suffix=".csv", dir=METADATA_OUTPUT.parent
    )
    os.close(metadata_fd)
    parquet_tmp = Path(parquet_tmp_name)
    metadata_tmp = Path(metadata_tmp_name)
    try:
        derived.to_parquet(parquet_tmp, engine="pyarrow", compression="zstd", index=False)
        metadata.to_csv(metadata_tmp, index=False, encoding="utf-8")

        # Read persisted outputs back and validate on-disk values/schema.
        persisted = pd.read_parquet(parquet_tmp, engine="pyarrow")
        persisted_metadata = pd.read_csv(metadata_tmp, dtype=str, keep_default_na=False)
        if persisted.shape != (325834, 262):
            raise ValueError(f"Persisted Parquet shape mismatch: {persisted.shape}.")
        if persisted.columns.tolist() != expected_order:
            raise ValueError("Persisted Parquet feature order changed.")
        if not np.array_equal(persisted["label"].to_numpy(), raw["label"].to_numpy()):
            raise ValueError("Persisted Parquet labels do not exactly match the source labels.")
        if not np.array_equal(persisted[transition_names].to_numpy(), transition64):
            raise ValueError("Persisted Parquet transition values failed exact validation.")
        if persisted[expected_feature_names].isna().to_numpy().any():
            raise ValueError("Persisted Parquet contains missing derived feature values.")
        if not np.isfinite(persisted[expected_feature_names].to_numpy()).all():
            raise ValueError("Persisted Parquet contains NaN or infinity.")
        if int(persisted.duplicated(keep="first").sum()) != 0:
            raise ValueError("Persisted Parquet contains duplicate rows.")
        if len(persisted_metadata) != 261 or persisted_metadata["feature_name"].tolist() != expected_feature_names:
            raise ValueError("Persisted metadata CSV does not match the output schema.")
        representation_counts = persisted_metadata["representation"].value_counts().to_dict()
        if representation_counts != {"EARLY": 87, "TRANSITION": 87, "LATE": 87}:
            raise ValueError(f"Unexpected metadata representation counts: {representation_counts}.")

        if sha256_file(raw_zip) != expected_sha256:
            raise ValueError("The source ZIP hash changed during the build; output not finalized.")

        parquet_tmp.replace(PARQUET_OUTPUT)
        metadata_tmp.replace(METADATA_OUTPUT)
    finally:
        parquet_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)


        

    return {
        "rows": int(len(derived)),
        "derived_features": int(len(expected_feature_names)),
        "representation_counts": {key: int(value) for key, value in representation_counts.items()},
        "target": "label",
        "target_dtype": str(persisted["label"].dtype),
        "snapshot_dtype": "float32",
        "transition_dtype": "float64",
        "transition_exact_check": True,
        "nan_or_infinity": False,
        "duplicate_rows": 0,
        "source_sha256_unchanged": True,
        "parquet_path": str(PARQUET_OUTPUT.relative_to(PROJECT_ROOT)),
        "metadata_path": str(METADATA_OUTPUT.relative_to(PROJECT_ROOT)),
        "parquet_bytes": PARQUET_OUTPUT.stat().st_size,
        "metadata_bytes": METADATA_OUTPUT.stat().st_size,
    }


def main() -> None:
    if not RAW_ZIP.is_file():
        raise FileNotFoundError(f"Raw UCI archive not found: {RAW_ZIP}")
    if not AUDIT_JSON.is_file():
        raise FileNotFoundError(f"Step 0 audit JSON not found: {AUDIT_JSON}")
    mapping = load_mapping(AUDIT_JSON)
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    actual_sha = sha256_file(RAW_ZIP)
    if actual_sha != audit.get("zip_sha256"):
        raise ValueError("Raw ZIP SHA-256 differs from the completed Step 0 audit.")
    raw, digest_before = load_raw_data(RAW_ZIP)
    if digest_before != actual_sha:
        raise ValueError("Source ZIP changed while it was being read.")
    metadata = build_metadata(mapping)
    summary = validate_and_write(raw, mapping, metadata, RAW_ZIP, actual_sha)
    print("CAMO-FS temporal dataset build complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
