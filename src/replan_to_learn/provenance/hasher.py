"""
SIH26054 Replan to Learn: Provenance & Data Versioning.
Cryptographic & Canonical Hashing Engine.

Provides RFC 8785 JSON Canonicalization Scheme (JCS) serialization,
deterministic SHA256 hashing of Arrow RecordBatch streams (independent of Parquet
timestamps/codecs), file checksum computation, and manifest self-hashing.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import polars as pl
import pyarrow as pa


# ============================================================================
# 1. RFC 8785 JSON Canonicalization Scheme (JCS) Serialization
# ============================================================================

def _canonicalize_value(val: Any) -> Any:
    """Recursively convert data structures to RFC 8785 canonical JSON-serializable structures."""
    if val is None:
        return None
    elif isinstance(val, (bool, str)):
        return val
    elif isinstance(val, int):
        return val
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"NaN and Infinity are not permitted in RFC 8785 canonical JSON: {val}")
        if val == 0.0:
            return 0
        # If float represents an exact integer value within safe integer range, format cleanly
        if val.is_integer() and abs(val) < 9007199254740992:
            # Note: in Python json.dumps, float 1.0 dumps as 1.0, but ints dump as 1.
            # Keep as float unless specified, but normalize -0.0
            return val
        return val
    elif isinstance(val, (datetime.datetime, datetime.date)):
        if isinstance(val, datetime.datetime) and val.tzinfo is None:
            # Default to UTC if naive
            return val.replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        return val.isoformat().replace("+00:00", "Z")
    elif isinstance(val, dict):
        return {
            str(k): _canonicalize_value(v)
            for k, v in sorted(val.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(val, (list, tuple, set)):
        return [_canonicalize_value(v) for v in val]
    elif dataclasses.is_dataclass(val):
        return _canonicalize_value(dataclasses.asdict(val))
    elif hasattr(val, "model_dump") and callable(getattr(val, "model_dump")):
        return _canonicalize_value(val.model_dump())
    elif hasattr(val, "to_dict") and callable(getattr(val, "to_dict")):
        return _canonicalize_value(val.to_dict())
    else:
        # Fallback to string representation
        return str(val)


def canonical_json_dumps(obj: Any) -> str:
    """
    Serialize a Python object into an RFC 8785 (JCS) compliant canonical JSON string.

    Rules:
    - Lexicographically sorted dictionary keys (Unicode code point order)
    - Compact separators (no trailing spaces after ':' or ',')
    - Strict UTF-8 compatible encoding without BOM
    - NaN and Infinity are strictly rejected
    """
    canonical_obj = _canonicalize_value(obj)
    return json.dumps(
        canonical_obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_json_loads(s: Union[str, bytes]) -> Any:
    """Parse a canonical JSON string or UTF-8 bytes into Python objects."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return json.loads(s)


# ============================================================================
# 2. File Checksum Computation
# ============================================================================

def payload_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Compute deterministic SHA256 hex digest of a file on disk by streaming in chunks.

    Parameters
    ----------
    file_path : str | Path
        Path to the target file on disk.
    chunk_size : int
        Chunk size in bytes for reading (default 64 KB).

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA256 digest.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Target payload file does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Target payload path is a directory, not a file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


# ============================================================================
# 3. Canonical Arrow RecordBatch Stream Data Hashing
# ============================================================================

def canonical_data_sha256(
    data: Union[pa.Table, pl.DataFrame, pa.RecordBatchReader, Sequence[pa.RecordBatch]]
) -> str:
    """
    Compute a deterministic SHA256 hash over tabular Arrow / Polars data.

    To guarantee bit-level reproducibility independent of Parquet write timestamps,
    compression codec, row group partitioning, or column ordering:
    1. Converts input data to PyArrow Table.
    2. Combines all chunks to eliminate chunk fragmentation differences.
    3. Reorders columns lexicographically by column name.
    4. Strips custom metadata from the schema.
    5. Serializes to standard Arrow IPC RecordBatch stream buffer.
    6. Computes SHA256 over the continuous IPC byte buffer.

    Parameters
    ----------
    data : pa.Table | pl.DataFrame | pa.RecordBatchReader | Sequence[pa.RecordBatch]
        Input tabular telemetry/residual dataset.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA256 digest.
    """
    if isinstance(data, pl.DataFrame):
        table = data.to_arrow()
    elif isinstance(data, pa.Table):
        table = data
    elif isinstance(data, pa.RecordBatchReader):
        table = data.read_all()
    elif isinstance(data, (list, tuple)):
        if not data:
            # Empty table
            table = pa.table({})
        else:
            table = pa.Table.from_batches(data)
    else:
        raise TypeError(f"Unsupported data type for canonical hashing: {type(data)}")

    # Combine chunks to eliminate memory chunking variations
    table_combined = table.combine_chunks()

    # Reorder columns lexicographically
    sorted_col_names = sorted(table_combined.column_names)
    table_sorted = table_combined.select(sorted_col_names)

    # Strip schema metadata
    clean_schema = table_sorted.schema.with_metadata(None)

    # Serialize to Arrow IPC RecordBatch stream
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, clean_schema) as writer:
        for batch in table_sorted.to_batches():
            writer.write_batch(batch)

    stream_bytes = sink.getvalue()
    return hashlib.sha256(stream_bytes).hexdigest().lower()


# ============================================================================
# 4. Manifest Self-Hash Computation
# ============================================================================

def compute_manifest_hash(manifest_dict_or_obj: Any) -> str:
    """
    Compute deterministic SHA256 hash of a manifest's canonical JSON representation.

    Omits any self-referential 'manifest_hash' field to prevent circular dependency.

    Parameters
    ----------
    manifest_dict_or_obj : dict | ProvenanceManifest | Any
        Manifest object or dictionary representation.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA256 digest.
    """
    if dataclasses.is_dataclass(manifest_dict_or_obj):
        raw_dict = dataclasses.asdict(manifest_dict_or_obj)
    elif hasattr(manifest_dict_or_obj, "model_dump") and callable(getattr(manifest_dict_or_obj, "model_dump")):
        raw_dict = manifest_dict_or_obj.model_dump()
    elif hasattr(manifest_dict_or_obj, "to_dict") and callable(getattr(manifest_dict_or_obj, "to_dict")):
        raw_dict = manifest_dict_or_obj.to_dict()
    elif isinstance(manifest_dict_or_obj, dict):
        raw_dict = dict(manifest_dict_or_obj)
    else:
        raise TypeError(f"Cannot compute manifest hash for type: {type(manifest_dict_or_obj)}")

    # Make a shallow/deep copy and strip manifest_hash
    clean_dict = dict(raw_dict)
    clean_dict.pop("manifest_hash", None)
    if "extensions" in clean_dict and isinstance(clean_dict["extensions"], dict):
        ext_clean = dict(clean_dict["extensions"])
        ext_clean.pop("manifest_hash", None)
        clean_dict["extensions"] = ext_clean

    canonical_json_str = canonical_json_dumps(clean_dict)
    return hashlib.sha256(canonical_json_str.encode("utf-8")).hexdigest().lower()
