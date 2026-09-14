"""Copy tensor bytes into a gap-free safetensors file without requantization.

The source is never changed. A new destination is published only after the
complete copy is synced; an existing destination is never replaced.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile


WIDTH = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8,
    "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}
MAX_HEADER_BYTES = 32 * 1024**2
COPY_BYTES = 1024**2


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def layout(header, data_size):
    """Validate extents and describe a compact layout, allowing source gaps."""
    if not isinstance(header, dict) or data_size < 0:
        raise ValueError("Invalid header or truncated file")
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(v, str) for v in metadata.values()
    ):
        raise ValueError("Metadata values must be strings")
    rows = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor descriptor: {name}")
        dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
        if not isinstance(dtype, str) or dtype not in WIDTH:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f"Invalid shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(
            type(n) is not int for n in offsets
        ):
            raise ValueError(f"Invalid offsets: {name}")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"Tensor extends outside data: {name}")
        if end - start != math.prod(shape) * WIDTH[dtype]:
            raise ValueError(f"Tensor byte length disagrees with shape: {name}")
        rows.append((start, end, name, dtype, shape))

    old_end = new_end = 0
    result, spans = {}, []
    for start, end, name, dtype, shape in sorted(rows):
        if start < old_end:
            raise ValueError(f"Overlapping source tensors: {name}")
        result[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [new_end, new_end + end - start]}
        spans.append({"name": name, "source_start": start, "bytes": end - start,
                      "destination_start": new_end})
        old_end = end
        new_end += end - start
    if "__metadata__" in header:
        result["__metadata__"] = metadata
    return result, spans


def encode_header(header):
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    return raw + b" " * (-len(raw) % 8)


def compact_file(source, destination):
    """Return SHA256 receipts for the bytes copied for each tensor."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or os.path.lexists(destination):
        raise ValueError("Destination must be a new file")
    with source.open("rb") as original:
        initial = os.fstat(original.fileno())
        prefix = original.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing header length")
        n = struct.unpack("<Q", prefix)[0]
        if not 2 <= n <= MAX_HEADER_BYTES or n > initial.st_size - 8:
            raise ValueError("Invalid header length")
        header = json.loads(original.read(n), object_pairs_hook=_unique_object)
        fixed, spans = layout(header, initial.st_size - 8 - n)
        raw = encode_header(fixed)
        fd, temporary = tempfile.mkstemp(prefix=".compact-", dir=destination.parent)
        receipts = []
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(struct.pack("<Q", len(raw)))
                target.write(raw)
                for span in spans:
                    original.seek(8 + n + span["source_start"])
                    remaining, digest = span["bytes"], hashlib.sha256()
                    while remaining:
                        block = original.read(min(COPY_BYTES, remaining))
                        if not block:
                            raise ValueError("Truncated tensor data")
                        target.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    receipts.append({**span, "sha256": digest.hexdigest()})
                target.flush()
                os.fsync(target.fileno())
            final = os.fstat(original.fileno())
            if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
                final.st_size, final.st_mtime_ns, final.st_ctime_ns
            ):
                raise ValueError("Source changed during copy")
            # Same-filesystem link is atomic and refuses a concurrent destination.
            os.link(temporary, destination)
        finally:
            os.unlink(temporary)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        receipts = compact_file(args.source, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    source_bytes = args.source.stat().st_size
    destination_bytes = args.destination.stat().st_size
    print(json.dumps({
        "source_bytes": source_bytes,
        "destination_bytes": destination_bytes,
        # Signed delta so a caller can attest that only unreferenced bytes moved.
        "discarded_bytes": source_bytes - destination_bytes,
        "tensors": receipts,
    }, indent=2))


if __name__ == "__main__":
    main()
