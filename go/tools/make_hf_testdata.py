# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rewrite demucs-go safetensors test fixtures in the layout the Hugging Face
hub repositories use (demucs/hf.py): one safetensors file per model with the
constructor arguments in the metadata, plus the bag yaml. Pure Python, no
torch: the tensor bytes are copied as they are.

Usage:
    python3 go/tools/make_hf_testdata.py go/testdata/tiny.safetensors go/testdata/hf tiny
    python3 go/tools/make_hf_testdata.py go/testdata/tiny_bag.safetensors go/testdata/hf tiny_bag
"""
import json
from pathlib import Path
import struct
import sys


def read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    metadata = header.pop("__metadata__")
    return metadata, header, data


def write(path, metadata, entries, blobs):
    header = {"__metadata__": metadata}
    offset = 0
    out = []
    for name, (dtype, shape, blob) in zip(entries, blobs):
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        out.append(blob)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in out:
            f.write(blob)


def hf_kwargs(config):
    kwargs = dict(config)
    seg = kwargs["segment"]
    if isinstance(seg, dict):
        kwargs["segment"] = {"_type": "fraction", **seg}
    return kwargs


def main():
    src, out_dir, bag = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata, header, data = read(src)
    configs = json.loads(metadata["models"])
    weights = json.loads(metadata["weights"])
    sigs = []
    for idx, config in enumerate(configs):
        sig = f"{bag}{idx}"
        sigs.append(sig)
        prefix = f"models.{idx}."
        names = sorted(k for k in header if k.startswith(prefix))
        entries, blobs = [], []
        for name in names:
            e = header[name]
            a, b = e["data_offsets"]
            entries.append(name[len(prefix):])
            blobs.append((e["dtype"], e["shape"], data[a:b]))
        meta = {"klass": "demucs.htdemucs.HTDemucs", "args": "[]", "kwargs": json.dumps(hf_kwargs(config))}
        write(out_dir / f"{sig}.safetensors", meta, entries, blobs)
    lines = ["models: [" + ", ".join(f"'{s}'" for s in sigs) + "]"]
    if len(sigs) > 1:
        lines.append("weights: [")
        for row in weights:
            lines.append("  [" + ", ".join(f"{w:g}." if float(w).is_integer() else f"{w:g}" for w in row) + "],")
        lines.append("]")
    (out_dir / f"{bag}.yaml").write_text("\n".join(lines) + "\n")
    print("wrote", out_dir / f"{bag}.yaml", sigs)


if __name__ == "__main__":
    main()
