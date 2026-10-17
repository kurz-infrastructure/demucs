# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Export a pretrained Demucs model (or bag of models) to a single
safetensors file that can be loaded by the Go port in `go/`.

Only HTDemucs models are supported (htdemucs, htdemucs_ft, htdemucs_6s).

Usage:
    python3 go/tools/export_safetensors.py htdemucs htdemucs.safetensors
    python3 go/tools/export_safetensors.py -r ./release_models <sig> out.safetensors
"""
import argparse
import contextlib
import inspect
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys

import torch

try:  # an installed demucs (pip) works as well as this checkout
    import demucs  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demucs.apply import BagOfModels  # noqa: E402
from demucs.htdemucs import HTDemucs  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

FORMAT = "demucs-go/v1"


def _jsonable(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def model_config(model):
    """Full constructor arguments of `model`, defaults included."""
    args, kwargs = model._init_args_kwargs
    bound = inspect.signature(type(model).__init__).bind(model, *args, **kwargs)
    bound.apply_defaults()
    kwargs = dict(bound.arguments)
    kwargs.pop("self")
    segment = Fraction(model.segment).limit_denominator(1000)
    kwargs["segment"] = segment
    return {key: _jsonable(value) for key, value in kwargs.items()}


def write_safetensors(path, tensors, metadata, dtype):
    torch_dtype, st_dtype = {
        "f16": (torch.float16, "F16"),
        "f32": (torch.float32, "F32"),
    }[dtype]
    header = {"__metadata__": metadata}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        data = tensor.detach().to(torch_dtype).contiguous().cpu().numpy().tobytes()
        header[name] = {
            "dtype": st_dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)


def export(model, path, dtype="f16"):
    if isinstance(model, BagOfModels):
        models = list(model.models)
        weights = [list(map(float, w)) for w in model.weights]
    else:
        models = [model]
        weights = [[1.0] * len(model.sources)]
    tensors = {}
    configs = []
    for idx, sub in enumerate(models):
        if not isinstance(sub, HTDemucs):
            raise ValueError(f"only HTDemucs models are supported, got {type(sub).__name__}")
        configs.append(model_config(sub))
        for name, tensor in sub.state_dict().items():
            tensors[f"models.{idx}.{name}"] = tensor
    metadata = {
        "format": FORMAT,
        "models": json.dumps(configs),
        "weights": json.dumps(weights),
    }
    write_safetensors(path, tensors, metadata, dtype)


@contextlib.contextmanager
def _trusted_checkpoints():
    """torch>=2.6 defaults to `weights_only=True`, which rejects Demucs checkpoints
    because they pickle the model class, a `Fraction` segment and numpy metrics.
    Only use this exporter on checkpoints you trust (e.g. the official releases)."""
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def main():
    parser = argparse.ArgumentParser("export_safetensors", description=__doc__)
    parser.add_argument("name", help="pretrained model name or signature, e.g. htdemucs")
    parser.add_argument("out", type=Path, help="output .safetensors path")
    parser.add_argument("-r", "--repo", type=Path,
                        help="local model repo, instead of downloading from the remote repo")
    parser.add_argument("--dtype", choices=["f16", "f32"], default="f16",
                        help="storage dtype; released checkpoints are already stored as f16")
    args = parser.parse_args()
    with _trusted_checkpoints():
        model = get_model(args.name, repo=args.repo)
    export(model, args.out, args.dtype)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
