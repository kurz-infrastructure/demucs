# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Generate the reference data used by the Go tests in go/testdata.

A tiny, randomly initialized HTDemucs (and a bag of two) is exported along
with inputs and PyTorch outputs, so the Go port can be checked against the
reference implementation without downloading the released weights.

Usage:
    python3 go/tools/make_testdata.py go/testdata
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demucs.apply import BagOfModels, apply_model  # noqa: E402
from demucs.htdemucs import HTDemucs  # noqa: E402
from demucs.spec import ispectro, spectro  # noqa: E402
from export_safetensors import export, write_safetensors  # noqa: E402

SAMPLERATE = 44100


def tiny_model(seed):
    torch.manual_seed(seed)
    model = HTDemucs(
        sources=["drums", "vocals"],
        channels=4,
        dconv_comp=4,
        depth=4,
        # Use GroupNorm in the two innermost encoder/decoder layers too.
        norm_starts=2,
        dconv_mode=3,
        bottom_channels=32,
        t_layers=3,
        t_heads=4,
        t_dropout=0.0,
        segment=Fraction(1, 4),
        samplerate=SAMPLERATE,
    )
    # Randomize the scales and affine parameters, otherwise the residual
    # branches start almost silent and the comparison would not exercise them.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".scale"):
                param.uniform_(0.3, 1.0)
            elif "norm" in name and name.endswith(".weight"):
                param.normal_(1.0, 0.2)
            elif "norm" in name and name.endswith(".bias"):
                param.normal_(0.0, 0.2)
            # Round to what the f16 export stores, so the reference is exact.
            param.copy_(param.half().float())
    model.eval()
    return model


def normalized_apply(model, mix, **kwargs):
    """Mirror demucs.api.Separator.separate_tensor."""
    wav = mix.clone()
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std() + 1e-8
    out = apply_model(model, wav[None], **kwargs)[0]
    out *= ref.std() + 1e-8
    out += ref.mean()
    return out


class FixedShifts:
    """Replace random.randint in apply_model by a known sequence."""

    def __init__(self, offsets):
        self.offsets = list(offsets)
        self.original = random.randint

    def __enter__(self):
        offsets = iter(self.offsets)
        random.randint = lambda a, b: next(offsets)
        return self

    def __exit__(self, *exc):
        random.randint = self.original


def main():
    parser = argparse.ArgumentParser("make_testdata", description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = tiny_model(0)
    other = tiny_model(1)
    bag = BagOfModels([model, other], weights=[[1.0, 0.25], [0.5, 1.0]])
    export(model, args.out / "tiny.safetensors", dtype="f16")
    export(bag, args.out / "tiny_bag.safetensors", dtype="f16")

    torch.manual_seed(1234)
    seg = int(model.segment * SAMPLERATE)
    length = int(seg * 1.6)
    t = torch.arange(length) / SAMPLERATE
    # A few tones plus noise, with different content on each channel.
    mix = torch.stack([
        0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.05 * torch.randn(length),
        0.2 * torch.sin(2 * torch.pi * 330 * t + 1) + 0.05 * torch.randn(length),
    ])

    tensors = {"mix": mix}
    with torch.no_grad():
        # Raw forward on one full segment and on a shorter, zero padded one.
        tensors["forward_full"] = model(mix[None, :, :seg])[0]
        short = seg - 1000
        tensors["forward_short"] = model(mix[None, :, :short])[0]
        tensors["apply_split"] = normalized_apply(
            model, mix, shifts=0, split=True, overlap=0.25)
        offsets = [1234, 20000]
        with FixedShifts(offsets):
            tensors["apply_shifts"] = normalized_apply(
                model, mix, shifts=2, split=True, overlap=0.25)
        with FixedShifts([0, 0]):
            tensors["apply_bag"] = normalized_apply(
                bag, mix, shifts=0, split=True, overlap=0.25)
        z = spectro(mix[:1], 4096, 1024)
        tensors["stft_re"] = z.real
        tensors["stft_im"] = z.imag
        tensors["istft"] = ispectro(z, 1024, length=length)
        tensors["stft_input"] = mix[:1]

    metadata = {"shift_offsets": json.dumps(offsets), "forward_short_length": str(short)}
    write_safetensors(args.out / "reference.safetensors", tensors, metadata, "f32")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
