# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run a released model with PyTorch on an excerpt of an audio file and save
the input and output for the Go TestPretrained check.

Usage:
    python3 go/tools/make_pretrained_reference.py htdemucs test.mp3 reference.safetensors
    DEMUCS_TEST_MODEL=htdemucs.safetensors DEMUCS_TEST_REFERENCE=reference.safetensors \\
        go test -run Pretrained -v ./go
"""
import argparse
from pathlib import Path
import sys
import time

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demucs.pretrained import get_model  # noqa: E402
from export_safetensors import _trusted_checkpoints, write_safetensors  # noqa: E402
from make_testdata import normalized_apply  # noqa: E402


def main():
    parser = argparse.ArgumentParser("make_pretrained_reference", description=__doc__)
    parser.add_argument("name", help="pretrained model name, e.g. htdemucs")
    parser.add_argument("audio", type=Path, help="input audio (any format libsndfile reads)")
    parser.add_argument("out", type=Path)
    parser.add_argument("-r", "--repo", type=Path, help="local model repo")
    parser.add_argument("--seconds", type=float, default=10.5,
                        help="length of the excerpt, from the start of the file")
    args = parser.parse_args()

    with _trusted_checkpoints():
        model = get_model(args.name, repo=args.repo)
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    if sr != model.samplerate:
        sys.exit(f"{args.audio} is {sr} Hz, the model needs {model.samplerate} Hz")
    mix = torch.from_numpy(wav.T.copy())[:, :int(args.seconds * sr)]
    if mix.shape[0] == 1:
        mix = mix.repeat(2, 1)
    start = time.time()
    with torch.no_grad():
        out = normalized_apply(model, mix, shifts=0, split=True, overlap=0.25)
    print(f"torch: {mix.shape[-1] / sr:.1f}s of audio in {time.time() - start:.1f}s")
    write_safetensors(args.out, {"mix": mix, "out": out}, {}, "f32")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
