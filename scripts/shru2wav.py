"""Convert SHRU (.D23) files to WAV with TritonOA.

Usage: python shru2wav.py FILE.D23 [FILE.D23 ...] [--out DIR]
"""

import argparse
from pathlib import Path

from tritonoa.data.formats.shru import SHRUReader

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", type=Path)
parser.add_argument("--out", type=Path, default=Path("."))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

reader = SHRUReader()
for path in args.files:
    ds = reader.read(path)  # all records and channels; unconditioned, so ADC counts
    ds.data = ds.data / 2**23  # 24-bit counts -> ±1 full scale
    wav = args.out / f"{path.stem}.wav"
    ds.write_wav(wav)
    print(
        f"{wav}: {ds.data.shape[0]} channels, {ds.stats.sampling_rate:g} Hz, "
        f"start {ds.stats.time_init}"
    )
