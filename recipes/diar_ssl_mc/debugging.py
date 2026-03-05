import random
from pathlib import Path
import h5py
import torch
import numpy as np
import paderbox as pb
import soundfile as sf
from typing import Dict

from paderbox.io import dump_json, load_json

load_gcc_dir = Path("/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/standard_gcc/")
index = load_json(load_gcc_dir / 'index.json')
if not load_gcc_dir:
    raise FileNotFoundError(f"GCC directory {load_gcc_dir} does not exist")
file_stem = "ES2011a"
segment_id = f"ES2011a_393_401"
if file_stem not in index:
    raise KeyError(f"File stem '{file_stem}' not found in GCC index")
h5_filename = Path(index[file_stem])
if not h5_filename.is_absolute():
    h5_filename = load_gcc_dir / h5_filename
if not h5_filename.exists():
    raise FileNotFoundError(f"HDF5 file {h5_filename} not found")
with h5py.File(h5_filename, 'r') as f:
    if file_stem not in f:
        raise KeyError(f"Group '{file_stem}' not found in {h5_filename}")
    grp = f[file_stem]
    print(grp.keys()) # nur train daten liegen hier
    if segment_id not in grp:
        raise KeyError(f"Segment ID '{segment_id}' not found in group '{file_stem}'")
    gcc_features = grp[segment_id][()]