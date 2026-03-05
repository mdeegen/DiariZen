import argparse
import json
from pathlib import Path
from paderbox.io import load_json, dump_json

def merge_json_indexes(json_files, output_json):
    master_index = {}

    for jf_path in json_files:
        index = load_json(jf_path)
        master_index.update(index)

    dump_json(master_index, output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Path to out_dir")
    args = parser.parse_args()
    out_dir = Path(args.out_dir).expanduser().absolute()
    # out_dir = Path("/mnt/matylda5/qdeegen/data/AMI_AIS_ALI_NSF_CHiME7/data/gccs") /  f"standard_gcc/"


    # out_dir = Path("/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data/gccs/gcc_size2048_shift316/train")

    index_files = list(out_dir.glob('*.json'))
    merge_json_indexes(index_files, out_dir / 'index.json')