import os
from pathlib import Path
from paderbox.io.json_module import load_json, dump_json
import subprocess
import numpy as np




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Accuracy over the dataset.")
    parser.add_argument('--storage_dir', type=str, required=True, help='Directory where results are stored.')
    args = parser.parse_args()

    acc_list = []
    ov_acc = []
    res = {}
    storage_dir = args.storage_dir
    for dset in os.listdir(storage_dir):
        dset_path = os.path.join(storage_dir, dset)
        out_path = Path(dset_path) / "total_accuracy.json"
        acc = load_json(out_path)
        acc_list.append(acc["total_accuracy"]["accuracy"])
        ov_acc.append(acc["total_accuracy"]["ov_accuracy"])
    res["Accuracy"] = np.mean(acc_list)
    res["Ov_Accuracy"] = np.mean(ov_acc)
    print(f"Macro Accuracy: {res['Accuracy']:.4f}")
    print(f"Macro Ov_Accuracy: {res['Ov_Accuracy']:.4f}")
    out_path = Path(storage_dir) / "macro_accuracy.json"
    dump_json(res, out_path)