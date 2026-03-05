import os
from pathlib import Path
from paderbox.io.json_module import load_json, dump_json
import subprocess




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Accuracy over the dataset.")
    parser.add_argument('--storage_dir', type=str, required=True, help='Directory where results are stored.')
    args = parser.parse_args()

    storage_dir = args.storage_dir

    results = {}
    accuracies = []
    ov = []
    storage_dir = Path(storage_dir)
    out_path = storage_dir / "total_accuracy.json"
    for file in os.listdir(storage_dir):
        if file.endswith("accuracy.json") and file != "total_accuracy.json":
            file_path = storage_dir / file
            data = load_json(file_path)
            results[file] = data
            accuracies.append(data["accuracy"])
            ov.append(data["accuracy_ov"])
    total_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    ov_accuracy = sum(ov) / len(ov) if ov else 0.0
    results["total_accuracy"] = {"accuracy": total_accuracy, "ov_accuracy": ov_accuracy}
    print(f"Total Accuracy: {total_accuracy:.4f}, Results saved to {out_path}")
    print(f"Total Accuracy: {ov_accuracy:.4f}, Results saved to {out_path}")
    dump_json(results, out_path)