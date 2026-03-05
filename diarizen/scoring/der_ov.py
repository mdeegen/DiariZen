import os
from pathlib import Path
from paderbox.io.json_module import load_json
import subprocess

def check_rttm(input_path, output_path):
    with open(input_path, "r") as infile:
        lines = [line.strip() for line in infile if line.strip()]  # alle nicht-leeren Zeilen

    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            print(f"⚠️ Warnung: Zeile mit weniger als 9 Spalten übersprungen: {line}")
            continue
        new_lines.append(" ".join(parts[:9]))

    with open(output_path, "w") as outfile:
        outfile.write("\n".join(new_lines))
    return

def compute_der_ov(der, total, storage_dir, act_storage, ref_storage, collar, rank=None):
    outdir = str(storage_dir)
    if rank is not None:
        os.rename(outdir + f"/all_hyp_{rank}_dscore.json",
                  outdir + f"/all_hyp_{rank}_dscore_all_regions.json")
        os.rename(outdir + f"/all_hyp_{rank}_dscore_per_reco.json",
                  outdir + f"/all_hyp_{rank}_dscore_per_reco_all_regions.json")
    else:
        os.rename(outdir + f"/all_hyp_dscore.json",
                  outdir + f"/all_hyp_dscore_all_regions.json")
        os.rename(outdir + f"/all_hyp_dscore_per_reco.json",
                  outdir + f"/all_hyp_dscore_per_reco_all_regions.json")

    der_cmd = f"python -m meeteval.der dscore -h {act_storage} -r {ref_storage} --collar {collar} --regions nooverlap"
    # print(der_cmd)
    try:
        with open(os.devnull, "w") as fnull:
            subprocess.run(der_cmd, shell=True, stdout=fnull, stderr=fnull, check=True)
    except Exception as e:
        print(f"Warning: DER calculation failed for {act_storage} / {ref_storage}, returning 0", e)
        # assert False, der_cmd
        return der, 0
    if rank is not None:
        if not Path(storage_dir / f"all_hyp_{rank}_dscore.json").exists():
            print(f"Warning: JSON file does not exist {act_storage} / {ref_storage}, returning 0")
            return der, 0
        der_output = load_json(Path(storage_dir / f"all_hyp_{rank}_dscore.json"))
    else:
        der_output = load_json(Path(storage_dir / f"all_hyp_dscore.json"))
    der_s = der_output["error_rate"]
    nooverlap = der_output["scored_speaker_time"]
    overlap_time = total - nooverlap
    tau_ov = overlap_time / total
    tau_s = nooverlap / total
    der_ov = (der - tau_s * der_s) / (tau_ov + 1e-13)
    return der_ov, der_s


def compute_der(storage_dir, ref, collar, rank=None):
    if rank is not None:
        act_storage = str(Path(storage_dir) / f"all_hyp_{rank}.rttm")
    else:
        act_storage = str(Path(storage_dir) / f"all_hyp.rttm")
    ref_storage = str(ref)
    outdir = Path(storage_dir) / f".cache/"
    outdir = Path(outdir).resolve()
    os.makedirs(outdir, exist_ok=True)
    der_cmd = f"cd {outdir} && python -m meeteval.der dscore -h {act_storage} -r {ref_storage} --collar {collar}" #  --output {str(outdir)}"
    failed = False
    try:
        result = subprocess.run(der_cmd, shell=True, check=True, capture_output=True, text=True)
        # result = subprocess.run(["bash", "-i", "-c", der_cmd], check=True)
    except Exception as e:
        # print(e)
        # print(f"WARNING WARNING: DER computation failed, continuing without raising. Command: {der_cmd}")
        failed = True
        # assert False
    if failed:
        return 0, 0, 0, 0, 0, 0
    if rank is not None:
        der_output = load_json(storage_dir / f"all_hyp_{rank}_dscore.json")
    else:
        der_output = load_json(storage_dir / f"all_hyp_dscore.json")

    der = der_output["error_rate"]
    total = der_output["scored_speaker_time"]
    fa = der_output["falarm_speaker_time"] / total
    miss = der_output["missed_speaker_time"] / total
    conf = der_output["speaker_error_time"] / total
    der_ov, der_s = compute_der_ov(der, total, storage_dir, act_storage, ref_storage, collar, rank=rank)
    return der, der_ov, der_s, fa, miss, conf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute DER and overlap DER for diarization results.")
    parser.add_argument('--storage_dir', type=str, required=True, help='Directory where results are stored.')
    parser.add_argument('--ref', type=str, required=True, help='Directory where ref lies.')
    parser.add_argument('--collar', type=str, required=True, help='Collar for der.')
    args = parser.parse_args()

    storage_dir = args.storage_dir
    ref = args.ref
    collar = args.collar
    results = {}
    storage_dir = Path(storage_dir)
    out_path = storage_dir / "all_hyp.rttm"
    # os.makedirs(out_path, exist_ok=True)
    rttm_files = [f for f in storage_dir.glob("*.rttm") if f.name not in ("all_hyp.rttm", "referenz.rttm")]

    with open(out_path, "w") as outfile:
        for i, rttm_file in enumerate(rttm_files):
            with open(rttm_file, "r") as infile:
                content = infile.read().rstrip()
                outfile.write(content)
                if i < len(rttm_files) - 1:
                    outfile.write("\n")

    check_rttm(ref, storage_dir/"referenz.rttm")
    ref = storage_dir/"referenz.rttm"
    out_file = Path(storage_dir) / f"results{collar}.json"
    der, der_ov, der_s, fa, miss, conf = compute_der(storage_dir, ref, collar)
    with open(out_file, "w") as f:
        header = "File                 DER     Miss     FA       SpkE     DER_ov    DER_s"
        separator = "-" * len(header)
        row = "{:<18}{:7.3f}  {:7.3f}  {:6.3f}  {:7.3f}  {:8.3f}  {:7.3f}".format(
            "*** OVERALL ***", der, miss, fa, conf, der_ov, der_s
        )

        f.write(header + "\n")
        f.write(separator + "\n")
        f.write(row + "\n")

    print(f"Ergebnisse gespeichert in: {out_file}")
