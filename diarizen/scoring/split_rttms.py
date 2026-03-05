from pathlib import Path
from collections import defaultdict

def split(input_path):
    # input_path = Path("/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/all_hyp.rttm")
    output_dir = Path(input_path).parent / "split_rttm"
    output_dir.mkdir(exist_ok=True)

    # Group lines by session
    sessions = defaultdict(list)
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.strip().split()
                if len(parts) >= 2:
                    session_id = parts[1]
                    sessions[session_id].append(line.strip())

    # Write one file per session
    print("", len(sessions), "sessions found")
    for session_id, lines in sessions.items():
        session_file = output_dir / f"{session_id}.rttm"
        with open(session_file, "w") as f:
            f.write("\n".join(lines))  # No trailing newline at the end
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split RTTM file into multiple files by session.")
    parser.add_argument("--input_path", type=str, help="Path to the input RTTM file.")
    args = parser.parse_args()
    # input_path = Path("/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1/all_hyp.rttm")

    split(Path(args.input_path))  # Call the split function with the provided path