import random
import os
from collections import defaultdict
from math import ceil


def create_balanced_subset_with_meta(
        scp_path, uem_path, rttm_path,
        subset_ratio=0.05, output_dir="subset", seed=42,
        corpus_list=("AMI", "AISHELL4", "AliMeeting", "NOTSOFAR")
):
    """
    Erstellt ein balanciertes Subset (SCP + RTTM + UEM) aus großen Dateien.
    Corpus-Erkennung erfolgt anhand des Pfads.

    Args:
        scp_path: Pfad zur großen SCP-Datei
        rttm_path: Pfad zur RTTM-Datei
        uem_path: Pfad zur UEM-Datei
        subset_ratio: Anteil des Subsets (z.B. 0.05 für 5%)
        output_dir: Ordner für Subset-Dateien
        seed: Zufallsseed
        corpus_list: Liste der Corpora
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Lade SCP-Datei
    with open(scp_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # Gruppiere nach Corpus
    corpus_entries = defaultdict(list)
    for line in lines:
        utt_id, wav_path = line.split(maxsplit=1)
        corpus_name = None
        for name in corpus_list:
            if f"{name}" in wav_path:
                corpus_name = name
                break
        if corpus_name:
            corpus_entries[corpus_name].append((utt_id, wav_path))

    # Gesamtgröße
    total = sum(len(v) for v in corpus_entries.values())
    target_total = ceil(total * subset_ratio)
    print(f"Total entries: {total}, Target subset size: {target_total}")

    # Zielgröße pro Corpus
    corpus_targets = {
        k: max(1, ceil(len(v) / total * target_total)) for k, v in corpus_entries.items()
    }
    print("Target per corpus:", corpus_targets)

    selected_utts = set()
    combined_subset = []

    # Sampling pro Corpus
    for corpus, entries in corpus_entries.items():
        target_size = min(corpus_targets[corpus], len(entries))
        sampled = random.sample(entries, target_size)

        out_path = os.path.join(output_dir, f"{corpus}_subset.scp")
        with open(out_path, "w") as f:
            for utt_id, wav_path in sampled:
                f.write(f"{utt_id} {wav_path}\n")
        print(f"[{corpus}] {len(sampled)} → {out_path}")

        for utt_id, _ in sampled:
            selected_utts.add(utt_id)
            combined_subset.append(f"{utt_id} {_}")

    # Kombinierte SCP-Datei
    combined_scp = os.path.join(output_dir, "wav.scp")
    with open(combined_scp, "w") as f:
        f.write("\n".join(combined_subset) + "\n")
    print(f"Combined SCP saved at: {combined_scp}")

    # ---- RTTM und UEM filtern ----
    print("Filtering RTTM and UEM...")
    selected_prefixes = {utt.split()[0] for utt in combined_subset}

    with open(rttm_path, "r") as f:
        rttm_lines = [l.strip() for l in f if l.strip()]
    subset_rttm = [l for l in rttm_lines if l.split()[1] in selected_prefixes]

    rttm_out = os.path.join(output_dir, "rttm")
    with open(rttm_out, "w") as f:
        f.write("\n".join(subset_rttm) + "\n")
    print(f"Subset RTTM saved at: {rttm_out} ({len(subset_rttm)} lines)")

    # UEM
    with open(uem_path, "r") as f:
        uem_lines = [l.strip() for l in f if l.strip()]
    subset_uem = [l for l in uem_lines if l.split()[0] in selected_prefixes]

    uem_out = os.path.join(output_dir, "all.uem")
    with open(uem_out, "w") as f:
        f.write("\n".join(subset_uem) + "\n")
    print(f"Subset UEM saved at: {uem_out} ({len(subset_uem)} lines)")


if __name__ == "__main__":
    working_dir = "/mnt/scratch/tmp/qdeegen/AMI_AIS_ALI_NSF_CHiME7/data"
    subset_ratio = 0.3  # 20% Subset
    dataset = "dev"

    create_balanced_subset_with_meta(
        scp_path=f"{working_dir}/{dataset}/wav.scp",
        uem_path=f"{working_dir}/{dataset}/all.uem",
        rttm_path=f"{working_dir}/{dataset}/rttm",
        subset_ratio=subset_ratio,
        output_dir=f"{working_dir}/{dataset}_{subset_ratio*100:.0f}_percent_subset",
    )
