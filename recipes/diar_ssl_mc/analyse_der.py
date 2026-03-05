import os
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import matplotlib.patches as mpatches


def load_rttm_lines(file_path, session_id=None):
    """
    Lädt RTTM und gibt nur die Zeilen für session_id zurück (oder alle, wenn None).
    """
    annotation = Annotation()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != "SPEAKER":
                continue
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            if (session_id is None) or (file_id == session_id):
                annotation[Segment(start, start + duration)] = speaker
    return annotation


def plot_diarization(ref, hyp, out_path):
    metric = DiarizationErrorRate()
    metric(ref, hyp)
    details = metric(reference=ref, hypothesis=hyp, detailed=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    y_ref, y_hyp, y_err = 2, 1, 0  # Y-Positionen für die Spuren

    # Farben für Speaker
    speakers = list(set(ref.labels() + hyp.labels()))
    speaker_colors = {spk: plt.cm.tab10(i % 10) for i, spk in enumerate(speakers)}

    # --- Ground Truth Spur ---
    for segment, _, speaker in ref.itertracks(yield_label=True):
        ax.barh(y_ref, segment.end - segment.start, left=segment.start,
                color=speaker_colors[speaker], edgecolor='black', height=0.4)

    # --- Hypothesis Spur ---
    for segment, _, speaker in hyp.itertracks(yield_label=True):
        ax.barh(y_hyp, segment.end - segment.start, left=segment.start,
                color=speaker_colors[speaker], edgecolor='black', height=0.4)

    # --- Fehler Spur ---
    error_colors = {
        'correct': 'white',
        'missed speech': 'red',
        'false alarm': 'blue',
        'confusion': 'orange'
    }

    for segment, detail in details.items():
        # Falls detail nur ein float ist, wird es übersprungen
        if not isinstance(detail, dict) or 'type' not in detail:
            continue
        err_type = detail['type']
        ax.barh(y_err, segment.end - segment.start, left=segment.start,
                color=error_colors.get(err_type, 'gray'), edgecolor='black', height=0.4)

    ax.set_xlabel("Time (s)")
    ax.set_yticks([y_ref, y_hyp, y_err])
    ax.set_yticklabels(["Reference", "Hypothesis", "Errors"])

    # Legenden
    legend_speakers = [mpatches.Patch(color=color, label=spk) for spk, color in speaker_colors.items()]
    legend_errors = [mpatches.Patch(color=color, label=etype) for etype, color in error_colors.items()]
    ax.legend(handles=legend_speakers + legend_errors, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main(ref_file, hyp_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hyp_files = [f for f in os.listdir(hyp_dir) if f.endswith(".rttm")]

    for hyp_name in hyp_files:
        print(f"Processing: {hyp_name}")
        session_id = hyp_name.replace(".rttm", "")
        ref = load_rttm_lines(ref_file, session_id)
        hyp = load_rttm_lines(os.path.join(hyp_dir, hyp_name))

        if len(ref) == 0:
            print(f"WARNING: Keine Referenz für {session_id} gefunden.")
            continue

        out_path = os.path.join(out_dir, f"{session_id}_timeline.png")
        plot_diarization(ref, hyp, out_path)


if __name__ == "__main__":
    ref_file = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_mc/test_marc/NOTSOFAR1/rttm"
    hyp_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/exp/baseline_sc/infer_oracle_clustering/metric_Loss_best/avg_ckpt5/test_marc/NOTSOFAR1"
    out_dir = "plots_detailed_notsofar"
    main(ref_file, hyp_dir, out_dir)
