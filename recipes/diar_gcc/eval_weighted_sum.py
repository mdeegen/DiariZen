import numpy as np
import torch
import toml
import argparse
import matplotlib.pyplot as plt

# import pyannote.audio
from diarizen.ckpt_utils import load_metric_summary
from diarizen.utils import instantiate
from pathlib import Path



if __name__ == '__main__':
    """Load the model from checkpoint an analyse the weights used in the weighted sum of the layers."""
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config-path", "-c", type=str, default=None, help="Pfad zur TOML Konfiguration")
    parser.add_argument("--metric", "-m", type=str, default="F1score", help="choose best metric")
    args = parser.parse_args()

    val_metric = args.metric
    avg_ckpt_num = 1

    # from diarizen.models.wavlm.wavlm_counting import Model
    exp_name = "wavlm_counting_pruned_att"
    config_dir = Path(f"/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_gcc/exp/{exp_name}")
    config_path = max(config_dir.glob("config__*.toml"), key=lambda p: p.stat().st_mtime).expanduser().absolute()
    print(config_path, flush=True) #  get newest config file
    config = toml.load(config_path.as_posix())
    ckpt_path = config_path.parent / 'checkpoints'
    val_metric_summary = config_path.parent / 'val_metric_summary.lst'

    model = instantiate(config['model']["path"], args=config['model']["args"])
    # model = Model(**config['model']["args"])

    val_metric_lst = load_metric_summary(val_metric_summary, ckpt_path)
    val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[val_metric])
    # best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
    segmentation = val_metric_lst_sorted[0] # [:avg_ckpt_num]
    ckp = segmentation["bin_path"]
    state_dict = torch.load(ckp, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    weights = model.weight_sum.weight.data.cpu().numpy().flatten()

    print("Weights for the weighted sum of the layers:")
    for i, weight in enumerate(weights):
        print(f"Layer {i}: {weight:.4f}")

    weights_abs = np.abs(weights)
    weights_abs_norm = weights_abs / weights_abs.sum()  # relative Relevanz via Betrag

    print("Weights for the weighted sum of the layers:")
    for i, w in enumerate(weights):
        print(
            f"Layer {i:2d}: raw={w:.4f}  |w|={weights_abs[i]:.4f}  rel_relevance={weights_abs_norm[i]:.4f}  sign={'+' if w >= 0 else '-'}")


    colors = ['steelblue' if w >= 0 else 'tomato' for w in weights]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw weights mit Vorzeichen-Farbe
    axes[0].bar(range(len(weights)), weights, color=colors)
    axes[0].set_title("Raw Weights per WavLM Layer\n(blue=positive, red=negative)")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Weight")
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Betrag-normierte Relevanz mit Vorzeichen-Farbe
    axes[1].bar(range(len(weights_abs_norm)), weights_abs_norm, color=colors)
    axes[1].set_title("Relevance per WavLM Layer (|w| normalized)\n(blue=positive, red=negative)")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Relative Relevance")

    plt.tight_layout()
    plt.savefig(f"{config_path.parent}/weights_{config_path.parent.name}.png", dpi=250)
    plt.show()
    print("Plot saved to weighted_sum_analysis.png")

