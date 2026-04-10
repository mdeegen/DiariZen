import os
import pandas as pd
import argparse


def shorten_key(key):
    """Shorten keys for readability in tables."""
    if key == "accuracy":
        key = "acc"
    if key == "accuracy_ov":
        key = "acc_ov"

    # Normalize common substrings
    if "OV-Time" in key:
        key = "ov_time"
    if "Total active Time" in key or "Total active time" in key:
        key = "act_time"

    # Class labels
    if key.strip().lower().startswith("class "):
        # e.g. "class 0" -> "c0"
        parts = key.strip().split()
        # parts[1] = parts[1].replace("_", "\_")
        if len(parts) == 2 and parts[1].isdigit():
            key = f"cl{parts[1]}_"

    if key == "precision_macro":
        key = "pre_macro"
    if key == "f1_weighted":
        key = "f1_weight"
    if key == "recall_macro":
        key = "rec_macro"

    # Generic: lower + spaces -> underscores, underscores -> makecell linebreak within same cell
    return r"{\makecell{" + key.strip().lower().replace("_", r"\\") + "}}"

    return key
def parse_metrics_file(path):
    metrics = {}
    metrics_cl = {}
    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":")
                if key in ["class-wise metrics", "num_classes", ]:
                    continue
                key = shorten_key(key)

                if "=" in val:
                    vals = val.strip().split(",")
                    metrics_cl[key.strip().lower()[:-2] + "f1}}"] = float(vals[0].split("=")[-1].strip())
                    metrics_cl[key.strip().lower()[:-2] + "pre}}"] = float(vals[1].split("=")[-1].strip())
                    metrics_cl[key.strip().lower()[:-2] + "rec}}"] = float(vals[2].split("=")[-1].strip())
                else:
                    metrics[key.strip().lower()] = float(val.strip())
    return metrics, metrics_cl


def collect_experiment(exp_path):
    results = []
    results_cl = []
    exp_path = exp_path + "/infer_oracle_clustering/metric_F1score_best/avg_ckpt1/test_marc"
    for dataset in os.listdir(exp_path):
        dataset_path = os.path.join(exp_path, dataset)
        metrics_file = os.path.join(dataset_path, "metrics.txt")

        if not os.path.isfile(metrics_file):
            print("Warning: COuldnt find metrics file", flush=True)
            continue

        metrics, metrics_cl = parse_metrics_file(metrics_file)
        metrics["dataset"] = dataset
        metrics_cl["dataset"] = dataset
        results.append(metrics)
        results_cl.append(metrics_cl)

    return pd.DataFrame(results), pd.DataFrame(results_cl)


def to_wide_format(df, exp_name):

    exp_name = exp_name.replace("_", r"\_")
    df["experiment"] = exp_name
    df_wide = df.pivot(index="experiment", columns="dataset")
    df_wide.columns = [f"{d}_{m}" for m, d in df_wide.columns]
    return df_wide.reset_index()

def per_dataset(df_long, experiment_root, cl=False):

    for dataset in df_long["dataset"].unique():
        df_sub = df_long[df_long["dataset"] == dataset]

        # if not cl:
        #     # df_sub = df_sub.sort_values("\makecell{f1\\macro}")#, ascending=False)
        #     col = next(c for c in df_sub.columns if ("f1" in str(c).lower() and "macro" in str(c).lower()))
        #     df_sub = df_sub.sort_values(col)  # ggf. ascending=False
        df_sub = df_sub.sort_values("dataset")

        n_cols = len(df_sub.columns)
        col_format = "S" * (n_cols - 2) + "ll"

        if cl:
            latex = df_sub.to_latex(
                index=False,
                float_format="%.4f",
                bold_rows=False,
                column_format=col_format,
                escape=False,
                caption=None,
                label=None
            )
        else:
            latex = df_sub.to_latex(
                index=False,
                float_format="%.4f",
                # caption=f"Results on {dataset}",
                # label=f"tab:{dataset}",
                bold_rows=False,
                column_format=col_format,
                escape=False,
                caption=None,
                label=None,
            )
        if cl:
            with open(f"{experiment_root}/{dataset}_cl.tex", "w") as f:
                f.write(latex)
        else:
            with open(f"{experiment_root}/{dataset}.tex", "w") as f:
                f.write(latex)



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_dir", "-e", type=str, required=True)
    exp_name = arg_parser.parse_args().exp_dir

    experiment_root = os.path.join("/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/recipes/diar_gcc/exp/", exp_name)
    print("Aggregating metrics for ", exp_name, flush=True)
    df_long, df_long_cl = collect_experiment(experiment_root)
    df_wide = to_wide_format(df_long, exp_name)

    print("LONG FORMAT:")
    print(df_long)

    # print("\nWIDE FORMAT:")
    # print(df_wide)

    # save
    df_long.to_csv(experiment_root + "results_long.csv", index=False)
    df_long_cl.to_csv(experiment_root + "results_long_cl.csv", index=False)
    # df_wide.to_csv(experiment_root + "results_wide.csv", index=False)
    # df_wide.to_latex(
    #     experiment_root + "/results_wide.tex",
    #     index=False,
    #     float_format="%.4f",
    #     caption="Metrics on test set",
    #     label="table:wide",
    #     bold_rows=False,
    #     # column_format="lcccc",  # anpassen je nach metrics
    #     escape=False
    # )
    n_cols = len(df_long.columns)
    col_format = "S" * (n_cols - 2) + "ll"
    df_long.to_latex(
        experiment_root + "/results_long.tex",
        index=False,
        float_format="%.4f",
        caption=f"Eval metrics on test set: {exp_name}",
        label="table:long",
        bold_rows=False,
        column_format=col_format,
        escape=False
    )
    n_cols = len(df_long_cl.columns)
    col_format = "S" * (n_cols - 2) + "ll"
    df_long_cl.to_latex(
        experiment_root + "/results_long_cl.tex",
        index=False,
        float_format="%.4f",
        label="table:long_cl",
        bold_rows=False,
        column_format=col_format,
        escape=False
    )

    per_dataset(df_long, experiment_root)
    per_dataset(df_long_cl, experiment_root, cl=True)
