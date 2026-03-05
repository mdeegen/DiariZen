

import argparse
import toml
import json
from tqdm import tqdm
from diarizen.utils import instantiate


def main(config, out_dir):

    model_num_frames=399
    model_rf_duration=0.025
    model_rf_step=0.02
    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step

    test_dataset_config = config["test_dataset"]["args"]
    test_dataset_config["model_num_frames"] = model_num_frames
    test_dataset_config["model_rf_duration"] = model_rf_duration
    test_dataset_config["model_rf_step"] = model_rf_step

    train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)
    ov_train = []
    print("GO", len(train_dataset))
    for d in tqdm(train_dataset, desc="Compute ov for training set"):
        ov_train.append(d[4])
    print(sum(x == 1 for x in ov_train), sum(x == 0 for x in ov_train))
    print(sum(x == 1 for x in ov_train)/len(train_dataset), sum(x == 0 for x in ov_train)/len(train_dataset))
    print("Train finished", len(ov_train))

    validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
    ov_dev = []
    print("GO", len(validate_dataset))
    for d in tqdm(validate_dataset, desc="Compute ov for dev set"):
        ov_dev.append(d[4])
    print(sum(x == 1 for x in ov_dev), sum(x == 0 for x in ov_dev))
    print(sum(x == 1 for x in ov_dev)/len(validate_dataset), sum(x == 0 for x in ov_dev)/len(validate_dataset))
    print("Dev finished", len(ov_dev))

    # test_dataset = instantiate(config["test_dataset"]["path"], args=test_dataset_config)
    # ov_test = []
    # print("GO", len(test_dataset))
    # for d in tqdm(test_dataset, desc="Compute ov for test set"):
    #     ov_test.append(d[4])
    # print("TEST finished", len(ov_test))

    labels = {
        "train": ov_train,
        "dev": ov_dev,
        # "test": ov_test
    }
    with open(out_dir, "w") as f:
        json.dump(labels, f)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c", "--config", type=Path, required=True, help="Path to the config file."
    # )
    # args = parser.parse_args()
    # config = args.config
    config = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/conf/ov.toml"
    out_dir = "/mnt/matylda5/qdeegen/deploy/forschung/DiariZen/recipes/diar_ssl_mc/data_no_chime/ov.json"
    config = toml.load(config)

    # with open(out_dir, "r") as f:
    #     labels = json.load(f)
    # print(labels.keys())
    main(config, out_dir)