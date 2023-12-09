import click
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from matplotlib import cm
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image

from dolos.data import (
    CelebAHQDataset,
    RepaintP2CelebAHQDataset,
    RepaintP2CelebAHQ9KDataset,
    RepaintLDMCelebAHQDataset,
    LamaDataset,
    PluralisticDataset,
)
from dolos.methods.patch_forensics.networks.customnet import make_patch_xceptionnet
from dolos.methods.patch_forensics.train_full_supervision import (
    get_mask_size,
)


def get_best_model_path(model_dir):
    def get_score(filename):
        _, score = filename.split("=")
        *score, _ = score.split(".")
        return float(".".join(score))

    filenames = os.listdir(model_dir)
    filename = sorted(filenames, key=get_score, reverse=True)[0]
    return os.path.join(model_dir, filename)


def load_model_from_path(model_path, config, device):
    model = make_patch_xceptionnet(
        layername=config["last-layer"], frontend=config["frontend"]
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_model(supervision, train_config_name, config, device):
    model_dir = f"output/patch-forensics/{supervision}-supervision/{train_config_name}"
    model_path = get_best_model_path(model_dir)
    return load_model_from_path(model_path, config, device)


def get_predictions_path(
    method_name, supervision, train_config_name, predict_config_name
):
    return os.path.join(
        "output",
        method_name,
        "{}-supervision".format(supervision),
        "predictions-train-config-{}-predict-config-{}.npz".format(
            train_config_name, predict_config_name
        ),
    )


PREDICT_CONFIGS = {
    "celebahq-valid": {
        "dataset": CelebAHQDataset("valid"),
    },
    "celebahq-test": {
        "dataset": CelebAHQDataset("test"),
    },
    "repaint-p2-valid": {
        "dataset": RepaintP2CelebAHQDataset("valid"),
    },
    "repaint-p2-test": {
        "dataset": RepaintP2CelebAHQDataset("test"),
    },
    "repaint-p2-9k-valid": {
        "dataset": RepaintP2CelebAHQ9KDataset("valid"),
    },
    "repaint-p2-9k-test": {
        "dataset": RepaintP2CelebAHQ9KDataset("test"),
    },
    "repaint-ldm-valid": {
        "dataset": RepaintLDMCelebAHQDataset("valid"),
    },
    "repaint-ldm-test": {
        "dataset": RepaintLDMCelebAHQDataset("test"),
    },
    "lama-valid": {
        "dataset": LamaDataset("valid"),
    },
    "lama-test": {
        "dataset": LamaDataset("test"),
    },
    "pluralistic-valid": {
        "dataset": PluralisticDataset("valid"),
    },
    "pluralistic-test": {
        "dataset": PluralisticDataset("test"),
    },
}


def save_pred_as_img(dataset, i, pred, out_dir):
    key = dataset.get_file_name(i)
    out_img_path = os.path.join(out_dir, key + ".png")
    os.makedirs(out_dir, exist_ok=True)
    pred_color = cm.viridis(pred)
    pred_color = (255 * pred_color).astype("uint8")
    pred_image = Image.fromarray(pred_color)
    pred_image = pred_image.resize((256, 256), Image.Resampling.NEAREST)
    pred_image.save(out_img_path)


@click.command()
@click.option("-s", "supervision", type=click.Choice(["full", "weak"]), required=True)
@click.option("-t", "train_config_name", required=True)
@click.option("-p", "predict_config_name", required=True)
@click.option("-v", "to_visualize", is_flag=True, default=False)
@click.option("--save-images", "to_save_images", is_flag=True, default=False)
def main(
    supervision,
    train_config_name,
    predict_config_name,
    to_visualize=False,
    to_save_images=False,
):
    if supervision == "full":
        from dolos.methods.patch_forensics.train_full_supervision import CONFIGS
    elif supervision == "weak":
        from dolos.methods.patch_forensics.train_weak_supervision import CONFIGS
    else:
        assert False, "Unknown supervision type"

    train_config = CONFIGS[train_config_name]
    dataset = PREDICT_CONFIGS[predict_config_name]["dataset"]
    device = "cuda"

    method_name = "patch-forensics"
    model = load_model(supervision, train_config_name, train_config, device)

    num_images = len(dataset)
    mask_size = get_mask_size(train_config)
    mask_preds = np.zeros((num_images,) + mask_size)
    label_preds = np.zeros((num_images,))

    load_image = train_config["load-image"]

    out_dir = os.path.join(
        "output",
        method_name,
        "{}-supervision".format(supervision),
        "predictions-train-config-{}-predict-config-{}".format(
            train_config_name, predict_config_name
        ),
    )

    for i in tqdm(range(num_images)):
        image = load_image(dataset, i, split="valid")

        with torch.no_grad():
            image = image.to(device)
            mask_pred = model(image.unsqueeze(0))

        mask_pred = F.softmax(mask_pred, dim=1)
        mask_pred = mask_pred[0, 1]
        mask_pred = mask_pred.detach().cpu().numpy()

        mask_preds[i] = mask_pred
        label_preds[i] = np.mean(mask_pred)
        # TODO Maybe use max for the max-pooled version?!

        # save image out
        if to_save_images:
            save_pred_as_img(dataset, i, mask_pred, out_dir)

        if to_visualize:
            fig, axs = plt.subplots(nrows=1, ncols=2)

            mask_true = dataset.load_mask(i)

            axs[0].imshow(mask_true)
            axs[0].set_title("true")
            axs[0].set_axis_off()

            axs[1].imshow(mask_pred)
            axs[1].set_title("pred")
            axs[1].set_axis_off()

            st.pyplot(fig)
            pdb.set_trace()

    output_path = get_predictions_path(
        method_name, supervision, train_config_name, predict_config_name
    )
    np.savez(output_path, label_preds=label_preds, mask_preds=mask_preds)


if __name__ == "__main__":
    main()
