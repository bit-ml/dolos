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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dolos.data import (
    CelebAHQDataset,
    RepaintP2CelebAHQDataset,
    RepaintP2CelebAHQ9KDataset,
)
from dolos.methods.patch_forensics.predict import (
    get_best_model_path,
    get_predictions_path,
    save_pred_as_img,
)
from dolos.methods.grad_cam.model import xceptionnet


def load_model(supervision, train_config_name, config, device):
    model_dir = f"output/grad-cam/{supervision}-supervision/{train_config_name}"
    model_path = get_best_model_path(model_dir)
    return xceptionnet(device, filename=model_path)


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
}


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

    method_name = "grad-cam"
    model = load_model(supervision, train_config_name, train_config, device)

    num_images = len(dataset)
    mask_size = (299, 299)

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

    targets = [ClassifierOutputTarget(1)]
    target_layers = [model.block11]
    cam = GradCAM(model=model, target_layers=target_layers)

    def get_label_and_mask_pred(image):
        mask_pred = cam(input_tensor=image.unsqueeze(0), targets=targets)
        mask_pred = mask_pred[0]
        # mask_pred = Image.fromarray(mask_pred).resize((256, 256), Image.BILINEAR)
        # mask_pred = np.array(mask_pred)
        mask_pred = mask_pred / mask_pred.max()
        label_pred = cam.outputs[0, 1]
        return label_pred, mask_pred

    for i in tqdm(range(num_images)):
        image = load_image(dataset, i, split="valid")
        image = image.to(device)

        label_pred, mask_pred = get_label_and_mask_pred(image)

        mask_preds[i] = mask_pred
        label_preds[i] = label_pred
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
