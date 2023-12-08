import click
import os
import pdb

import numpy as np
import streamlit as st
import torch

from imageio import imread
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import streamlit as st

from matplotlib import cm

from dolos.methods.xception_attention.networks.xception_attention import (
    load_xception_attention_model,
)
from dolos.methods.xception_attention.train_full_supervision import (
    MASK_SIZE,
    load_image,
)
from dolos.data import (
    RepaintCleanDataset,
    RepaintV2CleanDataset,
    CelebAHQProcessedDataset,
)
from dolos.methods.patch_forensics.predict import (
    get_best_model_path,
    get_predictions_path,
    save_pred_as_img,
    PREDICT_CONFIGS,
)


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
    dataset = PREDICT_CONFIGS[predict_config_name]["dataset"]
    device = "cuda"

    method_name = "xception-attention"
    model_dir = (
        f"output/xception-attention/{supervision}-supervision/{train_config_name}"
    )
    model_path = get_best_model_path(model_dir)

    model = load_xception_attention_model(init="random")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    num_images = len(dataset)
    mask_preds = np.zeros((num_images,) + MASK_SIZE)
    label_preds = np.zeros((num_images,))

    out_dir = os.path.join(
        "output",
        method_name,
        "{}-supervision".format(supervision),
        "predictions-train-config-{}-predict-config-{}".format(
            train_config_name, predict_config_name
        ),
    )

    for i in tqdm(range(num_images)):
        image = load_image(dataset, i)

        with torch.no_grad():
            image = image.to(device)
            label_pred, mask_pred, _ = model(image.unsqueeze(0))

        mask_pred = mask_pred[0, 0]
        mask_pred = mask_pred.detach().cpu().numpy()

        mask_preds[i] = mask_pred
        label_preds[i] = label_pred[0, 1].detach().cpu().item()

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
