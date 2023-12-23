import json
import pdb
import random

import click

import numpy as np
import streamlit as st

from PIL import Image
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from dolos.methods.patch_forensics.predict import get_predictions_path, PREDICT_CONFIGS
from dolos.metrics.iou_ignite import iou


SEED = 1337
random.seed(SEED)


def evaluate_detection(
    method_name, supervision, train_config_name, dataset_name, to_visualize
):
    def load_predictions(*args):
        return np.load(get_predictions_path(*args))["label_preds"]

    CELEBAHQ_FAKE_DATASETS = {
        "repaint-p2",
    }

    assert dataset_name in CELEBAHQ_FAKE_DATASETS

    args = method_name, supervision, train_config_name
    pred_real = load_predictions(*args, "celebahq-test")
    pred_fake = load_predictions(*args, dataset_name + "-test")

    num_real = len(pred_real)
    num_fake = len(pred_fake)
    num = min(num_real, num_fake)

    def np_sample(xs, n):
        return np.array(random.sample(list(xs), n))

    def compute_ap1():
        pred_real_ss = np_sample(pred_real, num)
        pred_fake_ss = np_sample(pred_fake, num)

        true = np.hstack((np.zeros(num), np.ones(num)))
        pred = np.hstack((pred_real_ss, pred_fake_ss))

        return average_precision_score(true, pred)

    aps = [100 * compute_ap1() for _ in range(5)]
    ap_mean = np.mean(aps)
    ap_std = np.std(aps)

    print("Detecion · average precision:")
    print("{:.2f}±{:.1f}".format(ap_mean, 2 * ap_std))


def evaluate_localisation(
    method_name, supervision, train_config_name, dataset_name, to_visualize
):
    global num_samples_shown
    num_samples_shown = 0

    def load_predictions(*args):
        return np.load(get_predictions_path(*args))["mask_preds"]

    def evaluate1(i, dataset, predictions, τ, to_visualize=False):
        true_lg = dataset.load_mask(i)
        pred_sm = predictions[i]

        pred_lg = Image.fromarray(pred_sm).resize(true_lg.shape)
        pred_lg = np.array(pred_lg)
        pred_lg_binary = (pred_lg > τ).astype("float")
        pred_sm_binary = (pred_sm > τ).astype("float")

        true_sm = Image.fromarray(true_lg).resize(pred_sm.shape, resample=Image.Resampling.NEAREST)
        true_sm = np.array(true_sm)

        iou_lg_score = iou(pred_lg_binary.astype("bool"), true_lg.astype("bool"))
        iou_sm_score = iou(pred_sm_binary.astype("bool"), true_sm.astype("bool"))

        accuracy_lg_score = np.mean(pred_lg_binary == true_lg)
        accuracy_sm_score = np.mean(pred_sm_binary == true_sm)

        if to_visualize:
            st.markdown("## {} · IOU: {:.1f}%".format(i, 100 * iou_lg_score))
            cols = st.columns(4)
            cols[0].image(dataset.get_image_path(i), caption="input")
            cols[1].image(true_lg, caption="true")
            cols[2].image(pred_lg, clamp=True, caption="pred")
            cols[3].image(pred_lg_binary, caption="pred binary")
            st.markdown("---")

            # stop every 32 samples
            global num_samples_shown
            if num_samples_shown % 32 == 0:
                pdb.set_trace()
            num_samples_shown += 1

        return {
            "iou-lg": iou_lg_score,
            "iou-sm": iou_sm_score,
            "accuracy-lg": accuracy_lg_score,
            "accuracy-sm": accuracy_sm_score,
        }

    def evaluate(dataset, predictions, τ, to_visualize=False):
        metrics = [
            evaluate1(i, dataset, predictions, τ, to_visualize=to_visualize)
            for i in tqdm(range(len(dataset)))
        ]
        metrics_name = metrics[0].keys()
        results = {
            k: 100 * np.mean([m[k] for m in metrics]) for k in metrics_name
        }
        return {**results, "τ": τ}

    # dataset_valid = PREDICT_CONFIGS[dataset_name + "-valid"]["dataset"]
    dataset_test = PREDICT_CONFIGS[dataset_name + "-test"]["dataset"]

    # preds_valid = load_predictions(
    #     method_name, supervision, train_config_name, dataset_name + "-valid"
    # )
    preds_test = load_predictions(
        method_name, supervision, train_config_name, dataset_name + "-test"
    )

    if to_visualize:
        evaluate(dataset_test, preds_test, 0.5, to_visualize=to_visualize)

    # THRESHS = np.arange(0.0, 1.1, 0.1)
    # results_valid = [
    #     evaluate(dataset_valid, preds_valid, τ) for τ in THRESHS
    # ]

    # print(json.dumps(results_valid, indent=4, ensure_ascii=False))

    # result_best, *_ = sorted(results_valid, reverse=True, key=lambda r: r["score"])
    # τ_star = result_best["τ"]

    results_test = [
        # evaluate(dataset_test, preds_test, τ_star),
        evaluate(dataset_test, preds_test, 0.5),
    ]

    print("Localisation · IOU:")
    print(json.dumps(results_test, indent=4, ensure_ascii=False))

    return results_test


@click.command()
@click.option("-s", "supervision", type=click.Choice(["full", "weak"]), required=True)
@click.option("-t", "train_config_name", required=True)
@click.option("-d", "dataset_name", required=True)
@click.option("-v", "to_visualize", is_flag=True, default=False)
def main(supervision, train_config_name, dataset_name, to_visualize=False):
    method_name = "patch-forensics"
    print(supervision, train_config_name, dataset_name)
    if supervision == "weak" and dataset_name == "repaint-p2":
        evaluate_detection(
            method_name, supervision, train_config_name, dataset_name, to_visualize
        )
    evaluate_localisation(
        method_name, supervision, train_config_name, dataset_name, to_visualize
    )
    print()


if __name__ == "__main__":
    main()
