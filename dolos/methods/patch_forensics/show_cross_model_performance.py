import json
import os
import pdb

import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image
from toolz import first
from tqdm import tqdm

from dolos.utils import cache
from dolos.methods.patch_forensics.evaluate import evaluate_localisation


sns.set_theme(style="ticks", context="talk")


NAMES = {
    "repaint-p2-9k": "repaint",
    "repaint-ldm": "ldm",
    "lama": "lama",
    "pluralistic": "plura",
    "three-but-repaint-p2": "w/o repaint",
    "three-but-repaint-ldm": "w/o ldm",
    "three-but-lama": "w/o lama",
    "three-but-pluralistic": "w/o plura",
}


def get_cache_path(src, tgt):
    return f"/tmp/patch-forensics-{src}-{tgt}.json"


def evaluate1(src, tgt):
    results = evaluate_localisation(
        "patch-forensics",
        "full",
        src,
        tgt,
        to_visualize=False,
    )
    # Compared to the previous evaluation we are now using the a fixed threshold of 0.5 for all models.
    result = first(result for result in results if result["τ"] == 0.5)
    assert result["τ"] == 0.5
    return result["iou-lg"]


def add_plot(ax, sets_tr, sets_te):
    scores = [
        {
            "iou": cache(get_cache_path(src, tgt), evaluate1, src, tgt),
            "train on": NAMES[src],
            "test on": NAMES[tgt],
        }
        for src in sets_tr
        for tgt in sets_te
    ]
    print(scores)

    df = pd.DataFrame(scores)
    df = pd.pivot_table(
        df, columns="train on", index="test on", values="iou", sort=False
    )

    ax = sns.heatmap(data=df, annot=True, annot_kws={"fontsize": 18}, fmt=".1f", cbar=False, square=True, ax=ax)

    return ax


fig, axs = plt.subplots(
    ncols=2, figsize=(12, 6), sharey=True, gridspec_kw={"wspace": 0.05, "hspace": 0}
)
axs[0] = add_plot(
    axs[0],
    sets_tr=["repaint-p2-9k", "repaint-ldm", "lama", "pluralistic"],
    sets_te=["repaint-p2-9k", "repaint-ldm", "lama", "pluralistic"],
)

axs[1] = add_plot(
    axs[1],
    sets_tr=[
        "three-but-repaint-p2-9k",
        "three-but-repaint-ldm",
        "three-but-lama",
        "three-but-pluralistic",
    ],
    sets_te=["repaint-p2-9k", "repaint-ldm", "lama", "pluralistic"],
)
# remove ylabels and yxticklabels and yxticks
axs[1].set(ylabel=None, xlabel="train on (combinations of three)")

# fig.savefig("output/wmf23/imgs/cross-generator-performance-patch-forensics.pdf")
fig.savefig("output/wacv/imgs/cross-generator-performance-patch-forensics.pdf")

# fig.tight_layout()
st.pyplot(fig)
