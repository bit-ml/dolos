import random
import os
import streamlit as st

from tabulate import tabulate

from dolos.data import RepaintCleanDataset, PathSplitDataset, CelebAHQDataset
from dolos.methods.patch_forensics.predict import PREDICT_CONFIGS


# DATASETS = ["repaint-clean", "ldm-repaint", "lama", "pluralistic"]

SRCS = ["repaint-p2-celebahq-9k", "ldm-repaint-01", "lama-01", "pluralistic-01"]
TGTS = ["repaint-p2-celebahq-9k-test", "ldm-repaint-test", "lama-test", "pluralistic-test"]

split = "test"
dataset = RepaintCleanDataset(split)


def get_path_real(key):
    folder = os.path.join("data", "repaint", "inpainted", "celebahq")
    return os.path.join(folder, "gt", key + ".png")


def get_path_pred(src, tgt, key):
    return os.path.join(
        "output",
        "patch-forensics",
        "full-supervision",
        "predictions-train-config-{}-predict-config-{}".format(src, tgt),
        key + ".png",
    )


indices = list(range(len(dataset)))
random.shuffle(indices)
indices = [142, 443, 493, 545, 417, 263] + indices


def get_path_inpainted(tgt, key, i):
    d = PREDICT_CONFIGS[tgt]["dataset"]
    if isinstance(d, PathSplitDataset):
        return str(d.path_images / d.split / (key + ".png"))
    elif isinstance(d, CelebAHQDataset):
        return PREDICT_CONFIGS[tgt]["dataset"].get_image_path(i)
    else:
        assert False


for i in indices[:32]:
    datum = dataset.metadata[i]
    key = datum["key"]
    st.markdown(f"`{i}` · `{key}`")
    num_cols = len(SRCS) + 1
    cols = st.columns(num_cols)
    cols[0].image(get_path_real(key))
    cols[1].image(dataset.get_mask_path(i))
    for tgt in TGTS:
        cols = st.columns(num_cols)
        cols[0].image(get_path_inpainted(tgt, key, i))
        for j, src in enumerate(SRCS, start=1):
            cols[j].image(get_path_pred(src, tgt, key))
    st.markdown("---")


def prepare_cross_generator_results_for_paper(i):
    import shutil

    latex_folder = "WACV/imgs/cross-generator"

    datum = dataset.metadata[i]
    key = datum["key"]

    def get_image_path(type_, *args):
        if type_ == "image":
            return get_path_real(key)
        elif type_ == "mask":
            return dataset.get_mask_path(i)
        elif type_ == "inpainted":
            tgt = args[0]
            return get_path_inpainted(tgt, key, i)
        elif type_ == "pred":
            src = args[0]
            tgt = args[1]
            return get_path_pred(src, tgt, key)
        else:
            # print(inp)
            assert False

    def latex_img(type_, *args):
        # filename = type_ + "-" + "-".join(map(str, args)) + "-" + str(i) + ".png"
        filename = "-".join([key, type_] + list(args)) 
        filename = filename + ".png"
        src = get_image_path(type_, *args)
        dst = os.path.join("output/wacv", latex_folder, filename)
        shutil.copy(src, dst)
        return (
            r"\includegraphics[align=c,height=\hf]{"
            + latex_folder
            + "/"
            + filename
            + "}"
        )

    def latex_multicol(n, txt):
        return r"\multicolumn{" + str(n) + "}{c}{" + txt + "}"

    rows = [
        ["image", "mask", "", ""],
        [latex_img("image"), latex_img("mask"), "", ""],
        # ["", latex_multicol(3, "trained on"), "", ""],
        ["inpainted", "repaint", "ldm", "lama", "pluralistic"],
    ] + [
        [latex_img(f"inpainted", tgt)]
        + [latex_img("pred", src, tgt) for src in SRCS]
        for tgt in TGTS
    ]

    # return "\n".join("\n& ".join(row) + r" \\" for row in rows)
    return rows


rows1 = prepare_cross_generator_results_for_paper(443)
rows2 = prepare_cross_generator_results_for_paper(263)

rows = [r1 + r2 for r1, r2 in zip(rows1, rows2)] 

print(tabulate(rows, headers=[], tablefmt="latex_raw"))
