This repository is the implementation of the paper:

> Dragoș Țânțaru, Elisabeta Oneață, Dan Oneață.
> [Weakly-supervised deepfake localization in diffusion-generated images.](https://arxiv.org/pdf/2311.04584)
> WACV, 2024.

## Setup

The code depends on Pytorch, which has to be installed separately.
For example:

```bash
conda create -n dolos python=3.9
conda activate dolos
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then run:

```bash
pip install -e .
```

This repository also integrates code from the following repositories:

- [Patch Forensics](https://github.com/chail/patch-forensics)
- [Facial Forgery Detection](https://github.com/JStehouwer/FFD_CVPR2020)

## Data

Our datasets are available [here](https://sharing.speed.pub.ro/owncloud/index.php/s/8kPtPO8efjRMUAG).
They can be downloaded remotely as follows:

```bash
wget "https://sharing.speed.pub.ro/owncloud/index.php/s/8kPtPO8efjRMUAG/download" -O dolos-data.zip
```

After unzipping the dataset, with

```bash
unzip dolos-data.zip
```

set a symbolic link in the root of this repository; for example:

```bash
cd /path/to/dolos && ln -s /path/to/wacv_data data
```

**Note.**
The masks are given as black-and-with images, where black (`0`) means _fake_ and (white) `255` means _real_.
However, the models below adopt the opposite convention: `0` means _real_ and `1` means _fake_.

## Experiments

In the paper we consider three methods for weakly-supervised localization:

- **Grad-CAM.** Implemented in [`dolos/methods/grad_cam`](dolos/methods/grad_cam/).
- **Patches.** Implemented in [`dolos/methods/patch_forensics`](dolos/methods/patch_forensics/).
- **Attention.** Implemented in [`dolos/methods/xception_attention`](dolos/methods/xception_attention/).
For training, we initialize from the pretrained model provided by the authors.
The weights can be downloaded as follows:

```bash
mkdir assets
wget https://github.com/JStehouwer/FFD_CVPR2020/releases/download/v2.0/xcp_reg.tar -O assets/xcp_reg.tar
```

These three methods are evaluated on three experimental setups:

- **Setup A:** a weakly-supervised setup in which we have access to fully-generated images as fakes and, consequently, only image-level labels.
- **Setup B:** a weakly-supervised setup in which we have access to partially-manipulated images, but only with image-level labels (no localization information).
- **Setup C:** a fully-supervised setting in which we have access to ground-truth localization masks of partially-manipulated images.

The weakly-supervised setups (A and B) are implemented by the `train_weak_supervision.py` script,
while the fully-supervised setup (C) is implemented by the `train_full_supervision.py` script.
For example:

```bash
ls dolos/methods/xception_attention/train_weak_supervision.py
ls dolos/methods/xception_attention/train_full_supervision.py
```

### Main experiments (§5.1)

In this set of experiments we evaluate the localization capabilities of the three methods in the three setups.
The table below contains links to the scripts that can replicate the experiments in Table 2 from our paper:

| | Grad-CAM | Patches | Attention |
| --- | --- | --- | --- |
| Setup A | | [`run-patches-setup-a.sh`](dolos/scripts/run-patches-setup-a.sh) | [`run-attention-setup-a.sh`](dolos/scripts/run-attention-setup-a.sh) |
| Setup B | | [`run-patches-setup-b.sh`](dolos/scripts/run-patches-setup-b.sh) | [`run-attention-setup-b.sh`](dolos/scripts/run-attention-setup-b.sh) |
| Setup C | | [`run-patches-setup-c.sh`](dolos/scripts/run-patches-setup-c.sh) | [`run-attention-setup-c.sh`](dolos/scripts/run-attention-setup-c.sh) |

### Cross-dataset experiments (§5.2)

For this setup,
we train the Patches model on one source dataset (FFHQ) and evaluate it on another target dataset (CelebA-HQ);
note that the generation method is kept fixed (P2 diffusion).
Below are the scripts needed to obtain results for the three setups, corresponding to Table 3 in the paper:

- Setup A: [`run-patches-setup-a-ffhq.sh`](dolos/scripts/run-patches-setup-a-ffhq.sh)
- Setup B: [`run-patches-setup-b-ffhq.sh`](dolos/scripts/run-patches-setup-b-ffhq.sh)
- Setup C: [`run-patches-setup-c-ffhq.sh`](dolos/scripts/run-patches-setup-c-ffhq.sh)

### Cross-generator experiments (§5.3)

Here we evaluate the Patches detection model across multiple combinations of train-test datasets.
For these experiments, we vary the generation method, but keep the source dataset fixed (CelebA-HQ)

To train on a given dataset:
```bash
python dolos/methods/patch_forensics/train_full_supervision.py -c repaint-p2-9k
```

The datasets are specified by a configuration, and can be set to one of the following:
`repaint-p2-9k`,
`repaint-ldm`,
`lama`,
`pluralistic`,
`three-but-repaint-p2-9k`,
`three-but-repaint-ldm`,
`three-but-lama`,
`three-but-pluralistic`.

To obtain predictions for all combinations:

```bash
for src in repaint-p2-9k repaint-ldm lama pluralistic three-but-repaint-p2-9k three-but-repaint-ldm three-but-lama three-but-pluralistic; do
    for tgt in repaint-p2-9k repaint-ldm lama pluralistic; do
        for split in test; do
            python dolos/methods/patch_forensics/predict.py -s full -t ${src} -p ${tgt}-${split}
        done
    done
done
```

Finally, to obtain the evaluation metrics and generate Figure 6 in the paper:

```bash
streamlit run dolos/methods/patch_forensics/show_cross_model_performance.py
```

## Name

The name comes from Greek mythology; according to [Wikipedia](https://en.wikipedia.org/wiki/Dolos_(mythology)):

> Dolos (Ancient Greek: Δόλος "Deception") is the spirit of trickery. He is also a master at cunning deception, craftiness, and treachery.
> [...]
> Dolos became known for his skill when he attempted to make a fraudulent copy statue of Aletheia (Veritas), in order to trick people into thinking they were seeing the real statue.

## License
