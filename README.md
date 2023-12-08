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

## Experiments

In the paper we consider three methods for weakly-supervised localization:

- **Grad-CAM.** TODO
- **Patches.** Implemented in [`dolos/methods/patch_forensics`](dolos/methods/patch_forensics/).
- **Attention.** Implemented in [`dolos/methods/xception_attention`](dolos/methods/xception_attention/).

These methods are evaluated three experimental setups:

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

TODO

To replicate the experiments:

| | Grad-CAM | Patches | Attention |
| --- | --- | --- | --- |
| Setup A | | | |
| Setup B | | | |
| Setup C | | | |

### Cross-generator experiments (§5.3)

Train the Patch Forensics detection models on each dataset; for example:

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

Predict:

```bash
for src in repaint-p2-9k repaint-ldm lama pluralistic three-but-repaint-p2-9k three-but-repaint-ldm three-but-lama three-but-pluralistic; do
    for tgt in repaint-p2-9k repaint-ldm lama pluralistic; do
        for split in test; do
            python dolos/methods/patch_forensics/predict.py -s full -t ${src} -p ${tgt}-${split}
        done
    done
done
```

Evaluate (generates Figure 6 in the paper):

```bash
streamlit run dolos/methods/patch_forensics/show_cross_model_performance.py
```

## Name

The name comes from Greek mythology; according to [Wikipedia](https://en.wikipedia.org/wiki/Dolos_(mythology)):

> Dolos (Ancient Greek: Δόλος "Deception") is the spirit of trickery. He is also a master at cunning deception, craftiness, and treachery.
> [...]
> Dolos became known for his skill when he attempted to make a fraudulent copy statue of Aletheia (Veritas), in order to trick people into thinking they were seeing the real statue.

## License

## TODO

- [ ] Add license
- [ ] Release pretrained models
- [ ] Incorporate the Grad-CAM code.
- [ ] Scripts to run the main experiments.