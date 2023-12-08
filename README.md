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
- **Patch forensics.** Implemented in `dolos/methods/patch_forensics`.
- **Attention.** Implemented in `dolos/methods/xception_attention`.

These methods are evaluated three experimental setups:

- **Setup A:** a weakly-supervised setup in which we have access to fully-generated images as fakes and, consequently, only image-level labels.
- **Setup B:** a weakly-supervised setup in which we have access to partially-manipulated images, but only with image-level labels (no localization information).
- **Setup C:** a fully-supervised setting in which we have access to ground-truth localization masks of partially-manipulated images.

### Main experiments (§5.1)

To replicate the experiments:

### Cross-generator experiments (§5.3)

Train detection models on each locally-generated dataset:

```bash
python dolos/methods/patch_forensics/train_full_supervision.py -c repaint-clean-00
python dolos/methods/patch_forensics/train_full_supervision.py -c ldm-00
python dolos/methods/patch_forensics/train_full_supervision.py -c lama-00
python dolos/methods/patch_forensics/train_full_supervision.py -c pluralistic-00
```

Predict:

```bash
for src in repaint-clean ldm lama pluralistic; do
    for tgt in repaint-clean ldm lama pluralistic; do
        for split in test; do
            python dolos/methods/patch_forensics/predict.py -s full -t ${src}-00 -p ${tgt}-${split}
        done
    done
done
```

Evaluate:

```bash
streamlit run dolos/methods/patch_forensics/show_cross_model_performance.py
```

## Name

The name comes from Greek mythology; according [Wikipedia](https://en.wikipedia.org/wiki/Dolos_(mythology)):

> Dolos (Ancient Greek: Δόλος "Deception") is the spirit of trickery. He is also a master at cunning deception, craftiness, and treachery.
> [...]
> Dolos became known for his skill when he attempted to make a fraudulent copy statue of Aletheia (Veritas), in order to trick people into thinking they were seeing the real statue.

## License

## TODO

- [ ] Add license
- [ ] Release pretrained models
- [ ] Incorporate the Grad-CAM code.