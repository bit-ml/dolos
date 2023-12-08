import pdb

from functools import partial

import click
import torch

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Loss
from ignite.utils import convert_tensor

from imageio.v2 import imread

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchdata.datapipes.map import SequenceWrapper
from torchdata.datapipes.iter import IterableWrapper
from torchvision import transforms

from dolos.methods.patch_forensics.networks.customnet import make_patch_xceptionnet
from dolos.metrics.iou_ignite import IOU
from dolos.data import (
    ConcatDataset,
    RepaintDataset,
    RepaintCleanDataset,
    RepaintP2CelebAHQCleanSmallDataset,
    RepaintP2CelebAHQClean9KDataset,
    RepaintP2FFHQCleanDataset,
    LamaDataset,
    LDMRepaintDataset,
    PluralisticDataset,
)


BATCH_SIZE = 16
IMAGE_SIZE = (299, 299)
MASK_SIZE = (37, 37)
MASK_SIZE_HALF = (18, 18)
NORM = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]


def get_transform_image():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*NORM),
        ]
    )


def get_transform_mask(mask_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(mask_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


TRANSFORM_IMAGE = get_transform_image()
# TRANSFORM_MASK = get_transform_mask(MASK_SIZE)


def load_image(dataset, i, *args, **kwargs):
    return TRANSFORM_IMAGE(imread(dataset.get_image_path(i)))


CONFIGS = {
    "repaint-noisy-00": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-clean-00": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintCleanDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-clean-01": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintCleanDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "repaint-p2-celebahq-small": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2CelebAHQCleanSmallDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-p2-celebahq-9k": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2CelebAHQClean9KDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "repaint-p2-ffhq": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2FFHQCleanDataset,
        "load-image": load_image,
        "max-epochs": 30,
    },
    "repaint-p2-ffhq-block3": {
        "last-layer": "block3",
        "frontend": None,
        "dataset-class": RepaintP2FFHQCleanDataset,
        "load-image": load_image,
        "max-epochs": 75,
    },
    "repaint-p2-ffhq-lr-3e-4": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2FFHQCleanDataset,
        "load-image": load_image,
        "max-epochs": 30,
        "learning-rate": 3e-4,
    },
    "lama-00": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": LamaDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "lama-01": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": LamaDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "ldm-repaint-00": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": LDMRepaintDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "ldm-repaint-01": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": LDMRepaintDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "pluralistic-00": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": PluralisticDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "pluralistic-01": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": PluralisticDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "three-but-repaint": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 15,
    },
    "three-but-lama": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 15,
    },
    "three-but-pluralistic": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            RepaintCleanDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 15,
    },
    "three-but-ldm": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            LamaDataset(split),
            PluralisticDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 15,
    },
    "three-but-repaint-max-epochs-50": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 50,
    },
    "three-but-lama-max-epochs-50": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 50,
    },
    "three-but-pluralistic-max-epochs-50": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            RepaintCleanDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 50,
    },
    "three-but-ldm-max-epochs-50": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            LamaDataset(split),
            PluralisticDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 50,
    },
    "three-but-repaint-max-epochs-150": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-lama-max-epochs-150": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-pluralistic-max-epochs-150": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            RepaintCleanDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-ldm-max-epochs-150": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintCleanDataset(split),
            LamaDataset(split),
            PluralisticDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-lama-max-epochs-150-p2": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintP2CelebAHQClean9KDataset(split),
            PluralisticDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-pluralistic-max-epochs-150-p2": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            RepaintP2CelebAHQClean9KDataset(split),
            LDMRepaintDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-ldm-max-epochs-150-p2": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintP2CelebAHQClean9KDataset(split),
            LamaDataset(split),
            PluralisticDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "repaint-clean-fad": {
        "last-layer": "block2",
        "frontend": "fad",
        "dataset-class": RepaintCleanDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-clean-lfs": {
        "last-layer": "block2",
        "frontend": "lfs",
        "dataset-class": RepaintCleanDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
}


def get_datapipe(config, split, transform_mask):
    dataset = config["dataset-class"](split)
    load_image = config["load-image"]

    def get_sample(i):
        image = load_image(dataset, i)
        mask = transform_mask(imread(dataset.get_mask_path(i)))
        mask = (mask < 0.5).float()
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(1),
        }

    datapipe = SequenceWrapper(range(len(dataset)))
    datapipe = datapipe.map(get_sample)

    if split == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()

    datapipe = datapipe.batch(BATCH_SIZE)
    datapipe = datapipe.collate()

    return datapipe


def get_mask_size(config):
    if config["last-layer"] == "block3":
        return (19, 19)
    elif config["frontend"] in (None, "fad"):
        return MASK_SIZE
    elif config["frontend"] == "lfs":
        return MASK_SIZE_HALF
    else:
        assert False


@click.command()
@click.option("-c", "--config", "config_name", required=True)
def main(config_name):
    assert config_name in CONFIGS
    device = "cuda"

    config = CONFIGS[config_name]
    max_epochs = config["max-epochs"]
    lr = config.get("learning-rate", 1e-3)

    model = make_patch_xceptionnet(
        layername=config["last-layer"], frontend=config["frontend"]
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    transform_mask = get_transform_mask(get_mask_size(config))

    train_loader = get_datapipe(config, "train", transform_mask)
    valid_loader = get_datapipe(config, "valid", transform_mask)

    def output_transform(output):
        pred, true = output
        pred = F.softmax(pred, dim=1)[:, 1]
        true = true.float()
        return pred, true

    metrics = {
        "loss": Loss(F.cross_entropy),
        "iou": IOU(0.5, output_transform=output_transform),
    }

    def prepare_batch(batch):
        return {k: convert_tensor(v, device=device) for k, v in batch.items()}

    def update_model(engine, batch):
        optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch)
        image = batch["image"]
        mask_true = batch["mask"]
        mask_true = mask_true.squeeze(1).long()

        mask_pred = model(image)

        loss = F.cross_entropy(mask_pred, mask_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict_on_batch(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            image = batch["image"]
            mask_true = batch["mask"]
            mask_true = mask_true.squeeze(1).long()
            mask_pred = model(image)
        return mask_pred, mask_true

    trainer = Engine(update_model)
    evaluator = Engine(predict_on_batch)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            "{:3d} · {:6d} ◇ {:.5f}".format(
                engine.state.epoch,
                engine.state.iteration,
                engine.state.output,
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        print(
            "{:3d} · {:6s} ◇ {:.5f} · {:.5f}".format(
                trainer.state.epoch,
                "valid",
                metrics["loss"],
                metrics["iou"],
            )
        )

    def score_function(engine):
        return -engine.state.metrics["loss"]

    model_checkpoint = ModelCheckpoint(
        f"output/patch-forensics/full-supervision/{config_name}",
        n_saved=5,
        filename_prefix="best",
        score_function=score_function,
        score_name="neg-loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    trainer.run(train_loader, max_epochs=max_epochs, epoch_length=500)


if __name__ == "__main__":
    main()
