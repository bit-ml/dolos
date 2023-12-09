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
    RepaintP2FFHQDataset,
    RepaintP2CelebAHQDataset,
    RepaintP2CelebAHQ9KDataset,
    RepaintLDMCelebAHQDataset,
    LamaDataset,
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
    "setup-c": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2CelebAHQDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "setup-c-ffhq": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2FFHQDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "repaint-p2-9k": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintP2CelebAHQ9KDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "repaint-ldm": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": RepaintLDMCelebAHQDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "lama": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": LamaDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
    "pluralistic": {
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
            RepaintLDMCelebAHQDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-ldm": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintP2CelebAHQ9KDataset(split),
            LamaDataset(split),
            PluralisticDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-lama": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            RepaintP2CelebAHQ9KDataset(split),
            PluralisticDataset(split),
            RepaintLDMCelebAHQDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
    },
    "three-but-pluralistic": {
        "last-layer": "block2",
        "frontend": None,
        "dataset-class": lambda split: ConcatDataset(
            LamaDataset(split),
            RepaintP2CelebAHQ9KDataset(split),
            RepaintLDMCelebAHQDataset(split),
        ),
        "load-image": load_image,
        "max-epochs": 150,
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
@click.argument("config_name")
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
