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

from dolos.methods.patch_forensics.train_full_supervision import get_datapipe
from dolos.methods.xception_attention.networks.xception_attention import (
    load_xception_attention_model,
)
from dolos.metrics.iou_ignite import IOU
from dolos.data import RepaintCleanDataset, RepaintP2CelebAHQCleanDataset, RepaintP2FFHQCleanDataset


IMAGE_SIZE = (299, 299)
MASK_SIZE = (19, 19)
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


def get_transform_mask():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(MASK_SIZE),
            transforms.ToTensor(),
        ]
    )


TRANSFORM_IMAGE = get_transform_image()
TRANSFORM_MASK = get_transform_mask()


def load_image(dataset, i, *args, **kwargs):
    return TRANSFORM_IMAGE(imread(dataset.get_image_path(i)))


CONFIGS = {
    "00": {
        "dataset-class": RepaintCleanDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-p2-celebahq-clean": {
        "dataset-class": RepaintP2CelebAHQCleanDataset,
        "load-image": load_image,
        "max-epochs": 15,
    },
    "repaint-p2-ffhq-clean": {
        "dataset-class": RepaintP2FFHQCleanDataset,
        "load-image": load_image,
        "max-epochs": 30,
    },
}


@click.command()
@click.argument("config_name")
def main(config_name):
    assert config_name in CONFIGS
    device = "cuda"

    config = CONFIGS[config_name]

    # model = load_xception_attention_model(init="pretrained")
    model = load_xception_attention_model(init="random")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_loader = get_datapipe(config, "train", TRANSFORM_MASK)
    valid_loader = get_datapipe(config, "valid", TRANSFORM_MASK)

    metrics = {
        "loss": Loss(F.binary_cross_entropy),
        "iou": IOU(0.5),
    }

    def prepare_batch(batch):
        return {k: convert_tensor(v, device=device) for k, v in batch.items()}

    def update_model(engine, batch):
        optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch)
        image = batch["image"]
        mask_true = batch["mask"]

        _, mask_pred, _ = model(image)

        loss = F.binary_cross_entropy(mask_pred, mask_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict_on_batch(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            image = batch["image"]
            mask_true = batch["mask"]
            _, mask_pred, _ = model(image)
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
        f"output/xception-attention/full-supervision/{config_name}",
        n_saved=5,
        filename_prefix="best",
        score_function=score_function,
        score_name="neg-loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    trainer.run(train_loader, max_epochs=config["max-epochs"], epoch_length=500)


if __name__ == "__main__":
    main()
