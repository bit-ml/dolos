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

from dolos.methods.grad_cam.model import xceptionnetfcn
from dolos.methods.patch_forensics.train_full_supervision import (
    get_datapipe,
    load_image,
    get_transform_mask,
)
from dolos.metrics.iou_ignite import IOU
from dolos.data import RepaintP2CelebAHQDataset


MASK_SIZE = (19, 19)


CONFIGS = {
    "setup-c": {
        "dataset-class": RepaintP2CelebAHQDataset,
        "load-image": load_image,
        "max-epochs": 50,
    },
}


@click.command()
@click.argument("config_name")
def main(config_name):
    assert config_name in CONFIGS
    device = "cuda"

    config = CONFIGS[config_name]
    max_epochs = config["max-epochs"]
    lr = config.get("learning-rate", 1e-3)

    model = xceptionnetfcn(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    transform_mask = get_transform_mask(MASK_SIZE)

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
        f"output/grad-cam/full-supervision/{config_name}",
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
