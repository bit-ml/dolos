# TODO
# - [x] monitor full loss at validation?
#
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

from dolos.methods.patch_forensics.train_weak_supervision import get_datapipe
from dolos.methods.xception_attention.networks.xception_attention import load_xception_attention_model
from dolos.methods.xception_attention.train_full_supervision import load_image
from dolos.metrics.iou_ignite import IOU
from dolos.data import (
    CelebAHQProcessedDataset,
    FFHQProcessedDataset,
    P2CelebAHQDataset,
    P2FFHQDataset,
    RepaintCleanDataset,
    RepaintP2CelebAHQCleanSmallDataset,
)

# a · FFHQ     vs P2 (FFHQ)
# b · CelebAHQ vs P2 (CelebAHQ)
# c · CelebAHQ vs Repaint (CelebAHQ)


CONFIGS = {
    "setup-a-00": {
        "dataset-real": FFHQProcessedDataset,
        "dataset-fake": P2FFHQDataset,
        "load-image": load_image,
        "λ": 0.01,
    },
    "setup-a-01": {
        "dataset-real": FFHQProcessedDataset,
        "dataset-fake": P2FFHQDataset,
        "load-image": load_image,
        "λ": 0.1,
    },
    "setup-a-01-100": {
        "dataset-real": FFHQProcessedDataset,
        "dataset-fake": P2FFHQDataset,
        "load-image": load_image,
        "λ": 0.1,
        "max-epochs": 100,
    },
    "setup-a-02": {
        "dataset-real": FFHQProcessedDataset,
        "dataset-fake": P2FFHQDataset,
        "load-image": load_image,
        "λ": 1.0,
    },
    "setup-b-00": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 0.01,
    },
    "setup-b-01": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 0.1,
    },
    "setup-b-01-100": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 0.1,
        "max-epochs": 100,
    },
    "setup-b-02": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 1.0,
    },
    "setup-c-00": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintCleanDataset,
        "load-image": load_image,
        "λ": 0.01,
    },
    "setup-c-01": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintCleanDataset,
        "load-image": load_image,
        "λ": 0.1,
    },
    "setup-c-01-100": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintCleanDataset,
        "load-image": load_image,
        "λ": 0.1,
        "max-epochs": 100,
    },
    "setup-c-02": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintCleanDataset,
        "load-image": load_image,
        "λ": 1.0,
    },
    # new experiments for the WACV paper
    "setup-a-p2-celebahq-full": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 0.01,
        "max-epochs": 15,
        "batch-size": 16,
    },
    "setup-b-p2-celebahq-partial": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintP2CelebAHQCleanSmallDataset,
        "load-image": load_image,
        "λ": 0.01,
        "max-epochs": 15,
        "batch-size": 16,
    },
    "setup-a-p2-celebahq-full-λ-1": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image,
        "λ": 1,
        "max-epochs": 15,
        "batch-size": 16,
    },
    "setup-b-p2-celebahq-partial-λ-1": {
        "dataset-real": CelebAHQProcessedDataset,
        "dataset-fake": RepaintP2CelebAHQCleanSmallDataset,
        "load-image": load_image,
        "λ": 1,
        "max-epochs": 15,
        "batch-size": 16,
    },
}


@click.command()
@click.argument("config_name")
def main(config_name):
    assert config_name in CONFIGS
    config = CONFIGS[config_name]
    device = "cuda"

    model = load_xception_attention_model(init="pretrained")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_datapipe(config, "train")
    valid_loader = get_datapipe(config, "valid")

    def output_transform_1(output):
        pred = output["mask-pred"].amax(dim=(1, 2, 3))
        true = output["label-true"].float()
        return pred, true

    def output_transform_2(output):
        pred = output["label-pred"]
        true = output["label-true"]
        return pred, true

    loss_mask = Loss(F.binary_cross_entropy, output_transform=output_transform_1)
    loss_label = Loss(F.cross_entropy, output_transform=output_transform_2)
    loss = loss_label + config["λ"] * loss_mask

    metrics = {
        "loss-mask": loss_mask,
        "loss-label": loss_label,
        "loss": loss,
    }

    def prepare_batch(batch):
        return {k: convert_tensor(v, device=device) for k, v in batch.items()}

    def update_model(engine, batch):
        optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch)
        image = batch["image"]
        label_true = batch["label"]

        label_pred, mask_pred, _ = model(image)

        mask_max = mask_pred.amax(dim=(1, 2, 3))
        loss_mask = F.binary_cross_entropy(mask_max, label_true.float())
        loss_label = F.cross_entropy(label_pred, label_true)

        loss = loss_label + config["λ"] * loss_mask
        loss.backward()
        optimizer.step()

        _, label_pred_ = torch.max(label_pred, dim=1)
        accuracy = (label_true == label_pred_).float().mean()

        return {
            "loss": loss.item(),
            "loss-label": loss_label.item(),
            "loss-mask": loss_mask.item(),
            "accuracy": accuracy.item(),
        }

    def predict_on_batch(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            image = batch["image"]
            label_true = batch["label"]
            label_pred, mask_pred, _ = model(image)
        return {
            "mask-pred": mask_pred,
            "label-pred": label_pred,
            "label-true": label_true,
        }

    trainer = Engine(update_model)
    evaluator = Engine(predict_on_batch)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        losses_str = (f"{k} = {v:.5f}" for k, v in engine.state.output.items())
        losses_str = " · ".join(losses_str)
        print(
            "{:3d} · {:6d} ◇ {:s}".format(
                engine.state.epoch, engine.state.iteration, losses_str
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        losses_str = " · ".join(f"{k} = {v:.5f}" for k, v in metrics.items())
        print(
            "{:3d} · {:6s} ◇ {:s}".format(
                trainer.state.epoch,
                "valid",
                losses_str,
            )
        )

    def score_function(engine):
        return -engine.state.metrics["loss"]

    model_checkpoint = ModelCheckpoint(
        f"output/xception-attention/weak-supervision/{config_name}",
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
