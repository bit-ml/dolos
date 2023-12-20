import pdb

import click
import torch  # type: ignore

from ignite.engine import Engine, Events  # type: ignore
from ignite.handlers import ModelCheckpoint, global_step_from_engine  # type: ignore
from ignite.metrics import Loss  # type: ignore
from ignite.utils import convert_tensor  # type: ignore

from torch.nn import functional as F  # type: ignore
from torchdata.datapipes.map import SequenceWrapper  # type: ignore

from dolos.metrics.iou_ignite import IOU
from dolos.data import (
    CelebAHQDataset,
    P2CelebAHQDataset,
    RepaintP2CelebAHQ9KDataset,
)
from dolos.methods.grad_cam.model import xceptionnet
from dolos.methods.patch_forensics.train_weak_supervision import (
    get_datapipe,
    load_image_0,
)


CONFIGS = {
    "setup-a": {
        "dataset-real": CelebAHQDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image_0,
        "batch-size": 32,
    },
    "setup-b": {
        "dataset-real": CelebAHQDataset,
        "dataset-fake": RepaintP2CelebAHQ9KDataset,
        "load-image": load_image_0,
        "batch-size": 32,
    },
}


@click.command()
@click.argument("config_name")
def main(config_name):
    assert config_name in CONFIGS
    config = CONFIGS[config_name]
    device = "cuda"

    model = xceptionnet(device)
    # TODO alternative
    # torch.optim.AdamW(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = get_datapipe(config, "train")
    valid_loader = get_datapipe(config, "valid")

    metrics = {
        "loss": Loss(
            F.cross_entropy,
            output_transform=lambda output: (
                output["label-pred"],
                output["label-true"],
            ),
        ),
    }

    def prepare_batch(batch):
        return {k: convert_tensor(v, device=device) for k, v in batch.items()}

    def update_model(engine, batch):
        optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch)
        image = batch["image"]
        label_true = batch["label"]
        label_pred = model(image)

        loss = F.cross_entropy(label_pred, label_true)
        loss.backward()

        optimizer.step()

        pred_softmax = F.softmax(label_pred, dim=1)[:, 1]
        pred_binary = (pred_softmax > 0.5).int()
        accuracy = (label_true == pred_binary).float().mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    def predict_on_batch(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = prepare_batch(batch)
            image = batch["image"]
            label_true = batch["label"]
            label_pred = model(image)
        return {
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
        f"output/grad-cam/weak-supervision/{config_name}",
        n_saved=5,
        filename_prefix="best",
        score_function=score_function,
        score_name="neg-loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    trainer.run(train_loader, max_epochs=100, epoch_length=500)


if __name__ == "__main__":
    main()  # type: ignore
