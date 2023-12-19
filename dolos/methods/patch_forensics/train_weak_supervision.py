# TODO
# - [x] monitor full loss at validation?
#
import pdb

from functools import partial

import click
import torch  # type: ignore

from ignite.engine import Engine, Events  # type: ignore
from ignite.handlers import ModelCheckpoint, global_step_from_engine  # type: ignore
from ignite.metrics import Loss  # type: ignore
from ignite.utils import convert_tensor  # type: ignore

from torch.nn import functional as F  # type: ignore
from torchdata.datapipes.map import SequenceWrapper  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore

from dolos.methods.patch_forensics.networks.customnet import make_patch_xceptionnet
from dolos.methods.patch_forensics.networks.netutils import init_net
from dolos.metrics.iou_ignite import IOU
from dolos.data import (
    CelebAHQDataset,
    FFHQDataset,
    P2CelebAHQDataset,
    P2FFHQDataset,
    RepaintP2CelebAHQ9KDataset,
    RepaintP2FFHQDataset,
)
from dolos.methods.patch_forensics.train_full_supervision import (
    IMAGE_SIZE,
    NORM,
    load_image as load_image_full,
)

# a · FFHQ     vs P2 (FFHQ)
# b · CelebAHQ vs P2 (CelebAHQ)
# c · CelebAHQ vs Repaint (CelebAHQ)


TRANSFORM_IMAGE = {
    "train": transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=Image.Resampling.LANCZOS),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*NORM),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, interpolation=Image.Resampling.LANCZOS),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*NORM),
        ]
    ),
}


def load_image_0(dataset, i, split):
    """Use same procedure as Patch Forensics"""
    path = dataset.get_image_path(i)
    image = Image.open(path).convert("RGB")
    return TRANSFORM_IMAGE[split](image)


def output_transform_expand(pred, true):
    B, _, H, W = pred.shape
    true = true.view(-1, 1, 1).expand(B, H, W)
    return pred, true


def output_transform_pooled_max(pred, true):
    B, C, H, W = pred.shape
    pred = pred.reshape(B, C, H * W)
    diff = pred[:, 1] - pred[:, 0]
    idxs = diff.argmax(dim=1)
    pred = pred[torch.arange(B), :, idxs]
    return pred, true


def output_transform_pooled_avg(pred, true):
    # Take the mean after softmax.
    C = 0.7
    pred_softmax = torch.softmax(pred, dim=1)
    pred_softmax_mean = pred_softmax.mean(dim=(2, 3))
    pred = torch.log(pred_softmax_mean) + C
    return pred, true


CONFIGS = {
    "setup-a": {
        "dataset-real": CelebAHQDataset,
        "dataset-fake": P2CelebAHQDataset,
        "load-image": load_image_0,
        "last-layer": "block2",
        "frontend": None,
        "output-transform": output_transform_expand,
    },
    "setup-b": {
        "dataset-real": CelebAHQDataset,
        "dataset-fake": RepaintP2CelebAHQ9KDataset,
        "load-image": load_image_0,
        "last-layer": "block2",
        "frontend": None,
        "output-transform": output_transform_expand,
    },
    # Cross-dataset experiments
    "setup-a-ffhq": {
        "dataset-real": FFHQDataset,
        "dataset-fake": P2FFHQDataset,
        "load-image": load_image_0,
        "last-layer": "block2",
        "frontend": None,
        "output-transform": output_transform_expand,
    },
    "setup-b-ffhq": {
        "dataset-real": FFHQDataset,
        "dataset-fake": RepaintP2FFHQDataset,
        "load-image": load_image_0,
        "last-layer": "block2",
        "frontend": None,
        "output-transform": output_transform_expand,
    },
}


BATCH_SIZE = 64
LAST_LAYER = "block2"
INIT_TYPE = "xavier"


def get_datapipe(config, split):
    load_image = config["load-image"]
    batch_size = config.get("batch-size", BATCH_SIZE)

    def get_sample(dataset, label, i):
        return {
            "image": load_image(dataset, i, split),
            "label": torch.tensor(label),
        }

    dataset_real = config["dataset-real"](split)
    dataset_fake = config["dataset-fake"](split)

    datapipe_real = SequenceWrapper(range(len(dataset_real)))
    datapipe_real = datapipe_real.map(partial(get_sample, dataset_real, 0))

    datapipe_fake = SequenceWrapper(range(len(dataset_fake)))
    datapipe_fake = datapipe_fake.map(partial(get_sample, dataset_fake, 1))

    datapipe = datapipe_real.concat(datapipe_fake)

    if split == "train":
        datapipe = datapipe.shuffle()
        datapipe = datapipe.cycle()
    else:
        datapipe = datapipe.to_iter_datapipe()

    # datapipe = datapipe.header(64)
    datapipe = datapipe.batch(batch_size)
    datapipe = datapipe.collate()

    return datapipe


@click.command()
@click.argument("config_name")
def main(config_name):
    assert config_name in CONFIGS
    config = CONFIGS[config_name]
    device = "cuda"

    model = make_patch_xceptionnet(layername=LAST_LAYER)
    model = init_net(model, init_type=INIT_TYPE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = get_datapipe(config, "train")
    valid_loader = get_datapipe(config, "valid")

    output_transform = config["output-transform"]
    metrics = {
        "loss": Loss(
            F.cross_entropy,
            output_transform=lambda output: output_transform(
                output["mask-pred"], output["label-true"]
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

        mask_pred = model(image)
        pred, true = output_transform(mask_pred, label_true)

        loss = F.cross_entropy(pred, true)
        loss.backward()

        optimizer.step()

        pred_softmax = F.softmax(pred, dim=1)[:, 1]
        pred_binary = (pred_softmax > 0.5).int()
        accuracy = (true == pred_binary).float().mean()

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
            mask_pred = model(image)
        return {
            "mask-pred": mask_pred,
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
        f"output/patch-forensics/weak-supervision/{config_name}",
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
