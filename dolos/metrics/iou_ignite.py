import numpy as np

from ignite.metrics import Metric


def iou(pred, true):
    inter = pred & true
    union = pred | true
    if union.sum() == 0:
        return 1.0
    else:
        return inter.sum() / union.sum()


class IOU(Metric):
    def __init__(self, thresh, output_transform=lambda x: x, device="cpu"):
        self._ious = []
        self._thresh = thresh
        super(IOU, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._ious = []
        super(IOU, self).reset()

    def update(self, output):
        y_pred = output[0].detach().cpu()
        y_true = output[1].detach().cpu()

        pred = y_pred > self._thresh
        true = y_true > self._thresh

        self._ious.append(iou(pred, true))

    def compute(self):
        return np.mean(self._ious)
