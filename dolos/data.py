import os

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np

from PIL import Image


Split = Literal["train", "valid", "test"]


@dataclass
class PathDataset:
    path_images: Path
    split: Split

    def __post_init__(self):
        self.files = self.load_filelist(self.split)

    def load_filelist(self, split):
        return os.listdir(self.path_images / split)

    def get_file_name(self, i):
        return Path(self.files[i]).stem

    def get_image_path(self, i):
        return str(self.path_images / self.split / self.files[i])

    def __len__(self):
        return len(self.files)


class WithMasksPathDataset(PathDataset):
    def __init__(self, path_images, path_masks, split):
        self.path_masks = path_masks
        super().__init__(path_images, split)

    def get_mask_path(self, i):
        return str(self.path_masks / self.split / self.files[i])

    @staticmethod
    def load_mask_keep(path):
        """Assumes that the in the original mask:

        - `255` means unchanged content, and
        - `0` means modified content.

        """
        mask = np.array(Image.open(path))
        mask = 1 - (mask[:, :, 0] == 255).astype("float")
        return mask

    def load_mask(self, i):
        return self.load_mask_keep(self.get_mask_path(i))

    def __len__(self):
        return len(self.files)


# § · Real datasets


class CelebAHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/celebahq/real")
        super().__init__(path_images=path_images, split=split)


class FFHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/ffhq/real")
        super().__init__(path_images=path_images, split=split)


# § · Fake datasets: Fully-manipulated


class P2CelebAHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/celebahq/fake/p2")
        super().__init__(path_images=path_images, split=split)


class P2FFHQDataset(PathDataset):
    def __init__(self, split):
        path_images = Path("data/ffhq/fake/p2")
        super().__init__(path_images=path_images, split=split)


# § · Fake datasets: Partially-manipulated


class RepaintP2CelebAHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/repaint-p2")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class RepaintP2FFHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/ffhq/fake/repaint-p2")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class RepaintP2CelebAHQ9KDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/repaint-p2-9k")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class RepaintLDMCelebAHQDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/ldm")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class LamaDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/lama")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class PluralisticDataset(WithMasksPathDataset):
    def __init__(self, split):
        path_base = Path("data/celebahq/fake/pluralistic")
        super().__init__(
            path_images=path_base / "images",
            path_masks=path_base / "masks",
            split=split,
        )


class ConcatDataset:
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]

    def _get_dataset_index_and_offset(self, i):
        for j, length in enumerate(self.lengths):
            if i < length:
                return j, i
            i -= length
        raise IndexError

    def get_image_path(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j].get_image_path(i)

    def get_mask_path(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j].get_mask_path(i)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, i):
        j, i = self._get_dataset_index_and_offset(i)
        return self.datasets[j][i]
