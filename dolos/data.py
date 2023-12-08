import json
import os

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def get_log_dir(config_name):
    return "dolos/supervised_segmentation/output/" + config_name


class MyDataset(ABC, Dataset):
    @abstractmethod
    def get_file_name(self, i):
        raise NotImplementedError

    @abstractmethod
    def load_mask(self, i):
        pass

    def get_prediction_dir(self, config_name, predict_config_name):
        log_dir = get_log_dir(config_name)
        return f"{log_dir}/predictions-{predict_config_name}"

    def get_prediction_path(self, i, config_name, predict_config_name):
        file_ = self.get_file_name(i) + ".png"
        return os.path.join(
            self.get_prediction_dir(config_name, predict_config_name), file_
        )

    def make_prediction_dir(self, config_name, predict_config_name):
        os.makedirs(
            self.get_prediction_dir(config_name, predict_config_name), exist_ok=True
        )


class TemplateDataset:
    def __init__(self, base_path, split):
        self.base_path = base_path
        self.split = split
        self.files = self.load_filelist(split)

    def load_filelist(self, split):
        path = os.path.join(self.base_path, split + ".txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_image_path(self, i):
        return os.path.join(self.base_path, self.files[i])

    def __len__(self):
        return len(self.files)


class FFHQProcessedDataset(TemplateDataset):
    def __init__(self, split):
        base_path = "data/ffhq/processed-repaint-bicubic"
        super().__init__(base_path, split)


class CelebAHQProcessedDataset(TemplateDataset):
    def __init__(self, split):
        base_path = "data/celebahq/processed-repaint-bicubic"
        super().__init__(base_path, split)


class P2FFHQDataset(TemplateDataset):
    def __init__(self, split):
        base_path = "data/p2/ffhq"
        super().__init__(base_path, split)


class P2CelebAHQDataset(TemplateDataset):
    def __init__(self, split):
        base_path = "data/p2/celebahq"
        super().__init__(base_path, split)


class CelebAHQDataset(MyDataset):
    def __init__(self, split, transform=None):
        super().__init__()
        self.base_path = "data/repaint"
        self.split = split
        self.metadata = self.load_metadata(split)
        self.transform = transform

    def load_metadata(self, split):
        path = os.path.join(
            self.base_path, "data", "datasets", f"metadata-{split}.json"
        )
        with open(path, "r") as f:
            return json.load(f)

    def get_file_name(self, i):
        return self.metadata[i]["key"]

    def get_mask_path(self, i):
        key = self.get_file_name(i)
        folder = os.path.join(self.base_path, "inpainted", "celebahq")
        return os.path.join(folder, "gt_keep_mask", key + ".png")

    def load_mask(self, i):
        mask_path = self.get_mask_path(i)
        mask = np.array(Image.open(mask_path))
        mask = 1 - (mask[:, :, 0] == 255).astype("float")
        return mask

    def __getitem__(self, i):
        if i >= len(self):
            return IndexError

        image_path = self.get_image_path(i)

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = self.load_mask(i)

        sample = dict(image=image, mask=mask)

        if self.transform is not None:
            sample = self.transform(**sample)

        sample["image"] = np.moveaxis(sample["image"], -1, 0)
        sample["mask"] = np.expand_dims(sample["mask"], 0)
        # pdb.set_trace()

        return sample

    def __len__(self):
        return len(self.metadata)


class RepaintDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        super().__init__(split, transform)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = os.path.join(self.base_path, "inpainted", "celebahq")
        return os.path.join(folder, "inpainted", key + ".png")


class RepaintCleanDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        super().__init__(split, transform)

    def load_metadata(self, split):
        path = os.path.join(
            self.base_path, "data", "datasets", f"metadata-{split}.json"
        )
        with open(path, "r") as f:
            return json.load(f)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = os.path.join(self.base_path, "inpainted", "celebahq")
        return os.path.join(folder, "clean_repaint", self.split, key + ".png")


# class RepaintV2CleanDataset(TemplateDataset):
#     def __init__(self, split):
#         base_path = "data/repaint-v2/inpainted/celebahq-v2/clean_repaint"
#         super().__init__(base_path, split)


class RepaintV2CleanDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.base_path = "data/repaint-v2"
        self.metadata = self.load_metadata(split)
        self.transform = transform

    def load_metadata(self, split):
        path = f"data/repaint-v2/inpainted/celebahq-v2/metadata-{split}.json"
        with open(path, "r") as f:
            return json.load(f)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = "data/repaint-v2/inpainted/celebahq-v2"
        return os.path.join(folder, "clean_repaint", self.split, key + ".png")


class LamaDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        super().__init__(split, transform)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = "/home/doneata/src/lama/outputs/celebahq"
        return os.path.join(folder, key + "_mask000.png")


class PluralisticDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        super().__init__(split, transform)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = "/home/doneata/src/pluralistic-inpainting/results"
        return os.path.join(folder, key + "_out_0.png")


class LDMRepaintDataset(CelebAHQDataset):
    def __init__(self, split, transform=None):
        super().__init__(split, transform)

    def get_image_path(self, i):
        key = self.get_file_name(i)
        folder = Path("data/ldm-repaint/celebahq")
        return str(folder / self.split / (key + ".png"))


class FaceAppDataset(MyDataset):
    def __init__(self, split, transform=None):
        super().__init__()
        self.base_path = "data/faceapp"
        self.transform = transform

        if split == "valid":
            self.split = "validation"
        else:
            self.split = split

        self.metadata = self.load_metadata(self.split)

    def load_metadata(self, split):
        path = os.path.join(self.base_path, f"{split}.txt")
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def get_file_name(self, i):
        file_ = self.metadata[i]
        split, file_ = file_.split("/")
        file_, _ = file_.split(".")
        return file_

    def get_image_path(self, i):
        file_ = self.metadata[i]
        return os.path.join(self.base_path, file_)

    def get_mask_path(self, i):
        file_ = self.metadata[i]
        split, file_ = file_.split("/")
        return os.path.join(self.base_path, split + "_mask", file_)

    def load_image(self, i):
        image_path = self.get_image_path(i)
        image = Image.open(image_path)
        image = image.resize((256, 256), Image.Resampling.LANCZOS)
        image = np.array(image)
        return image

    def load_mask(self, i):
        mask_path = self.get_mask_path(i)
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.Resampling.LANCZOS)
        mask = np.array(mask)
        mask = (mask[:, :, 0] / 255 > 0.1).astype("float")
        return mask

    def __getitem__(self, i):
        if i >= len(self):
            return IndexError

        image = self.load_image(i)
        mask = self.load_mask(i)

        sample = dict(image=image, mask=mask)

        sample["image"] = np.moveaxis(sample["image"], -1, 0)
        sample["mask"] = np.expand_dims(sample["mask"], 0)

        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
            sample["mask"] = self.transform(sample["mask"])

        return sample

    def __len__(self):
        return len(self.metadata)


def load_mask_keep(path):
    mask = np.array(Image.open(path))
    mask = 1 - (mask[:, :, 0] == 255).astype("float")
    return mask


class PathSplitDataset:
    def __init__(self, path_images, path_masks, split):
        self.path_images = path_images
        self.path_masks = path_masks
        self.split = split
        self.files = self.load_filelist(split)

    def load_filelist(self, split):
        return os.listdir(self.path_images / split)

    def get_file_name(self, i):
        return Path(self.files[i]).stem

    def get_image_path(self, i):
        return str(self.path_images / self.split / self.files[i])

    def get_mask_path(self, i):
        return str(self.path_masks / self.split / self.files[i])

    def load_mask(self, i):
        return load_mask_keep(self.get_mask_path(i))

    def __len__(self):
        return len(self.files)


class RepaintP2CelebAHQCleanDataset(PathSplitDataset):
    def __init__(self, split):
        path_base = Path("data/repaint/p2/celebahq/fake")
        super().__init__(
            path_images=path_base / "clean",
            path_masks=path_base / "ground_truth" / "mask",
            split=split,
        )


class RepaintP2FFHQCleanDataset(PathSplitDataset):
    def __init__(self, split):
        path_base = Path("data/repaint/p2/ffhq/fake")
        super().__init__(
            path_images=path_base / "clean",
            path_masks=path_base / "ground_truth" / "mask",
            split=split,
        )


class RepaintP2CelebAHQCleanSmallDataset(PathSplitDataset):
    def __init__(self, split):
        path_base = Path("data/repaint/p2/celebahq/small/fake")
        super().__init__(
            path_images=path_base / "clean",
            path_masks=path_base / "ground_truth" / "mask",
            split=split,
        )


class RepaintP2CelebAHQClean9KDataset(PathSplitDataset):
    def __init__(self, split):
        path_base = Path("data/repaint/p2/celebahq/fake-9k")
        super().__init__(
            path_images=path_base / "clean",
            path_masks=path_base / "ground_truth" / "mask",
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