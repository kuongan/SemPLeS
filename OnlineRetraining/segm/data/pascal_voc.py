# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pathlib import Path

from segm.data.base import BaseMMSeg
from segm.data import utils
from segm.config import dataset_dir

PASCAL_VOC_CONFIG_PATH = Path(__file__).parent / "config" / "pascal_voc.py"
PASCAL_VOC_CATS_PATH = Path(__file__).parent / "config" / "pascal_voc.yml"


class PascalVOCDataset(BaseMMSeg):
    def __init__(self, image_size, crop_size, split, ann_dir=None, eval_split=None, **kwargs):
        self.ann_dir = ann_dir
        self.eval_split = eval_split
        super().__init__(
            image_size, crop_size, split, PASCAL_VOC_CONFIG_PATH, **kwargs
        )
        self.names, self.colors = utils.dataset_cat_description(
            PASCAL_VOC_CATS_PATH
        )
        self.n_cls = 21
        self.ignore_label = 255
        self.reduce_zero_label = False

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "voc12"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path / "VOCdevkit/VOC2012/"
            if self.ann_dir:
                config.data.train.ann_dir = self.ann_dir
        elif self.split == "val":
            config.data.val.data_root = path / "VOCdevkit/VOC2012/"
            if self.eval_split is not None:
                config.data.val.split = self.eval_split
        elif self.split == "test":
            # raise ValueError("Test split is not valid for Pascal Context dataset")
            config.data.test.data_root = path / "VOCdevkit/VOC2012/"
            if self.eval_split is not None:
                config.data.test.split = self.eval_split
        config = super().update_default_config(config)
        return config

    def test_post_process(self, labels):
        return labels