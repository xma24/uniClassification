import os
import pickle
import random
from functools import partial

import albumentations as A
import numpy as np
import torch
import torchmetrics
import torchvision.transforms as transforms
import torchvision.transforms as tf
import torchvision.transforms.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from mmcv.parallel import collate
from PIL import Image, ImagePalette
from timm.data.auto_augment import rand_augment_transform

from configs.config import (
    DataConfig,
    ExtraDatasetConfig,
    NetConfig,
    TestingConfig,
    TrainingConfig,
    ValidationConfig,
)

from datasets.datasets_tiny_imagenet_200 import TinyImageNet


class UniDataloader:
    def __init__(self):
        super(UniDataloader, self).__init__()

    def get_dataloader(self, split, num_gpus):

        data_config = {}

        if split == "train":
            shuffle = True
            drop_last = True
            batch_size = TrainingConfig.batch_size

            train_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=DataConfig.image_size),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                    ),
                    A.RandomCrop(
                        height=DataConfig.image_crop_size,
                        width=DataConfig.image_crop_size,
                    ),
                    A.RGBShift(
                        r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

            dataset = TinyImageNet(
                DataConfig.data_root, train=True, transform=train_transform
            )

            if TrainingConfig.subtrain:
                sampling_pool = np.arange(len(dataset))
                np.random.shuffle(sampling_pool)
                num_sampling = int(TrainingConfig.subtrain_ratio * len(dataset))
                sublist = list(sampling_pool[:num_sampling])
                dataset = torch.utils.data.Subset(dataset, sublist)
                print("==>> sampled dataset: ", len(dataset))

        elif split == "val":
            shuffle = False
            drop_last = False
            batch_size = ValidationConfig.batch_size

            val_transform = A.Compose(
                [
                    A.SmallestMaxSize(max_size=DataConfig.image_size),
                    A.CenterCrop(
                        height=DataConfig.image_crop_size,
                        width=DataConfig.image_crop_size,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

            dataset = TinyImageNet(
                DataConfig.data_root, train=False, transform=val_transform
            )

            if ValidationConfig.sub_val:
                sampling_pool = np.arange(len(dataset))
                np.random.shuffle(sampling_pool)
                num_sampling = int(ValidationConfig.subval_ratio * len(dataset))
                sublist = list(sampling_pool[:num_sampling])
                dataset = torch.utils.data.Subset(dataset, sublist)
                print("==>> sampled dataset: ", len(dataset))

        elif split == "test":

            shuffle = False
            drop_last = False
            batch_size = TestingConfig.batch_size

            dataset = TinyImageNet(DataConfig.data_root, train=True)

            if ValidationConfig.sub_val:
                sampling_pool = np.arange(len(dataset))
                np.random.shuffle(sampling_pool)
                num_sampling = int(ValidationConfig.subval_ratio * len(dataset))
                sublist = list(sampling_pool[:num_sampling])
                dataset = torch.utils.data.Subset(dataset, sublist)
                print("==>> sampled dataset: ", len(dataset))

        else:
            print("\n The split is not valid.\n")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=DataConfig.workers,
            collate_fn=collate,
            pin_memory=DataConfig.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=True,
            # sampler=sampler,
            # collate_fn=partial(collate, samples_per_gpu=batch_size),
            # worker_init_fn=init_fn,
        )
        return dataloader

    def get_train_dataloader(self, num_gpus=1):

        self.train_loader = self.get_dataloader("train", num_gpus)

        return self.train_loader

    def get_val_dataloader(self, num_gpus=1):

        self.val_loader = self.get_dataloader("val", num_gpus)

        return self.val_loader

    def get_test_dataloader(self, num_gpus=1):

        self.test_loader = self.get_dataloader("test", num_gpus)

        return self.test_loader


if __name__ == "__main__":

    dataloader_class = UniDataloader()
    train_dataloader = dataloader_class.get_train_dataloader(num_gpus=4)
    print("train_dataloder: {}".format(len(train_dataloader)))
    """>>>
    # train_dataloder: 185
    """

    val_dataloader = dataloader_class.get_val_dataloader(num_gpus=4)
    print("val_dataloader: {}".format(len(val_dataloader)))
    """>>>
    # val_dataloader: 32
    """

    # test_dataloader = dataloader_class.get_test_dataloader(num_gpus=4)
    # print("test_dataloader: {}".format(len(test_dataloader)))
    # """>>>
    # # test_dataloader: 32
    # """

    print("\n Checking Train Dataloader \n ")
    for batch_idx, data_dict in enumerate(train_dataloader):
        """>>>"""

        img, gt_label = data_dict["img"], data_dict["gt_label"]

        """>>>

        """
        # print("==>> img_metas: ", img_metas)
        print("==>> img.shape: ", img.shape)
        print("==>> gt_label.shape: ", gt_label)

        break

    print("\n Checking Val Dataloader \n")
    for batch_idx, data_dict in enumerate(val_dataloader):

        img, gt_label = data_dict["img"], data_dict["gt_label"]

        """>>>

        """
        # print("==>> img_metas: ", img_metas)
        print("==>> img.shape: ", img.shape)
        print("==>> gt_label.shape: ", gt_label)

        import sys

        sys.exit()
