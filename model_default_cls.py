import copy
import math
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
)

from configs.config import DataConfig, NetConfig, TrainingConfig
from dataloader import UniDataloader
from model_utils import get_classification_metrics
from utils import CosineAnnealingWarmRestartsDecay, console_logger_start


class DefaultModel(pl.LightningModule):
    def __init__(self, logger=None):
        super(DefaultModel, self).__init__()

        self.n_logger = logger
        self.console_logger = console_logger_start()
        self.UsedUniDataloader = UniDataloader()
        self.val_scales = 1
        self.test_scales = 1
        self.ignore = DataConfig.class_ignore
        self.log_config_step = dict(
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log_config_epoch = dict(
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.module_lr_dict = dict(placeholder=0.0)

        (
            self.train_accuracy,
            self.train_precision,
            self.train_recall,
            self.val_accuracy,
            self.val_precision,
            self.val_recall,
            self.test_accuracy,
            self.test_precision,
            self.test_recall,
        ) = get_classification_metrics(DataConfig.num_classes, self.ignore)

    def train_dataloader(self):

        return self.UsedUniDataloader.get_train_dataloader()

    def val_dataloader(self):

        return self.UsedUniDataloader.get_val_dataloader()

    def lr_logging(self):

        """>>> Capture the learning rates and log it out using logger;"""
        lightning_optimizer = self.optimizers()
        param_groups = lightning_optimizer.optimizer.param_groups

        for param_group_idx in range(len(param_groups)):

            sub_param_group = param_groups[param_group_idx]

            """>>>
            print("==>> sub_param_group: ", sub_param_group.keys())
            # ==>> sub_param_group:  dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov', 'maximize', 'initial_lr'])
            """

            sub_lr_name = "lr/lr_" + str(param_group_idx)
            """>>>
            print("lr: {}, {}".format(sub_lr_name, sub_param_group["lr"]))
            # lr: lr_0, 0.001
            # lr: lr_1, 0.08
            """

            self.log(
                sub_lr_name,
                sub_param_group["lr"],
                batch_size=TrainingConfig.batch_size,
                **self.log_config_step,
            )

    def poly_lr_scheduler(
        self, optimizer, init_lr, iter, lr_decay_iter=1, max_iter=1000, power=0.9
    ):
        """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

        """
        if iter % lr_decay_iter or iter > max_iter:
            return optimizer

        lr = init_lr * (1 - iter / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def configure_optimizers(self):

        optimizer = self.get_optim()

        if TrainingConfig.pretrained_weights:
            max_epochs = TrainingConfig.pretrained_weights_max_epoch
        else:
            max_epochs = TrainingConfig.max_epochs

        if TrainingConfig.scheduler == "cosineAnn":
            eta_min = TrainingConfig.eta_min

            T_max = max_epochs
            last_epoch = -1

            sch = CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "cosineAnnWarm":

            last_epoch = -1
            eta_min = TrainingConfig.eta_min
            T_mult = (max_epochs // TrainingConfig.T_0) - 1

            sch = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=TrainingConfig.T_0,
                T_mult=T_mult,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "cosineAnnWarmDecay":
            last_epoch = -1
            eta_min = TrainingConfig.eta_min
            decay = TrainingConfig.decay

            sch = CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=TrainingConfig.T_0,
                T_mult=TrainingConfig.T_mult,
                eta_min=eta_min,
                last_epoch=last_epoch,
                decay=decay,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }

        elif TrainingConfig.scheduler == "CosineAnnealingLR":
            steps = 10
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "step":

            step_size = int(TrainingConfig.step_ratio * max_epochs)
            gamma = TrainingConfig.gamma
            sch = StepLR(optimizer, step_size=step_size, gamma=gamma)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch, "monitor": "train_loss"},
            }
        elif TrainingConfig.scheduler == "none":
            return optimizer

    def different_lr(self, module_lr_dict, lr):
        def is_key_included(module_name, n):
            return module_name in n

        def is_any_key_match(module_lr_dict, n):
            indicator = False
            for key in module_lr_dict.keys():
                if key in n:
                    indicator = True
            return indicator

        params = list(self.named_parameters())

        grouped_parameters = [
            {
                "params": [
                    p for n, p in params if not is_any_key_match(module_lr_dict, n)
                ],
                "lr": lr,
            },
        ]

        for key in module_lr_dict.keys():
            sub_param_list = []
            for n, p in params:
                if is_key_included(key, n):
                    if module_lr_dict[key] == 0.0:
                        p.requires_grad = False
                    sub_param_list.append(p)
            sub_parameters = {
                "params": sub_param_list,
                "lr": module_lr_dict[key],
            }
            grouped_parameters.append(sub_parameters)

        return grouped_parameters

    def get_optim(self):

        lr = NetConfig.lr

        if TrainingConfig.pretrained_weights:
            lr = TrainingConfig.pre_lr

        if not hasattr(torch.optim, NetConfig.opt):
            print("Optimiser {} not supported".format(NetConfig.opt))
            raise NotImplementedError

        optim = getattr(torch.optim, NetConfig.opt)

        if TrainingConfig.strategy == "colossalai":
            from colossalai.nn.optimizer import HybridAdam

            optimizer = HybridAdam(self.parameters(), lr=lr)
        else:

            if TrainingConfig.single_lr:
                print("Using a single learning rate for all parameters")
                grouped_parameters = [{"params": self.parameters()}]
            else:
                print("Using different learning rates for all parameters")
                grouped_parameters = self.different_lr(self.module_lr_dict, lr)

            # print("\n ==>> grouped_parameters: \n", grouped_parameters)

            if NetConfig.opt == "Adam":

                lr = 0.001
                betas = (0.9, 0.999)
                eps = 1e-08
                weight_decay = 0.0

                optimizer = torch.optim.Adam(
                    grouped_parameters,
                    lr=lr,
                    betas=betas,
                    eps=1e-08,
                    weight_decay=weight_decay,
                )
            elif NetConfig.opt == "Lamb":
                weight_decay = 0.02
                betas = (0.9, 0.999)

                optimizer = torch.optim.Lamb(
                    grouped_parameters, lr=lr, betas=betas, weight_decay=weight_decay
                )
            elif NetConfig.opt == "AdamW":
                eps = 1e-8
                betas = (0.9, 0.999)
                weight_decay = 0.05

                optimizer = torch.optim.AdamW(
                    grouped_parameters,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=weight_decay,
                )
            elif NetConfig.opt == "SGD":
                momentum = 0.9
                weight_decay = 0.0001

                optimizer = torch.optim.SGD(
                    grouped_parameters,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )

            else:
                optimizer = optim(grouped_parameters, lr=NetConfig.lr)

        optimizer.zero_grad()

        return optimizer

    # def test_step(self, batch, batch_idx):

    #     """>>> C15: 处理数据,跟training部分一样;"""
    #     data_dict = batch
    #     self.test_scales = len(data_dict["img"])
    #     test_loss_ms = 0
    #     for scale_idx in range(self.test_scales):
    #         images, masks = (
    #             data_dict["img"][scale_idx].to(self.device),
    #             data_dict["gt_semantic_seg"][scale_idx].long().to(self.device),
    #         )
    #         if len(masks.shape) == 4:
    #             masks = masks.squeeze(1)
    #         if self.ignore:
    #             masks[masks == torch.tensor(255).to(self.device)] = torch.tensor(
    #                 DataConfig.num_classes
    #             ).to(self.device)

    #         """>>> C16: 计算model的输出和validation loss; """
    #         model_output, _ = self.forward(images, masks)
    #         model_output = F.interpolate(
    #             model_output,
    #             [masks.shape[-2], masks.shape[-1]],
    #             mode="bilinear",
    #             align_corners=False,
    #         )
    #         model_predictions = model_output.argmax(dim=1)
    #         test_loss_out = self.criterion_seg(model_output, masks)
    #         test_loss = test_loss_out

    #         test_loss_ms += test_loss

    #         self.test_iou.update(model_predictions, masks)
    #         self.test_precision.update(model_predictions, masks)
    #         self.test_recall.update(model_predictions, masks)

    #     self.log(
    #         "test_loss",
    #         test_loss_ms,
    #         on_step=True,
    #         on_epoch=True,
    #         batch_size=ValidationConfig.batch_size * self.test_scales,
    #     )

    # def test_epoch_end(self, outputs):

    #     """>>> C16.1: compute()函数得到的结果是list;"""
    #     test_epoch_iou = self.test_iou.compute()
    #     test_epoch_precision = self.test_precision.compute()
    #     test_epoch_recall = self.test_recall.compute()

    #     """>>> C18: 计算最终结果的时候,如果有ignore的class的话,最后一个class的参与计算metric; """
    #     if self.ignore:
    #         test_epoch_precision_mean = torch.mean(test_epoch_precision[:-1]).item()
    #         test_epoch_recall_mean = torch.mean(test_epoch_recall[:-1]).item()
    #         test_epoch_iou_mean = torch.mean(test_epoch_iou[:-1]).item()
    #     else:
    #         test_epoch_precision_mean = torch.mean(test_epoch_precision).item()
    #         test_epoch_recall_mean = torch.mean(test_epoch_recall).item()
    #         test_epoch_iou_mean = torch.mean(test_epoch_iou).item()

    #     self.log(
    #         "test_iou_epoch",
    #         test_epoch_iou_mean,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #         batch_size=ValidationConfig.batch_size * self.test_scales,
    #     )

    #     """>>> C19: 实验结果只需要在一个gpu上plot就可以了,所有GPU上的结果是一样的; """
    #     if self.global_rank == 0:
    #         self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

    #         # assert (
    #         #     DataConfig.num_classes == test_epoch_iou.shape[0]
    #         # ), "Something is wrong."

    #         for i in range(DataConfig.num_classes):
    #             self.console_logger.info(
    #                 "{0: <15}, iou: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
    #                     DataConfig.classes[i],
    #                     test_epoch_iou[i].item(),
    #                     test_epoch_precision[i].item(),
    #                     test_epoch_recall[i].item(),
    #                 )
    #             )
    #         self.console_logger.info("iou_mean: {0:.4f} ".format(test_epoch_iou_mean))

    #         self.console_logger.info(
    #             "precision_mean: {0:.4f} ".format(test_epoch_precision_mean)
    #         )
    #         self.console_logger.info(
    #             "recall_mean: {0:.4f} ".format(test_epoch_recall_mean)
    #         )

    #     self.test_iou.reset()
    #     self.test_precision.reset()
    #     self.test_recall.reset()
