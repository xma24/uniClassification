import os

import torch
from rich import print

from configs.config import DataConfig, NetConfig, TrainingConfig, ValidationConfig
from model_default_cls import DefaultModel
from model_utils import (
    get_resnet_backbone,
    get_resnet_head,
    get_resnet_model,
    get_resnet_neck,
)


class Model(DefaultModel):
    def __init__(self, logger=None):
        super(Model, self).__init__()

        self.module_lr_dict = dict(placeholder=0.0)

        self.model = get_resnet_model()
        self.backbone = get_resnet_backbone(self.model)
        self.neck = get_resnet_neck(self.model)
        self.head = get_resnet_head(self.model)
        del self.model

    def forward(self, images, labels=None, epoch=None, batch_idx=None):

        x = self.backbone(images)

        x = self.neck(x)

        x = self.head.pre_logits(x)

        output = self.head.fc(x)

        return output

    def training_step(self, batch, batch_idx):

        self.lr_logging()

        data_dict = batch
        images, gt_labels = (
            data_dict["img"].to(self.device),
            data_dict["gt_label"].to(self.device),
        )

        model_outputs = self.forward(images)
        train_losses_dict = self.head.loss(model_outputs, gt_labels)

        train_loss = train_losses_dict["loss"]

        losses = {"loss": train_loss}

        self.log(
            "train_loss",
            train_loss,
            batch_size=TrainingConfig.batch_size,
            **self.log_config_step,
        )

        model_predictions = model_outputs.argmax(dim=1)

        self.train_accuracy.update(model_predictions, gt_labels)
        self.train_precision.update(model_predictions, gt_labels)
        self.train_recall.update(model_predictions, gt_labels)

        if batch_idx % 10 == 0:
            self.console_logger.info(
                "epoch: {0:04d} | loss_train: {1:.4f}".format(
                    self.current_epoch, losses["loss"]
                )
            )
        if TrainingConfig.use_ema:
            self.model_ema.update(self.model)

        return {"loss": losses["loss"]}

    def training_epoch_end(self, outputs):

        cwd = os.getcwd()
        print("==>> Expriment Folder: ", cwd)

        train_accuracy = self.train_accuracy.compute()

        """>>> If self.ignore is used, the last clss is fake and will not calculated in the metric; """
        if self.ignore:
            train_accuracy_mean = torch.mean(train_accuracy[:-1]).item()
        else:
            train_accuracy_mean = torch.mean(train_accuracy).item()

        self.log(
            "train_accuracy_epoch",
            train_accuracy_mean,
            batch_size=TrainingConfig.batch_size,
            **self.log_config_epoch,
        )

        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def validation_step(self, batch, batch_idx):

        data_dict = batch
        images, gt_labels = (
            data_dict["img"].to(self.device),
            data_dict["gt_label"].to(self.device),
        )

        model_outputs = self.forward(images)
        val_losses_dict = self.head.loss(model_outputs, gt_labels)
        val_loss = val_losses_dict["loss"]

        model_predictions = model_outputs.argmax(dim=1)

        self.val_accuracy.update(model_predictions, gt_labels)
        self.val_precision.update(model_predictions, gt_labels)
        self.val_recall.update(model_predictions, gt_labels)

        self.log(
            "val_loss",
            val_loss,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_step,
        )

        return val_loss

    def validation_epoch_end(self, outputs):

        """>>> The compute() function will return a list;"""
        val_epoch_accuracy = self.val_accuracy.compute()
        val_epoch_precision = self.val_precision.compute()
        val_epoch_recall = self.val_recall.compute()

        if self.ignore:
            val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy[:-1].item())
            val_epoch_precision_mean = torch.mean(val_epoch_precision[:-1]).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall[:-1]).item()
        else:
            val_epoch_accuracy_mean = torch.mean(val_epoch_accuracy).item()
            val_epoch_precision_mean = torch.mean(val_epoch_precision).item()
            val_epoch_recall_mean = torch.mean(val_epoch_recall).item()

        self.log(
            "val_epoch_accuracy",
            val_epoch_accuracy_mean,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> We use the "user_metric" variable to monitor the performance on val set; """
        user_metric = val_epoch_accuracy_mean
        self.log(
            "user_metric",
            user_metric,
            batch_size=ValidationConfig.batch_size * self.val_scales,
            **self.log_config_epoch,
        )

        """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
        if self.global_rank == 0:
            self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

            for i in range(DataConfig.num_classes):
                self.console_logger.info(
                    "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
                        DataConfig.classes[i],
                        val_epoch_accuracy[i].item(),
                        val_epoch_precision[i].item(),
                        val_epoch_recall[i].item(),
                    )
                )
            self.console_logger.info(
                "acc_mean: {0:.4f} ".format(val_epoch_accuracy_mean)
            )

            self.console_logger.info(
                "precision_mean: {0:.4f} ".format(val_epoch_precision_mean)
            )
            self.console_logger.info(
                "recall_mean: {0:.4f} ".format(val_epoch_recall_mean)
            )

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    # def test_step(self, batch, batch_idx):

    #     data_dict = batch
    #     images, gt_labels = (
    #         data_dict["img"].to(self.device),
    #         data_dict["gt_label"].to(self.device),
    #     )

    #     model_outputs = self.forward(images)
    #     test_losses_dict = self.head.loss(model_outputs, gt_labels)
    #     test_loss = test_losses_dict["loss"]

    #     model_predictions = model_outputs.argmax(dim=1)

    #     self.test_accuracy.update(model_predictions, gt_labels)
    #     self.test_precision.update(model_predictions, gt_labels)
    #     self.test_recall.update(model_predictions, gt_labels)

    #     self.log(
    #         "test_loss",
    #         test_loss,
    #         on_step=True,
    #         on_epoch=True,
    #         batch_size=ValidationConfig.batch_size * self.test_scales,
    #     )

    #     return test_loss

    # def test_epoch_end(self, outputs):

    #     """>>> The compute() function will return a list;"""
    #     test_epoch_accuracy = self.test_accuracy.compute()
    #     test_epoch_precision = self.test_precision.compute()
    #     test_epoch_recall = self.test_recall.compute()

    #     if self.ignore:
    #         test_epoch_accuracy_mean = torch.mean(test_epoch_accuracy[:-1].item())
    #         test_epoch_precision_mean = torch.mean(test_epoch_precision[:-1]).item()
    #         test_epoch_recall_mean = torch.mean(test_epoch_recall[:-1]).item()
    #     else:
    #         test_epoch_accuracy_mean = torch.mean(test_epoch_accuracy).item()
    #         test_epoch_precision_mean = torch.mean(test_epoch_precision).item()
    #         test_epoch_recall_mean = torch.mean(test_epoch_recall).item()

    #     self.log(
    #         "test_epoch_accuracy",
    #         test_epoch_accuracy_mean,
    #         prog_bar=True,
    #         logger=True,
    #         sync_dist=True,
    #         batch_size=ValidationConfig.batch_size * self.test_scales,
    #     )

    #     """>>> We use the "user_metric" variable to monitor the performance on val set; """
    #     user_metric = test_epoch_accuracy_mean
    #     self.log(
    #         "user_metric",
    #         user_metric,
    #         on_step=False,
    #         on_epoch=True,
    #         batch_size=ValidationConfig.batch_size * self.test_scales,
    #     )

    #     """>>> Plot the results to console using only one gpu since the results on all gpus are the same; """
    #     if self.global_rank == 0:
    #         self.console_logger.info("epoch: {0:04d} ".format(self.current_epoch))

    #         for i in range(DataConfig.num_classes):
    #             self.console_logger.info(
    #                 "{0: <15}, acc: {1:.4f} | precision: {2:.4f} | recall: {3:.4f}".format(
    #                     DataConfig.classes[i],
    #                     test_epoch_accuracy[i].item(),
    #                     test_epoch_precision[i].item(),
    #                     test_epoch_recall[i].item(),
    #                 )
    #             )
    #         self.console_logger.info(
    #             "acc_mean: {0:.4f} ".format(test_epoch_accuracy_mean)
    #         )

    #         self.console_logger.info(
    #             "precision_mean: {0:.4f} ".format(test_epoch_precision_mean)
    #         )
    #         self.console_logger.info(
    #             "recall_mean: {0:.4f} ".format(test_epoch_recall_mean)
    #         )

    #     self.test_accuracy.reset()
    #     self.test_precision.reset()
    #     self.test_recall.reset()
