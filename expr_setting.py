from termcolor import colored, cprint
import sys

from configs.config import (
    DataConfig,
    NetConfig,
)


class ExprSetting(object):
    def __init__(self):

        self.model_class = self.dynamic_models()
        self.dataloader_class = self.dynamic_dataloaders()
        self.lr_logger, self.model_checkpoint = self.checkpoint_setting()
        self.early_stop = self.earlystop_setting()

    def checkpoint_setting(self):
        from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

        lr_logger = LearningRateMonitor(logging_interval="epoch")

        model_checkpoint = ModelCheckpoint(
            filename="{epoch}-{val_loss:.2f}-{user_metric:.2f}",
            save_last=True,
            save_weights_only=True,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        return lr_logger, model_checkpoint

    def earlystop_setting(self):
        # ## https://www.youtube.com/watch?v=vfB5Ax6ekHo
        from pytorch_lightning.callbacks import EarlyStopping

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=2000000,
            strict=False,
            verbose=False,
            mode="min",
        )
        return early_stop

    def dynamic_dataloaders(self):

        if DataConfig.dataloader_name == "dataloader":
            from dataloader import UniDataloader
        else:
            sys.eixt("Please check your dataloader name in config file ... ")

        UsedUniDataloader = UniDataloader()
        return UsedUniDataloader

    def dynamic_models(self):
        if NetConfig.model_name == "model":
            from model import Model
        else:
            sys.eixt("Please check your model name in config file ... ")

        UniModel = Model
        return UniModel
