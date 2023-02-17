import datetime
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import pytz
import torch
import torch.backends.cudnn as cudnn
import torch.onnx
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import QuantizationAwareTraining
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from rich import print

import wandb
from expr_setting import ExprSetting

cudnn.benchmark = False


matplotlib.use("Agg")

plt.style.use("ggplot")

warnings.filterwarnings("ignore")

from configs.config import (
    DataConfig,
    NetConfig,
    TestingConfig,
    TrainingConfig,
    ValidationConfig,
)


def load_pre_weights(model):
    if TrainingConfig.pretrained_weights:
        checkpoint = torch.load(TrainingConfig.pretrained_weights_path)
        print("init keys of checkpoint:{}".format(checkpoint.keys()))

        state_dict = checkpoint["state_dict"]
        print("init keys :{}".format(state_dict.keys()))

        # updated_state_dict = {}
        # for k in list(state_dict.keys()):
        #     updated_state_dict["model." + k] = state_dict[k]
        # print("==>> updated_state_dict: ", updated_state_dict.keys())

        updated_state_dict = state_dict

        model.load_state_dict(updated_state_dict, strict=True)

        TrainingConfig.max_epochs = TrainingConfig.pretrained_weights_max_epoch

        print("\n Using pretrained weights ... \n")
    else:
        print("\n No pretrained weights are used ... \n")

    return model

    # if DeeplabV3PlusConfig.pretrained_cls_weights:
    #     """>>> C4: 查看一下model里面的模块都有哪些; 方便之后改写加载pretrained weights的部分;"""
    #     # backbone_list = list(backbone.children())
    #     # print("==>> backbone_list: ", backbone_list)

    #     """>>> C5: segmentation使用的backbone一般都是在imagenet上pretrained; 在加载pretrained weights的时候一定要注意key是否match; """
    #     state_dict = torch.load(DeeplabV3PlusConfig.pretrained_cls_weights_path)[
    #         "state_dict"
    #     ]
    #     # print("==>> state_dict: ", state_dict.keys())

    #     """>>> C6: 因为segmentation使用的backbone只是进行feature提取,没有涉及到classification, 所以要把FC这些key和value去除; """
    #     backbone_state_dict = {}
    #     fc_state_dict = {}
    #     for k in list(state_dict.keys()):
    #         if k.startswith("fc."):
    #             fc_state_dict[k[len("fc.") :]] = state_dict[k]
    #         else:
    #             backbone_state_dict[k] = state_dict[k]
    #         del state_dict[k]
    #     """>>> C7: 为了避免model加载pretrained weights错误, load_state_dict()函数里面的strict一定要设置成True; 如果出错的话,就会报错,方便debug; """
    #     model.backbone.load_state_dict(backbone_state_dict, strict=True)

    #     print("\n Using pretrained CLS BACKBONE weights ... \n")


def init_expr_config():

    expr_setting = ExprSetting()

    lr_logger, model_checkpoint, early_stop, model_class, dataloader_class = (
        expr_setting.lr_logger,
        expr_setting.model_checkpoint,
        expr_setting.early_stop,
        expr_setting.model_class,
        expr_setting.dataloader_class,
    )

    os.makedirs(DataConfig.work_dirs, exist_ok=True)

    seed_everything(DataConfig.random_seed)

    if DataConfig.classes is None:
        DataConfig.classes = np.arange(DataConfig.num_classes)

    if isinstance(TrainingConfig.num_gpus, int):
        num_gpus = TrainingConfig.num_gpus
    elif TrainingConfig.num_gpus == "autocount":
        TrainingConfig.num_gpus = torch.cuda.device_count()
        num_gpus = TrainingConfig.num_gpus
    else:
        gpu_list = TrainingConfig.num_gpus.split(",")
        num_gpus = len(gpu_list)

    if TrainingConfig.logger_name == "neptune":
        print("Not implemented")
        exit(0)
    elif TrainingConfig.logger_name == "csv":
        own_logger = CSVLogger(DataConfig.logger_root)
    elif TrainingConfig.logger_name == "wandb":
        run_name = datetime.datetime.now(tz=pytz.timezone("US/Central")).strftime(
            "%Y-%m-%d-%H-%M"
        )
        own_logger = WandbLogger(
            project=TrainingConfig.wandb_name,
            settings=wandb.Settings(code_dir="."),
            name=run_name,
        )
    else:
        own_logger = CSVLogger(DataConfig.logger_root)

    if TrainingConfig.training_mode == False:
        num_gpus = 1
    print("num of gpus: {}".format(num_gpus))

    return lr_logger, model_checkpoint, early_stop, model_class, own_logger, num_gpus


def pl_trainer_config(
    lr_logger, model_checkpoint, early_stop, model_class, own_logger, num_gpus
):
    """
    - The setting of pytorch lightning Trainer:
        (https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/trainer/trainer.py)
    """
    if TrainingConfig.cpus:
        print("using CPUs to do experiments ... ")
        trainer = pl.Trainer(
            num_nodes=TrainingConfig.num_nodes,
            precision=TrainingConfig.precision,
            accelerator="cpu",
            strategy=DDPStrategy(find_unused_parameters=True),
            # profiler="pytorch",
            logger=own_logger,
            callbacks=[lr_logger, model_checkpoint],
            log_every_n_steps=1,
            check_val_every_n_epoch=ValidationConfig.val_interval,
            auto_scale_batch_size="binsearch",
            replace_sampler_ddp=False,
        )
    else:
        if TrainingConfig.strategy == "ddp" or TrainingConfig.strategy == "ddp_sharded":

            print("using {} to do experiments; ".format(TrainingConfig.strategy))

            trainer = pl.Trainer(
                devices=num_gpus,
                num_nodes=TrainingConfig.num_nodes,
                precision=TrainingConfig.precision,
                accelerator=TrainingConfig.accelerator,
                strategy=TrainingConfig.strategy,
                logger=own_logger,
                callbacks=[lr_logger, model_checkpoint],
                log_every_n_steps=1,
                max_epochs=TrainingConfig.max_epochs,
                check_val_every_n_epoch=ValidationConfig.val_interval,
                auto_scale_batch_size="binsearch",
                # resume_from_checkpoint="",
                # sync_batchnorm=True if num_gpus > 1 else False,
                # plugins=DDPPlugin(find_unused_parameters=False),
                # track_grad_norm=1,
                # progress_bar_refresh_rate=TrainingConfig.progress_bar_refresh_rate,
                # profiler="pytorch",  # "simple", "advanced","pytorch"
            )
        elif TrainingConfig.strategy == "deepspeed":
            from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

            print("using DeepSpeed to do experiments; ")

            deepspeed_strategy = DeepSpeedStrategy(
                offload_optimizer=True,
                allgather_bucket_size=5e8,
                reduce_bucket_size=5e8,
            )

            trainer = pl.Trainer(
                devices=num_gpus,
                num_nodes=TrainingConfig.num_nodes,
                precision=TrainingConfig.precision,
                accelerator=TrainingConfig.accelerator,
                strategy=deepspeed_strategy,
                logger=own_logger,
                callbacks=[lr_logger, model_checkpoint],
                log_every_n_steps=1,
                max_epochs=TrainingConfig.max_epochs,
                check_val_every_n_epoch=ValidationConfig.val_interval,
                auto_scale_batch_size="binsearch",
            )
        elif TrainingConfig.strategy == "bagua":
            from pytorch_lightning.strategies import BaguaStrategy

            print("using Bagua to do experiments; Not using lr_find; ")

            bagua_strategy = BaguaStrategy(
                algorithm=TrainingConfig.bagua_sub_strategy
            )  # "gradient_allreduce"; bytegrad"; "decentralized"; "low_precision_decentralized"; qadam"; async";

            trainer = pl.Trainer(
                devices=num_gpus,
                num_nodes=TrainingConfig.num_nodes,
                precision=TrainingConfig.precision,
                accelerator=TrainingConfig.accelerator,
                strategy=bagua_strategy,
                logger=own_logger,
                callbacks=[lr_logger, model_checkpoint],
                log_every_n_steps=1,
                max_epochs=TrainingConfig.max_epochs,
                check_val_every_n_epoch=ValidationConfig.val_interval,
                auto_scale_batch_size="binsearch",
            )
        else:
            print("\n Trainer configuration is wrong. \n")

    return trainer


if __name__ == "__main__":

    (
        lr_logger,
        model_checkpoint,
        early_stop,
        model_class,
        own_logger,
        num_gpus,
    ) = init_expr_config()

    model = model_class(own_logger)
    model = load_pre_weights(model)

    trainer = pl_trainer_config(
        lr_logger, model_checkpoint, early_stop, model_class, own_logger, num_gpus
    )

    if TrainingConfig.training_mode:
        trainer.fit(model)
    else:
        trainer.validate(model, dataloaders=model.val_dataloader())
    # trainer.test(model, dataloaders=val_dataloader)
