class DataConfig:
    data_root = "/data/SSD1/data/tiny-imagenet-200/"
    logger_root = "./work_dirs/main_csv_logs/"
    work_dirs = "./work_dirs/"
    fix_train_data = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image_size = 256
    image_crop_size = 224

    workers = 16
    pin_memory = True
    random_seed = 42

    num_classes = 200
    class_ignore = False

    cls_names = None
    classes = None
    class_idx = None

    dataset_name = "tinyimagenet200"
    dataloader_name = "dataloader"
    palette = None
    extra_args = {}


class ExtraDatasetConfig:
    extra_args = {}


class NetConfig:

    """>>> Must Changed for Different Expr"""

    backbone_name = "resnet50"
    model_real_name = "resnet50"
    lr = 0.08
    backbone_lr = 0.08
    opt = "SGD"  # AdamW, Adam, SGD

    model_name = "model"
    # WEIGHT_DECAY = 0.0005
    # BETA = 0.5
    # MOMENTUM = 0.9
    # EPS = 0.00000001 # 1e-8
    # AMSGRAD = False0

    extra_args = {}


class TrainingConfig:

    training_mode = True
    # training_mode = False

    batch_size = 16 * 3
    precision = 16

    pretrained_weights = False
    pre_backbone_lr = 0.01
    pre_lr = 0.01
    pretrained_weights_max_epoch = 40
    pretrained_weights_path = "XXXX.ckpt"

    scheduler = "cosineAnn"  # "step", "cosineAnnWarm", "poly", "cosineAnn", "cosineAnnWarmDecay"
    eta_min = 0.0  # for cosineAnn
    T_0 = 5  # for cosineAnnWarm; cosineAnnWarmDecay
    T_mult = 3  # for cosineAnnWarm; cosineAnnWarmDecay
    decay = 0.5  # for cosineAnnWarmDecay
    T_max = T_0 + T_mult * T_0

    step_ratio = 0.3  # for StepLR
    gamma = 0.1  # for StepLR
    poly_lr: False

    max_epochs = T_max

    subtrain = False
    subtrain_ratio = 1

    single_lr = False
    use_ema = False

    interpolation: False
    logger_name = "wandb"  # "neptune", "csv", "wandb"
    cpus = False
    num_gpus = "autocount"
    num_nodes = 1
    wandb_name = (
        "pt-"
        + NetConfig.model_real_name
        + "-"
        + DataConfig.dataset_name
        + "-"
        + NetConfig.backbone_name
    )
    ckpt_path = "none"
    onnx_model = "./work_dirs/default.onnx"
    resume = "none"
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1

    use_torchhub = False
    use_timm = False

    lr_find = False

    extra_args = {}


class ValidationConfig:
    batch_size = 16  # 16;

    # if DataConfig.num_classes > 500:
    #     val_interval = TrainingConfig.max_epochs - 5
    # elif DataConfig.num_classes > 100:
    #     val_interval = 20
    # else:
    #     val_interval = 1

    val_interval = 1

    sub_val = False
    subval_ratio = 1
    extra_args = {}


class TestingConfig:
    batch_size = 1
    ckpt_path = "none"
    multiscale = False
    imageration = [1.0]
    slidingscale = False
    extra_args = {}


class ResNetConfig:
    placeholder = 0

    # model = dict(
    #     type="ImageClassifier",
    #     backbone=dict(
    #         type="ResNet", depth=50, num_stages=4, out_indices=(3,), style="pytorch"
    #     ),
    #     neck=dict(type="GlobalAveragePooling"),
    #     head=dict(
    #         type="LinearClsHead",
    #         num_classes=DataConfig.num_classes,
    #         in_channels=2048,
    #         loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
    #         topk=(1, 5),
    #     ),
    # )

    # model settings
    model = dict(
        type="ImageClassifier",
        backbone=dict(
            type="ResNet_CIFAR",
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style="pytorch",
        ),
        neck=dict(type="GlobalAveragePooling"),
        head=dict(
            type="LinearClsHead",
            num_classes=DataConfig.num_classes,
            in_channels=2048,
            loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        ),
    )
