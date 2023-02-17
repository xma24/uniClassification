import torchmetrics

from configs.config import ResNetConfig, DataConfig

from models.models_builder import build_classifier

""">>> TorchMetrics: https://torchmetrics.readthedocs.io/en/latest/; """


def get_classification_metrics(num_classes, ignore):
    if ignore:
        pass
    else:
        train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        train_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="none"
        )

        test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="none"
        )

    return (
        train_accuracy,
        train_precision,
        train_recall,
        val_accuracy,
        val_precision,
        val_recall,
        test_accuracy,
        test_precision,
        test_recall,
    )


def get_resnet_model():

    model = build_classifier(ResNetConfig.model)

    # from resnet_without_mmlab import build_resnet, Bottleneck

    # model = build_resnet(
    #     block=Bottleneck,
    #     layers=[3, 4, 6, 3],
    #     weights=None,
    #     progress=True,
    #     num_classes=DataConfig.num_classes,
    #     zero_init_residual=False,
    #     groups=1,
    #     width_per_group=64,
    #     replace_stride_with_dilation=[False, False, False],
    #     norm_layer=None,
    # )

    return model


def get_resnet_backbone(model):

    backbone = model.backbone

    return backbone


def get_resnet_neck(model):

    neck = model.neck

    return neck


def get_resnet_head(model):

    head = model.head

    return head


if __name__ == "__main__":
    model = get_resnet_model()
    print("==>> model: ", model)
