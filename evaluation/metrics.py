import torch


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        (2 * intersection + smooth)
        /
        (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + smooth
        )
    )

    return dice.mean().item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        - intersection
    )

    iou = (
        (intersection + smooth)
        /
        (union + smooth)
    )

    return iou.mean().item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_positive = (
        predictions * (1 - targets)
    ).sum(dim=1)

    precision = (
        (true_positive + smooth)
        /
        (true_positive + false_positive + smooth)
    )

    return precision.mean().item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0), -1
    )

    targets = targets.view(
        targets.size(0), -1
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_negative = (
        (1 - predictions) * targets
    ).sum(dim=1)

    recall = (
        (true_positive + smooth)
        /
        (true_positive + false_negative + smooth)
    )

    return recall.mean().item()