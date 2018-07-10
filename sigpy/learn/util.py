import numpy as np
import sigpy as sp


def labels_to_scores(labels):
    """Convert labels to scores.

    Args:
        labels (array): One-dimensional label array.

    Returns:
        array: Score array of shape (len(labels), max(labels) + 1).

    """

    device = sp.util.get_device(labels)
    xp = device.xp
    
    with device:
        num_classes = labels.max() + 1
        scores = sp.util.zeros(
            [len(labels), num_classes], dtype=np.float32, device=device)
        scores[xp.arange(len(labels)), labels] = 1

    return scores


def scores_to_labels(scores):
    """Convert scores to labels, by setting peak index to label.

    Args:
        scores (array): Two-dimensional score array.

    Returns:
        array: Label array of lengths scores.shape[0].

    """

    device = sp.util.get_device(scores)
    xp = device.xp
    
    with device:
        return xp.argmax(scores, axis=1)
