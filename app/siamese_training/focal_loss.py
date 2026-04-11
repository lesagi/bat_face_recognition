"""
Binary Focal Loss for Siamese network training.

Focal loss addresses class imbalance by down-weighting easy (well-classified)
examples and focusing training on hard (misclassified) examples.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    https://arxiv.org/abs/1708.02002
"""

import tensorflow as tf


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary focal loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        p_t = p if y=1, else (1-p)
        alpha_t = alpha if y=1, else (1-alpha)

    With gamma=0 and alpha=0.5 this reduces to standard binary cross-entropy.

    Args:
        alpha: Weight for the positive class. Values > 0.5 upweight positives
            (anchors/same-bat pairs). Typical range: 0.25-0.75.
        gamma: Focusing parameter. Higher values down-weight easy examples more
            aggressively. gamma=0 disables focusing (standard BCE). Typical: 2.0.
        from_logits: Whether predictions are raw logits or probabilities.
        reduction: Loss reduction mode. Use 'none' for per-sample weighting
            compatibility with the existing ClassWeightCalculator pipeline.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction=tf.keras.losses.Reduction.NONE,
        name: str = "binary_focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape.rank > 1 else y_pred

        if self.from_logits:
            p = tf.sigmoid(y_pred)
        else:
            p = y_pred

        epsilon = tf.keras.backend.epsilon()
        p = tf.clip_by_value(p, epsilon, 1.0 - epsilon)

        # p_t = probability of the true class
        p_t = tf.where(y_true >= 0.5, p, 1.0 - p)

        # alpha_t = per-sample alpha based on true label
        alpha_t = tf.where(y_true >= 0.5, self.alpha, 1.0 - self.alpha)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Standard cross-entropy term: -log(p_t)
        ce = -tf.math.log(p_t)

        loss = alpha_t * focal_weight * ce
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
        })
        return config
