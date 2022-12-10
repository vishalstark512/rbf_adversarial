import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


class CustomLosses:
    """Class to write custom loss functions
        This will include loss functions like dice coefficient, BCE
        Focal loss.
        Add new loss functions here if you come across them
    """

    def dice_coef(self, y_true, y_pred, smooth=1):
        """Implements the focal loss function.
            Args:
                y_true: true targets tensor.
                y_pred: predictions tensor.
                smooth: smoothness factor.
            Returns:
                Dice coef (float): float value between 0 and 1
            """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return dice

    def dice_loss(self, y_true, y_pred, smooth=1):
        """Based on Dice coeff"""

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        return 1 - dice

    @tf.function
    def sigmoid_focal_crossentropy(self, y_true, y_pred, alpha = 0.25,
                                   gamma= 2.0, from_logits = True,) -> tf.Tensor:
        """Implements the focal loss function.
        Args:
            y_true: true targets tensor.
            y_pred: predictions tensor.
            alpha: balancing factor.
            gamma: modulating factor.
        Returns:
            Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
            same shape as `y_true`; otherwise, it is scalar.
        """
        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Get the cross_entropy for each entry
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

        # If logits are provided then convert the predictions into probabilities
        if from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred

        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = 1.0
        modulating_factor = 1.0

        if alpha:
            alpha = tf.cast(alpha, dtype=y_true.dtype)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        if gamma:
            gamma = tf.cast(gamma, dtype=y_true.dtype)
            modulating_factor = tf.pow((1.0 - p_t), gamma)

        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)