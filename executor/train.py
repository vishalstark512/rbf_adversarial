# -*- coding: utf-8 -*-
"""train class"""
import numpy as np
import tensorflow as tf
from utils.losses import CustomLosses
import tensorflow.keras.backend as K
from dataloader.dataloader import Chunk_data_gen


class Train:
    """Train class
    This contains a custom_loss function
    amd a function to train the model
    """
    def __init__(self, model):
        self.model = model
        self.split = 0.2
        self.epochs = 20
        self.batch_size = 256
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='logs', monitor="val_loss",
                                                             verbose=1, save_best_only=True, mode="auto")

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

    def custom_loss(self, y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        loss1 = bce(y_true, y_pred)
        loss2 = CustomLosses.dice_loss(self, y_true, y_pred)
        loss3 = CustomLosses.sigmoid_focal_crossentropy(self, y_true, y_pred)

        loss = loss1 + loss2 + loss3

        return loss

    def train(self, x, y):
        """Compiles and trains the model"""
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=self.custom_loss,
                           metrics=[self.dice_coef])

        print("Model Compiled")
        # LOG.info('Training started')

        x = tf.cast(np.asarray(x).astype(np.float32), dtype=tf.float32)
        x = tf.expand_dims(x, axis=-1)
        y = tf.cast(np.asarray(y).astype(np.float32), dtype=tf.float32)
        y = tf.expand_dims(y, axis=-1)

        model_history = self.model.fit(x, y, epochs=self.epochs,
                                       batch_size=self.batch_size,
                                       verbose=1,
                                       validation_split=self.split,
                                       callbacks=[self.checkpoint])

        return model_history.history['loss'], model_history.history['val_loss']

    def train_with_gen(self, n_size, input_img_paths, label_img_paths):
        """Compiles and trains the model with datagen"""
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=self.custom_loss,
                           metrics=[self.dice_coef])

        # LOG.info('Training started')

        train_data_gen, val_data_gen = Chunk_data_gen(n_size, input_img_paths, label_img_paths)

        model_history = self.model.fit(train_data_gen, epochs=self.epochs,
                                       verbose=1,
                                       callbacks=[self.checkpoint],
                                       validation_data=val_data_gen)

        return model_history.history['loss'], model_history.history['val_loss']