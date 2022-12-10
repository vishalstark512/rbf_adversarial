# -*- coding: utf-8 -*-
"""Unet model"""

# internal
from .base_model import BaseModel
# external
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Dropout, Input, \
    MaxPool2D, Conv2DTranspose, concatenate
from tensorflow_examples.models.pix2pix import pix2pix


class CNNBlocks:
    def __init__(self, kernel_size):
        self.activation = "relu"
        self.reg = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)
        self.kernel = kernel_size
        self.dropout = 0.1

    def conv_down(self, n_conv, inputs):
        cd = Conv2D(n_conv, self.kernel, activation=self.activation,
                    kernel_regularizer=self.reg, padding='same')(inputs)
        cd = Dropout(self.dropout)(cd)
        cd = Conv2D(n_conv, self.kernel, activation=self.activation,
                    kernel_regularizer=self.reg, padding='same')(cd)

        return cd

    def concat(self, n_conv, inputs, skip):
        con = Conv2DTranspose(n_conv, (2, 2), strides=(2, 2), padding='same')(inputs)
        con = concatenate([con, skip])

        return con

class UNet(BaseModel):
    """Unet Model class"""
    def __init__(self, config):
        super().__init__(config)

        self.model = None
        self.output_channels = self.config.model.output

    def build(self):
        """Build the keras model based"""

        inputs = tf.keras.layers.Input(shape=self.config.model.input)

        conv_block = CNNBlocks(kernel_size=3)

        # Down block
        d1 = conv_block.conv_down(self.config.model.filter.l1, inputs)
        p1 = MaxPool2D((2, 2))(d1)
        d2 = conv_block.conv_down(self.config.model.filter.l2, p1)
        p2 = MaxPool2D((2, 2))(d2)
        d3 = conv_block.conv_down(self.config.model.filter.l3, p2)
        p3 = MaxPool2D((2, 2))(d3)
        d4 = conv_block.conv_down(self.config.model.filter.l4, p3)
        p4 = MaxPool2D((2, 2))(d4)
        d5 = conv_block.conv_down(self.config.model.filter.l5, p4)

        # Up block
        u6 = conv_block.concat(self.config.model.filter.l4, d5, d4)
        u6 = conv_block.conv_down(self.config.model.filter.l4, u6)
        u7 = conv_block.concat(self.config.model.filter.l3, u6, d3)
        u7 = conv_block.conv_down(self.config.model.filter.l3, u7)
        u8 = conv_block.concat(self.config.model.filter.l2, u7, d2)
        u8 = conv_block.conv_down(self.config.model.filter.l2, u8)
        u9 = conv_block.concat(self.config.model.filter.l1, u8, d1)
        u9 = conv_block.conv_down(self.config.model.filter.l1, u9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        return self.model


