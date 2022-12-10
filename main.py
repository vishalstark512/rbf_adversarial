# -*- coding: utf-8 -*-
""" main.py """

import numpy as np

from dataloader.dataloader import Dataloader
from configs.config import CFG
from model.unet import UNet
from executor.train import Train


def run():
    """Builds model, loads_data, and evaluates"""
    # Give full paths
    n_size = 8
    input_img_paths = 'D:/mine/AISummer/project2/data/Task04_Hippocampus/imagesTr/'
    label_img_paths = 'D:/mine/AISummer/project2/data/Task04_Hippocampus/labelsTr/'

    model = UNet(CFG)
    final_model = model.build()
    # print(final_model.summary())

    # loading Train Class
    train_model = Train(final_model)

    # # Load data in one go
    data = Dataloader(input_img_paths, label_img_paths)
    X, y = data.data_load()
    print(X.shape, y.shape)

    train_loss, val_loss = train_model.train(X, y)

    # To load the 3D data cubes for 3d UNet
    # X, y = data.data_load_3d()

    # This is a function that will go directly into model training using Datagen
    # Use this when you have low GPU memory
    # train_loss, val_loss = train_model.train_with_gen(n_size, input_img_paths, label_img_paths)


if __name__ == "__main__":
    run()
