# -*- coding: utf-8 -*-
"""Data Loader"""
import glob
import random
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from .preprocess import Pre_process


class Dataloader:
    """Data Loader class"""

    def __init__(self, input_img_paths, label_img_paths):
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths
        self.cuboid_depth = 64
        self.target_width = 64
        self.target_height = 64

    def load_preprocessed_data(self, image_path, label=False):
        """Loads the data and pre-processes it
        Args:
            image_path (str): image directory path
            label (bool): extra args for labels
        Return:
            returns the preprocessed data (n_samples x D) x W x H
        """

        data_array = []
        full_path = image_path + "*.*"
        for file_name in glob.iglob(full_path):
            img = sitk.ReadImage(file_name)
            img = sitk.GetArrayFromImage(img)
            # these are 3D images need each frame
            for frame in img:
                # print(frame.shape)
                temp = Pre_process.pad_images(self, img=frame, target_width=self.target_width,
                                              target_height=self.target_height)
                if label:
                    temp = temp.astype(int)
                else:
                    temp = Pre_process.normalize_images(self, img=temp)
                data_array.append(temp)

        # Shape: n_samples (list) can't be converted to array because every sample has different channels
        return np.array(data_array, dtype='object')

    def load_preprocessed_3d_data(self, image_path, label=False):
        """Loads the data and pre-processes it
        Args:
            image_path (str): image directory path
            label (bool): extra args for labels
        Return:
            returns the preprocessed data n_samples x D x W X H
        """

        data_array = []
        full_path = image_path + "*.*"
        for file_name in glob.iglob(full_path):
            img = sitk.ReadImage(file_name)
            img = sitk.GetArrayFromImage(img)

            img_array = []
            for frame in img:
                # print(frame.shape)
                temp = Pre_process.pad_images(self, img=frame, target_width=self.target_width,
                                              target_height=self.target_height)
                if label:
                    temp = temp.astype(int)
                else:
                    temp = Pre_process.normalize_images(self, img=temp)
                img_array.append(temp)
            img_array = np.array(img_array, dtype='object')

            n_frames_to_add = self.cuboid_depth - img.shape[0]
            empty_image = np.zeros((n_frames_to_add, self.target_width, self.target_height))
            # Stacking empty frame sto make cube
            temp = np.vstack((empty_image, img_array))
            data_array.append(temp)

        data_array = np.array(data_array, dtype='object')
        # Shape: n_samples (list) can't be converted to array because every sample has different channels
        return data_array

    def data_load(self):
        """Takes the path of the directory and loads the data
        Args:
        Return:
            return the arrays of input image and label ( (n_samples x D) x W X H)
        """

        processed_images = self.load_preprocessed_data(self.input_img_paths, label=False)
        processed_labels = self.load_preprocessed_data(self.label_img_paths, label=True)

        return processed_images, processed_labels

    def data_load_3d(self):
        """Return 4d cubes of data
        Args:
        Returns:
            4d Arrays n_samples x DxWxH
        """
        processed_images = self.load_preprocessed_3d_data(self.input_img_paths, label=False)
        processed_labels = self.load_preprocessed_3d_data(self.label_img_paths, label=True)

        return processed_images, processed_labels



class Chunk_data_loader(tf.keras.utils.Sequence):
    """ Loads data in chunks and useful when the process are getting out of memory
    Args:
        batch_size (int): number of samples in one batch
        input_img_paths (str): image paths
        label_img_paths (str): label paths
    Returns:
        image_batch, label_batch
    """

    def __init__(self, batch_size, input_img_paths, label_img_paths):
        self.batch_size = batch_size
        self.input_img_paths = input_img_paths
        self.label_img_paths = label_img_paths
        self.val_samples = 20

    def __len__(self):
        return len(self.label_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.label_img_paths[i: i + self.batch_size]

        # Loading images
        img_batch_array = []
        for file_name in batch_input_img_paths:
            img = sitk.ReadImage(file_name)
            img = sitk.GetArrayFromImage(img)

            for frame in img:
                # Padding the image and normalizing it
                temp = Pre_process.pad_images(self, img=frame, target_width=64, target_height=64)
                temp = Pre_process.normalize_images(self, img=temp)
                img_batch_array.append(temp)
        img_batch_array = np.array(img_batch_array)

        # Loading label
        label_batch_array = []
        for file_name in batch_target_img_paths:
            img = sitk.ReadImage(file_name)
            img = sitk.GetArrayFromImage(img)

            for frame in img:
                # Padding the image and converting it into int
                temp = Pre_process.pad_images(self, img=frame, target_width=64, target_height=64)
                temp = temp.astype(int)
                label_batch_array.append(temp)
        label_batch_array = np.array(label_batch_array, dtype='object')

        img_batch_array = tf.cast(np.asarray(img_batch_array).astype(np.float32), dtype=tf.float32)
        img_batch_array = tf.expand_dims(img_batch_array, axis=-1)
        label_batch_array = tf.cast(np.asarray(label_batch_array).astype(np.float32), dtype=tf.float32)
        label_batch_array = tf.expand_dims(label_batch_array, axis=-1)

        return img_batch_array, label_batch_array


def Chunk_data_gen(batch_size, input_img_paths, label_img_paths):
    """This calls our custom Data generator"""
    input_paths = []
    val_samples = 40
    full_path = input_img_paths + "*.*"
    for file_name in glob.iglob(full_path):
        input_paths.append(file_name)
    input_paths = np.array(input_paths)

    label_paths = []
    full_path = label_img_paths + "*.*"
    for file_name in glob.iglob(full_path):
        label_paths.append(file_name)
    label_paths = np.array(label_paths)

    random.Random(1337).shuffle(input_paths)
    random.Random(1337).shuffle(label_paths)
    train_input_img_paths = input_paths[:-val_samples]
    train_input_label_paths = label_paths[:-val_samples]

    val_input_img_paths = input_paths[-val_samples:]
    val_input_label_paths = label_paths[-val_samples:]

    train_data_gen = Chunk_data_loader(batch_size, train_input_img_paths, train_input_label_paths)
    val_data_gen = Chunk_data_loader(batch_size, val_input_img_paths, val_input_label_paths)

    return train_data_gen, val_data_gen

###########################

    # @staticmethod
    # def data_load(data_config):
    #     """Loads dataset from path"""
    #     return tfds.load(data_config.path, with_info=data_config.load_with_info)
