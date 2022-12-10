import numpy as np


class Pre_process:
    """Data Loader class"""

    def pad_images(self, img, target_width: int = 64, target_height: int = 64):
        """Padding the sizes fo the images
        Args:
            img (np.array): input image to be padded
            target_width (int): width of the frame
            target_height (int): height of the frame
        Return:
            padded_img (np.array): Padded image
        """
        # print(img.shape)
        padded_image = np.pad(img, [(0, target_width - img.shape[0]),
                                    (0, target_height - img.shape[1])], mode='constant')

        return padded_image

    def normalize_images(self, img, normalization_type='divide_255'):
        """
        Args:
            img: numpy 4D array
            normalization_type: `str`, available choices:
                - divide_255
                - divide_256
                - by_chanels
        """
        if normalization_type == 'divide_255':
            images = img / 255
        elif normalization_type == 'divide_256':
            images = img / 256
        elif normalization_type is None:
            pass
        else:
            raise Exception("Unknown type of normalization")
        return images


