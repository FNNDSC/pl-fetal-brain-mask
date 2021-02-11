import cv2
import logging
import numpy as np
from medpy.io import load, save  # TODO replace with nibabel
from skimage.measure import label
from skimage.morphology import binary_closing, cube

from fetal_brain_mask.model import Unet

# TODO
# normalization and reshaping can probably be done with library
# from skimage.transform import resize

logger = logging.getLogger(__name__)


class MaskingTool:
    def __init__(self):
        self.model = Unet()

    def create_mask(self, img_path: str,  output: str, smoothen=True):
        logger.info('Processing ' + img_path)
        img, hdr = self.get_image_data(img_path)

        resize_needed = False
        original_shape = (img.shape[2], img.shape[1])
        if img.shape[1] != 256 or img.shape[2] != 256:
            img = self.resize_data(img)
            resize_needed = True

        res = self.model.predict_mask(img)

        if smoothen:
            # it would be better for this to be put in its own plugin
            res = binary_closing(np.squeeze(res), cube(2))
            try:
                labels = label(res)
                res = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint16)
            except Exception as e:
                logger.error(e)
                logger.error('Failed to apply smoothing for ' + img_path)

        if resize_needed:
            res = self.resize_data(res.astype(np.uint16), target=original_shape)

        # remove extra dimension
        res = np.squeeze(res)
        # return result into shape (256,256, X)
        res = np.moveaxis(res, 0, -1)

        save(res, output, hdr)

    @staticmethod
    def normalize_uint8(img_slice):
        """
        Normalizes the image to be in the range of 0-255
        it round up negative values to 0 and caps the top values at the
        97% value as to avoid outliers
        """
        img_slice[img_slice < 0] = 0
        flat_sorted = np.sort(img_slice.flatten())

        # dont consider values greater than 97% of the values
        top_3_limit = int(len(flat_sorted) * 0.97)
        limit = flat_sorted[top_3_limit]

        img_slice[img_slice > limit] = limit

        rows, cols = img_slice.shape
        # create new empty image
        new_img = np.zeros((rows, cols))
        max_val = np.max(img_slice)
        if max_val == 0:
            return new_img

        # normalize all values
        for i in range(rows):
            for j in range(cols):
                new_img[i, j] = int((float(img_slice[i, j]) / float(max_val)) * 255)

        return new_img

    @staticmethod
    def resize_data(image, target=(256, 256)):
        image = np.squeeze(image)
        resized_img = []
        for i in range(image.shape[0]):
            img_slice = cv2.resize(image[i, :, :], target)
            resized_img.append(img_slice)

        image = np.array(resized_img, dtype=np.uint16)
        return image[..., np.newaxis]

    @classmethod
    def get_image_data(cls, fname: str):
        """
        Returns the image data, image matrix and header of a particular file
        """
        data, hdr = load(fname)
        # axes have to be switched from (256,256,x) to (x,256,256)
        data = np.moveaxis(data, -1, 0)

        norm_data = []
        # normalize each image slice
        for i in range(data.shape[0]):
            img_slice = data[i, :, :]
            norm_data.append(cls.normalize_uint8(img_slice))

        # remake 3D representation of the image
        data = np.array(norm_data, dtype=np.uint16)

        data = data[..., np.newaxis]
        return data, hdr
