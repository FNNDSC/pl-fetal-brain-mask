import cv2
import logging
import numpy as np
import nibabel as nib
from skimage.measure import label
from skimage.morphology import binary_closing, cube

from fetal_brain_mask.model import Unet

logger = logging.getLogger(__name__)


class MaskingTool:
    def __init__(self):
        self.model = Unet()

    def create_mask(self, input_filename: str,  output_filename: str, smoothen=True):
        logger.info('Processing ' + input_filename)

        image = nib.load(input_filename)
        data = image.get_fdata(caching='unchanged')
        # axes have to be switched from (256,256,x) to (x,256,256)
        data = np.moveaxis(data, -1, 0)
        # normalize each image slice
        data = np.array([self.normalize_uint8(islice) for islice in data], dtype=np.uint16)
        data = data[..., np.newaxis]

        resize_needed = False
        original_shape = (data.shape[2], data.shape[1])
        if data.shape[1] != 256 or data.shape[2] != 256:
            data = self.resize_data(data)
            resize_needed = True

        # do prediction
        data = self.model.predict_mask(data)

        if smoothen:
            # it would be better for this to be put in its own plugin
            data = binary_closing(np.squeeze(data), cube(2))
            try:
                labels = label(data)
                data = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1).astype(np.uint16)
            except Exception as e:
                logger.error(e)
                logger.error('Failed to apply smoothing for ' + input_filename)

        if resize_needed:
            data = self.resize_data(data.astype(np.uint16), target=original_shape)

        # remove extra dimension
        data = np.squeeze(data)
        # return result into shape (256,256, X)
        data = np.moveaxis(data, 0, -1)

        # create Nifti object with same header, but new data
        image = image.__class__(data, image.affine, header=image.header)
        nib.save(image, output_filename)

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
        # maybe we should use a statistical method here instead?
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
        # maybe use a library for this?
        image = np.squeeze(image)
        resized_img = []
        for i in range(image.shape[0]):
            img_slice = cv2.resize(image[i, :, :], target)
            resized_img.append(img_slice)

        image = np.array(resized_img, dtype=np.uint16)
        return image[..., np.newaxis]
