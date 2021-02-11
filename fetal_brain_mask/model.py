from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import logging

try:
    from importlib.resources import files
except ModuleNotFoundError:
    from importlib_resources import files

logger = logging.getLogger(__name__)


class Unet:
    """
    Unet class to manage the loading of model and weights and predictive use.
    """
    def __init__(self):
        logger.debug('Using unet')

        logger.debug('Loading model')
        json_model = files(f'{__package__}.data').joinpath('unet_model.json').read_text()
        self.unet_model = model_from_json(json_model)

        logger.debug('Loading weights')
        weight_path = files(f'{__package__}.data').joinpath('unet_weights.h5')
        self.unet_model.load_weights(str(weight_path))

        self.graph = tf.compat.v1.get_default_graph()
        logger.debug('Model initialized')

    @staticmethod
    def get_generator(image, bs=1):
        """
        getGenerator Returns generator that will be used for
        predicting, it takes a single 3D image and returns a generator
        of its slices
        """

        # rescale data to its trained mode
        image_datagen = ImageDataGenerator(rescale=1.0/255)
        image_datagen.fit(image, augment=True)
        image_generator = image_datagen.flow(x=image, batch_size=bs, shuffle=False)
        return image_generator

    def predict_mask(self, image, threshold=0.5):
        """
        predict_mask creates a prediction for a whole 3D image
        """
        image_gen = self.get_generator(image)

        # https://github.com/tensorflow/tensorflow/issues/14356#issuecomment-447588953
        with self.graph.as_default():
            mask = self.unet_model.predict_generator(image_gen, steps=len(image))

        # only keep pixels with more than 0.5% probability of being brain
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0

        return mask
