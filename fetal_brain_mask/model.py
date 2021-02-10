from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import abc

import numpy as np
import sys, os

bundle_dir = None
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MaskingModel(metaclass=abc.ABCMeta):
    '''abstract class to guarantee children will
    have predict_mask method'''
    @abc.abstractmethod
    def predict_mask(self, image):
        '''abstract method, to be implemented by child classes'''
        pass

class Unet(MaskingModel):
    '''Unet class to manage the loding of model and weights and
    predictive use'''

    def __init__(self):
        '''Class constructor get json model and h5 weigths and load model'''

        if bundle_dir:
            weight_path = os.path.join(bundle_dir, 'models/weights/unet_weights.h5')
            model_path = os.path.join(bundle_dir, 'models/json_models/unet_model.json')
        else:
            weight_path = '/neuro/users/jinwoo.hong/github_work/masking_tool/models/weights/unet_weights.h5'
            model_path = '/neuro/users/jinwoo.hong/github_work/masking_tool/models/json_models/unet_model.json' 

        json_file = open(model_path, 'r')
        json_model = json_file.read()
        json_file.close()

        self.unet_model = model_from_json(json_model)
        self.unet_model.load_weights(weight_path)

    def __getGenerator(self, image, bs=1):
        '''getGenerator Returns generator that will be used for
        prdicting, it takes a single 3D image and returns a generator
        of its slices'''

        # rescale data to its trained mode
        image_datagen = ImageDataGenerator(rescale=1./255)
        image_datagen.fit(image, augment = True)
        image_generator = image_datagen.flow(x = image, batch_size=bs,
                shuffle = False)

        return image_generator

    def predict_mask(self, image):
        '''predict_mask creates a prediction for a whole 3D image'''
        image_gen = self.__getGenerator(image)
        mask = self.unet_model.predict_generator(image_gen, steps=len(image))
        # only keep pixels with more than 0.5% probability of being brain
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        return mask
