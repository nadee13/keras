from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import inception_v3_nadee
from . import keras_modules_injection


@keras_modules_injection
def InceptionV3(*args, **kwargs):
    return inception_v3_nadee.InceptionV3(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return inception_v3_nadee.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return inception_v3_nadee.preprocess_input(*args, **kwargs)
