import os
import argparse

import tensorflow as tf

from tensorflow import keras
import numpy as np
from keras.models import load_model
from keras.datasets import mnist

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

tf.reset_default_graph()

parser = argparse.ArgumentParser()
parser.add_argument('-m','--modelfile', required=True)

args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

sess = tf.Session()
keras.backend.set_session(sess)

model_keras = load_model(args.modelfile)

score = model_keras.evaluate(x_test, y_test, verbose=0)
print('model test loss:', score[0])
print('model test accuracy:', score[1])

model_clever = KerasModelWrapper(model_keras)
fgsm = FastGradientMethod(model_clever)

gen_graph = fgsm.generate(x_train[0], eps=0.3, clip_min=0., clip_max=1.)
print(gen_graph)
