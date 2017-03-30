


import os
import numpy
from data_utils import load_data
from numpy import array
from keras.models import load_model

import tensorflow as tf
tf.python.control_flow_ops = tf

valid_data_path = 'data_valid.csv'


def custom_objective(y_true, y_pred):
    tensor = y_true - y_pred
    squares = tf.square(tensor)
    norm = tf.reduce_sum(squares)
    return norm

def predict():
    model = load_model('quant_model.h5',custom_objects={'custom_objective':custom_objective})
    valid_input, valid_output = load_data(valid_data_path)
    prediction =  model.predict(valid_input)
    numpy.savetxt("prediction_outputs.csv", prediction, delimiter=",")

if __name__ == '__main__':
    predict()
