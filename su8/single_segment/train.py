
import os
from data_utils import load_data
from keras.layers.core import Dense, Activation, Dropout, Flatten, Merge
from keras.models import Sequential
from numpy import array
from keras.callbacks import CSVLogger
from tensorflow.python.ops import control_flow_ops
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras import backend as K


training_data_path = 'data_training.csv'
valid_data_path = 'data_valid.csv'

def custom_objective(y_true, y_pred):
    tensor = y_true - y_pred 
    squares = tf.square(tensor)
    norm = tf.reduce_sum(squares)
    return norm



def main():

    model = Sequential()

    # input dim, Re(U) represented as a vector , SU(8) length 64
    # output dim 36, 
    # single step

    model.add(Dense(output_dim=2000,input_dim=64))
    
    model.add(Dense(4000,activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(4000,activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(input_dim=4000,output_dim=36))

    train_input, train_output = load_data(training_data_path)
    valid_input, valid_output = load_data(valid_data_path)	

    checkpoint = ModelCheckpoint('quant_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    csv_logger = CSVLogger('training.log',separator=',', append=False)

    model.compile(optimizer='adam',loss=custom_objective)
    model.summary()

    model.fit(train_input, train_output, validation_data=(valid_input,valid_output), nb_epoch=500, batch_size=64, callbacks=[checkpoint,csv_logger])
    model.save('quant_model.h5')
    
if __name__ == '__main__':
    main()
