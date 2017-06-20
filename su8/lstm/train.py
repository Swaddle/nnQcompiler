
import os
from data_utils import load_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.recurrent import GRU
from keras.callbacks import CSVLogger
from numpy import array
from tensorflow.python.ops import control_flow_ops
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# tf.python.control_flow_ops = tf
training_data_path = 'data_training.csv'
valid_data_path = 'data_valid.csv'


# custom objective - unscaled Euclidean norm
def custom_objective(y_true, y_pred):
    tensor = y_true - y_pred 
    squares = tf.square(tensor)
    norm = tf.reduce_sum(squares)
    return norm


def main():

    model = Sequential()

    # input Re(U) represented as a vector, SU(8) length 64 - split into (8,8)
    # output Re(U_i), so that U is factored into products = U_n U_{n-1}...U_2 U_1 
    # this example n = 10, so (80, 8)

    model.add(GRU(80, return_sequences=True, input_shape=(8,8)))
    model.add(Dropout(0.2))
    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(80, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(input_dim=640,output_dim=640))
  
    train_input, train_output = load_data(training_data_path)
    valid_input, valid_output = load_data(valid_data_path)	 
  
    checkpoint = ModelCheckpoint('quant_model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    # estop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    csv_logger = CSVLogger('training.log',separator=',', append=False)

    model.compile(optimizer='Nadam',loss=custom_objective)
    model.summary()

    model.fit(train_input, train_output, validation_data=(valid_input,valid_output), epochs=5000, batch_size=64, callbacks=[checkpoint, csv_logger])
    model.save('quant_model.h5')


if __name__ == '__main__':
    main()
