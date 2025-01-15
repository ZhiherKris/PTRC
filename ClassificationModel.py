from SequenceDataProcess import (trans_level_array,
                                 one_hot_sequence,
                                 two_mer_onehot_sequence,
                                 three_mer_onehot_sequence,
                                 four_mer_onehot_sequence)
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Add, Reshape, Bidirectional
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.layers import Activation, dot, concatenate
from keras.layers import RepeatVector, Dense, Permute, Lambda, Flatten, multiply
from keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Flatten, Dense, Attention
import tensorflow as tf
from PredictDataAnaly import center_0, center_1
from tensorflow import keras
from keras.optimizers import Adam
import datetime


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

trans_level_labels = np.zeros(trans_level_array.shape[0])
trans_level_labels[trans_level_array.flatten() < center_0] = 0
trans_level_labels[(trans_level_array.flatten() >= center_0) & (trans_level_array.flatten() <= center_1)] = 1
trans_level_labels[trans_level_array.flatten() > center_1] = 2
labels_onehot = keras.utils.to_categorical(trans_level_labels, 3)


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

train_dataset = tf.data.Dataset.from_tensor_slices((three_mer_onehot_sequence, labels_onehot)).batch(1)

if tf.test.is_gpu_available():
    with tf.device('/GPU:0'):

        input_layer = Input(shape=(three_mer_onehot_sequence.shape[1], three_mer_onehot_sequence.shape[2]))
        conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
        conv_layer2 = Conv1D(filters=32, kernel_size=3, activation='relu')(pooling_layer)
        pooling_layer2 = MaxPooling1D(pool_size=2)(conv_layer2)
        bilstm_layer = Bidirectional(LSTM(units=124, return_sequences=True))(pooling_layer2)
        attention_layer = Attention()([bilstm_layer, bilstm_layer])
        flatten_layer = Flatten()(attention_layer)
        dense_layer = Dense(units=256, activation='relu')(flatten_layer)
        dense_layer2 = Dense(units=128, activation='relu')(dense_layer)
        dense_layer3 = Dense(units=64, activation='relu')(dense_layer2)
        output_layer = Dense(units=3, activation='softmax')(dense_layer3)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model.summary()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(train_dataset, epochs=100, batch_size=1, callbacks=[tensorboard_callback])
        model.save('CNN_BiLSTM_2.h5')

else:
    print("no gpu")

history.history.keys()
print(history.history.keys())
loss = history.history['loss']
Accuracy = history.history['accuracy']
plt.figure()
plt.subplot(121)
plt.plot(loss)
plt.title('LOSS')
plt.subplot(122)
plt.plot(Accuracy)
plt.title('Accuracy')
plt.show()
