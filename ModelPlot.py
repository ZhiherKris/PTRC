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
from keras.layers import RepeatVector, Dense, Permute, Lambda, Flatten, multiply, Conv1D, MaxPooling1D
from keras import backend as K
import tensorflow as tf
from keras.optimizers import Adam

from collections import defaultdict
import visualkeras
from PIL import ImageFont


trans_level_array = np.array(trans_level_array, dtype=np.float32)
max_value = np.max(trans_level_array)
trans_level_array = trans_level_array / max_value


input_layer = Input(shape=(three_mer_onehot_sequence.shape[1], three_mer_onehot_sequence.shape[2]))
conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
bilstm_layer = Bidirectional(LSTM(124, return_sequences=False))(pooling_layer)
# attention_layer = Attention()([bilstm_layer, bilstm_layer])
flatten_layer = Flatten()(bilstm_layer)
dense_layer = Dense(124, activation='relu')(flatten_layer)
dense_layer2 = Dense(248, activation='relu')(dense_layer)
output_layer = Dense(3, activation='softmax')(dense_layer2)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

layer_text_mapping = {
    0: "Input Layer     ",
    1: "Conv Layer",
    2: "Pooling Layer",
    3: "BiLSTM Layer",
    4: "Flatten Layer",
    5: "Dense Layer",
    6: "Dense Layer",
    7: "Output Layer"
}

font = ImageFont.truetype(font="c:\\WINDOWS\\FONTS\\TIMESBD.TTF", size=40)


visualkeras.layered_view(
    model,
    to_file='model_visualization.png',
    min_z=20,
    min_xy=20,
    draw_volume=True,
    padding=100,
    text_callable=lambda index, layer: (layer_text_mapping.get(index, layer.name), False),
    spacing=50,
    font_color='black',
    font=font,

).show()

