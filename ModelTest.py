from SequenceDataProcess import (sequence_design_array, trans_level_array, one_hot_sequence,
                                 new_sequence, trans_test_array, new_sequence_2mer, new_sequence_3mer, new_sequence_4mer)
from SimpleLSTM import max_value
from keras.models import load_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# from ImageSequence import kron_sequence_test
# import h5py

Mer = 3
Model_NAME = 'MLP_class.h5'

loaded_model = load_model(Model_NAME)
if Mer == 1:
    predicted_features_normalized = loaded_model.predict(new_sequence)
    y_predict = predicted_features_normalized * max_value
elif Mer == 2:
    predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
    y_predict = predicted_features_normalized * max_value
elif Mer == 3:
    predicted_features_normalized = loaded_model.predict(new_sequence_3mer)
    y_predict = predicted_features_normalized * max_value
elif Mer == 4:
    predicted_features_normalized = loaded_model.predict(new_sequence_4mer)
    y_predict = predicted_features_normalized * max_value

# print(y_predict.shape)
# print(y_predict)
# print(trans_test_array.shape)
r2 = r2_score(trans_test_array, y_predict)
print('R2 = ', r2)
x = range(0, 15)
y = [i for i in x]
plt.plot(x, y, color='red')
plt.scatter(trans_test_array, y_predict, s=10)
plt.show()


# BiLSTM
# loaded_model = load_model('BiLSTM.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence)
# y_predict = predicted_features_normalized*max_value


# transformer+lstm_1
# loaded_model = load_model('Transformer.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence)
# y_predict = predicted_features_normalized*max_value


# transformer+lstm_2mer
# loaded_model = load_model('Transformer_2mer.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
# y_predict = predicted_features_normalized*max_value

# transformer+lstm_2mer_different_head
# loaded_model = load_model('Transformer_2mer_head=2.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
# y_predict = predicted_features_normalized*max_value

# transformer+lstm_2mer_different_key
# loaded_model = load_model('Transformer_2mer_key=128.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
# y_predict = predicted_features_normalized*max_value

# transformer+lstm_3mer
# loaded_model = load_model('Transformer_3mer.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_3mer)
# y_predict = predicted_features_normalized * max_value

# only onehot
# loaded_model = load_model('SimpleLSTM.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence)
# y_predict = predicted_features_normalized*max_value

# 2-mer+onehot
# loaded_model = load_model('SimpleLSTM_2mer.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
# y_predict = predicted_features_normalized*max_value

# 3-mer+onehot
# loaded_model = load_model('SimpleLSTM_3mer.h5')
# predicted_features_normalized = loaded_model.predict(new_sequence_3mer)
# y_predict = predicted_features_normalized * max_value


# CNN
# model = load_model('SimpleCNN.h5')
# predicted_features_normalized = model.predict(kron_sequence_test)
# y_predict = predicted_features_normalized * max_value


