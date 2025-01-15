from SequenceDataProcess import (sequence_design_array, trans_level_array, one_hot_sequence,
                                 new_sequence, trans_test_array, new_sequence_2mer,
                                 new_sequence_3mer, new_sequence_4mer)
from SimpleLSTM import max_value
from keras.models import load_model
import matplotlib.pyplot as plt
from PredictDataAnaly import center_0, center_1
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

test_trans_level_labels = np.zeros(trans_test_array.shape[0])
test_trans_level_labels[trans_test_array.flatten() < center_0] = 0
test_trans_level_labels[(trans_test_array.flatten() >= center_0) & (trans_test_array.flatten() <= center_1)] = 1
test_trans_level_labels[trans_test_array.flatten() > center_1] = 2

Mer = 3
Model_NAME = 'CNN_BiLSTM_2.h5'
class_names = ['low ', 'med', 'high']
roc_colors = ['red', 'green', 'blue']
colors = ["#666666", "#1eff00", "#0070ff", "#a335ee", "#a268a8", "#ff8000"]
# my_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
my_cmap = mcolors.ListedColormap(colors)

loaded_model = load_model(Model_NAME)
if Mer == 1:
    predicted_features_normalized = loaded_model.predict(new_sequence)
elif Mer == 2:
    predicted_features_normalized = loaded_model.predict(new_sequence_2mer)
elif Mer == 3:
    predicted_features_normalized = loaded_model.predict(new_sequence_3mer)
elif Mer == 4:
    predicted_features_normalized = loaded_model.predict(new_sequence_4mer)


predicted_classes = np.argmax(predicted_features_normalized, axis=1)

print(predicted_classes)
print(test_trans_level_labels)
accuracy = np.sum(predicted_classes == test_trans_level_labels) / len(test_trans_level_labels)
print(f'Accuracy: {accuracy * 100:.4f}%')

# 计算F1分数、召回率和精确度
f1 = f1_score(test_trans_level_labels, predicted_classes, average='weighted')
recall = recall_score(test_trans_level_labels, predicted_classes, average='weighted')
precision = precision_score(test_trans_level_labels, predicted_classes, average='weighted')

print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')

# ROC AUC指标
n_classes = 3
true_labels_bin = label_binarize(test_trans_level_labels, classes=[0, 1, 2])
predicted_probs_bin = np.zeros(true_labels_bin.shape)
# 假设你有一个模型进行预测并得到每个类的概率
predicted_probs = predicted_features_normalized  # 获取每个类别的预测概率
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(true_labels_bin[:, i], predicted_probs[:, i])
    roc_auc = auc(fpr, tpr)  # 计算AUC值
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (class: {class_names[i]}) area = {roc_auc:.4f}',
             color=roc_colors[i])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc='lower right', bbox_to_anchor=(1, 0), borderaxespad=0., fontsize=12)
plt.grid()
plt.show()


cm = confusion_matrix(test_trans_level_labels, predicted_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap="Purples",
            xticklabels=np.unique(test_trans_level_labels),
            yticklabels=np.unique(test_trans_level_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix')
plt.show()

