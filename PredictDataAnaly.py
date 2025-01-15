import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from itertools import chain
import matplotlib.font_manager as fm
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
data_path = 'D:\\BioData\\PTRC实验数据.xlsx'
df = pd.read_excel(data_path, sheet_name=5)
test_shape = 0
train_shape = 576 - test_shape
seq_map = {
    'SH5': 'CGTATTTACGCCTTCAGGTAGGCGGA',
    'SH8': 'CGTATTACCGTTTTGACCTACTTCAAAACGCC',
    'SH10': 'CGTATTTAACCAGTAGATCATCCAATCTACTGGTGC',
    'SH12': 'CGTATTTTGCTGTAGGTGTTGTGGTAAACACCTACAGCCT',
    'SH15': 'CGTATTATCGATCCGAAGTGCTCATCCCCGAGCACTTCGGATCGCG',
    'SH18': 'CGTATTTCCCAACACGGAATACCTACATCCCGGTAGGTATTCCGTGTTGGCT',
    'L4': 'CGTATTTCGTCTAAGTTAAAGCTAACTTAGACTC',
    'L8': 'CGTATTAAGAAGTCTATCGAGCGTGGGATAGACTTCCA',
    'L10': 'CGTATTCTAGTAGATCGCCTTCCCCCAAGCGATCTACTCT',
    'L12': 'CGTATTCGCCCGGCTGTCGGGGGCCGAAAAGACAGCCGGGAA',
    'L14': 'CGTATTCAGGACCTGCCGGGGATCGAAGTGGGCGGCAGGTCCAA',
    'EXT': 'CGTATAACTAACTAAAGTTTTAGAGCTAGACACACACACA',
    '4A': 'AAACCCAGGACACACACACAATG',
    '5A': 'AAACCCAGGAGCACACACACATG',
    '6A': 'AAACCAAGGAGCACACACACATG',
    '7A': 'AAACCAAGGAGGCACACACAATG',
    '8A': 'AAACTAAGGAGGCACACACAATG',
    '9A': 'AAACTAAGGAGGTCACACACATG',
    '4N': 'AAACCCAGGACACACACACAA',
    '5N': 'AAACCCAGGAGCACACACACA',
    '6N': 'AAACCAAGGAGCACACACACA',
    '7N': 'AAACCAAGGAGGCACACACAA',
    '8N': 'AAACTAAGGAGGCACACACAA',
    '9N': 'AAACTAAGGAGGTCACACACA',
    'N21': 'TGAAAACACAAACCTCAACA',
    'N27': 'TGAAAACACAAACCTCAACAAACACC',
    'N33': 'TGAAAACACAAACCTCAACAAACACCACACAC',
    'N39': 'TGAAAACACAAACCTCAACAAACACCACACACGAATTC',
    'N45': 'TGAAAACACAAACCTCAACAAACACCACACACGAATTCAACACC',
    'N51': 'TGAAAACACAAACCTCAACAAACACCACACACGAATTCAACACCAACACC',
    'N39SH': 'TGAAAACACAAACCGGCTTCAACAAAAGCCGGTCAGAA',
}


sequence_design = df.iloc[1:train_shape + 1, 0]
sequence_design_array = np.array(sequence_design)
sequence_design_array = sequence_design_array.reshape(train_shape, 1)
trans_level = df.iloc[1:train_shape + 1, 6]
trans_level_array = np.array(trans_level)
trans_level_array = trans_level_array.reshape(train_shape, 1)

sorted_data = np.sort(trans_level_array)[::-1]
data = sorted_data.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(data)

labels = kmeans.labels_
centers = kmeans.cluster_centers_.flatten()
center_0 = centers[0]
distances = np.abs(data.flatten() - center_0)
index_of_center_0 = np.argmin(distances)
center_1 = centers[1]
distances = np.abs(data.flatten() - center_1)
index_of_center_1 = np.argmin(distances)

low_trans = sorted_data[0: index_of_center_0]
med_trans = sorted_data[index_of_center_0: index_of_center_1]
high_trans = sorted_data[index_of_center_1: train_shape]

low_trans_seq = sequence_design_array[0: index_of_center_0]
med_trans_seq = sequence_design_array[index_of_center_0: index_of_center_1]
high_trans_seq = sequence_design_array[index_of_center_1: train_shape]

flatten_sorted_data = sorted_data.tolist()
flatten_sorted_data = list(chain(*flatten_sorted_data))

if False:
    plt.figure(figsize=(10, 6))
    category_labels = []
    low_x = np.random.uniform(-0.05, 0.05, 576)
    med_x = np.random.uniform(0.95, 1.05, 576)
    high_x = np.random.uniform(1.95, 2.05, 576)
    for index, x in enumerate(flatten_sorted_data):
        if x < center_0:
            category_labels.append('low_bin')
        elif center_0 <= x < center_1:
            category_labels.append('med_bin')
        else:
            category_labels.append('high_bin')

    df = pd.DataFrame({'Value': flatten_sorted_data, 'Category': category_labels})
    sns.boxplot(x='Category', y='Value', data=df, width=0.5, gap=0.1, dodge=False, saturation=1,
                palette={"low_bin": 'red', "med_bin": 'green', "high_bin": 'blue'},
                boxprops=dict(alpha=0.3), fliersize=0)

    for index, x in enumerate(flatten_sorted_data):
        if x < center_0:
            plt.scatter(low_x[index], x, s=1.5, color='red')
        elif center_0 <= x < center_1:
            plt.scatter(med_x[index], x, s=1.5, color='green')
        else:
            plt.scatter(high_x[index], x, s=1.5, color='blue')
    plt.ylabel('Transcript-level')
    plt.show()

if False:
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_data, color='grey')
    plt.plot(range(index_of_center_0), sorted_data[:index_of_center_0],  color='red')
    plt.fill_between(x=range(index_of_center_0), y1=0, y2=flatten_sorted_data[0:index_of_center_0],
                     alpha=0.3, color='red')

    plt.plot(range(index_of_center_0, index_of_center_1), sorted_data[index_of_center_0:index_of_center_1],
             color='green')
    plt.fill_between(x=range(index_of_center_0, index_of_center_1),
                     y1=0, y2=flatten_sorted_data[index_of_center_0:index_of_center_1],
                     alpha=0.3, color='green')

    plt.plot(range(index_of_center_1, train_shape), sorted_data[index_of_center_1:train_shape],  color='blue')
    plt.fill_between(x=range(index_of_center_1, train_shape),
                     y1=0, y2=flatten_sorted_data[index_of_center_1:train_shape],
                     alpha=0.3, color='blue')

    plt.axhline(y=centers[0], xmin=0, xmax=index_of_center_0/train_shape, color='red', linestyle='--')
    plt.axhline(y=centers[1], xmin=0, xmax=index_of_center_1/train_shape, color='green', linestyle='--')

    plt.xlim(0, train_shape)
    plt.ylim(0, 16)
    plt.text(0, centers[0], f'Transcription Level: {centers[0]:.3f}, Index: {index_of_center_0}',
             color='red', verticalalignment='bottom')
    plt.text(0, centers[1], f'Transcription Level: {centers[1]:.3f}, Index: {index_of_center_1}',
             color='green', verticalalignment='bottom')
    plt.xlabel('Index')
    plt.ylabel('Transcription Level')
    plt.legend()
    plt.grid()
    plt.show()


# # 将数组转换为列表
# flat_list = [item[0] for item in high_trans_seq]
# # 分词
# words = []
# for item in flat_list:
#     words.extend(item.split('-'))
# # 统计词段出现频率
# word_counts = Counter(words)
# # 计算总词段数
# total_words = sum(word_counts.values())
# # 计算每个词段的概率
# word_probabilities = {word: count / total_words for word, count in word_counts.items()}
# # 输出结果
# for word, probability in word_probabilities.items():
#     print(f"词段: {word}, 概率: {probability:.4f}")
#
# if True:
#     labels = list(word_probabilities.keys())
#     sizes = list(word_probabilities.values())
#     plt.figure(figsize=(10, 8))
#     plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
#     plt.axis('equal')
#     plt.show()
