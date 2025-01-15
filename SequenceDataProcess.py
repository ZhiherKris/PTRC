import pandas as pd
import numpy as np
# from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

data_path = 'D:\\BioData\\PTRC实验数据.xlsx'
table = 2
df = pd.read_excel(data_path, sheet_name=4)

test_shape = 100
train_shape = 576 - test_shape

trans_level = df.iloc[1:train_shape + 1, 6]
trans_level_array = np.array(trans_level)
trans_level_array = trans_level_array.reshape(train_shape, 1)

sequence_design = df.iloc[1:train_shape + 1, 0]
sequence_design_array = np.array(sequence_design)
sequence_design_array = sequence_design_array.reshape(train_shape, 1)

sequence_test = df.iloc[577 - test_shape:577, 0]
sequence_test_array = np.array(sequence_test)
sequence_test_array = sequence_test_array.reshape(test_shape, 1)

trans_test = df.iloc[577 - test_shape:577, 6]
trans_test_array = np.array(trans_test)
trans_test_array = trans_test_array.reshape(test_shape, 1)

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

for i in range(len(sequence_design_array)):
    for j in range(len(sequence_design_array[i])):
        letters = sequence_design_array[i][j].split('-')
        replaced = [seq_map[letter] for letter in letters]
        sequence_design_array[i][j] = ''.join(replaced)

for i in range(len(sequence_test_array)):
    for j in range(len(sequence_test_array[i])):
        letters = sequence_test_array[i][j].split('-')
        replaced = [seq_map[letter] for letter in letters]
        sequence_test_array[i][j] = ''.join(replaced)

# print(sequence_test_array)

base_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
kmer_dict_2 = {
    'AA': 0, 'AT': 1, 'AC': 2, 'AG': 3,
    'TA': 4, 'TT': 5, 'TC': 6, 'TG': 7,
    'CA': 8, 'CT': 9, 'CC': 10, 'CG': 11,
    'GA': 12, 'GT': 13, 'GC': 14, 'GG': 15
}

kmer_dict_3 = {
    'AAA': 0, 'AAT': 1, 'AAC': 2, 'AAG': 3,
    'ATA': 4, 'ATT': 5, 'ATC': 6, 'ATG': 7,
    'ACA': 8, 'ACT': 9, 'ACC': 10, 'ACG': 11,
    'AGA': 12, 'AGT': 13, 'AGC': 14, 'AGG': 15,
    'TAA': 16, 'TAT': 17, 'TAC': 18, 'TAG': 19,
    'TTA': 20, 'TTT': 21, 'TTC': 22, 'TTG': 23,
    'TCA': 24, 'TCT': 25, 'TCC': 26, 'TCG': 27,
    'TGA': 28, 'TGT': 29, 'TGC': 30, 'TGG': 31,
    'CAA': 32, 'CAT': 33, 'CAC': 34, 'CAG': 35,
    'CTA': 36, 'CTT': 37, 'CTC': 38, 'CTG': 39,
    'CCA': 40, 'CCT': 41, 'CCC': 42, 'CCG': 43,
    'CGA': 44, 'CGT': 45, 'CGC': 46, 'CGG': 47,
    'GAA': 48, 'GAT': 49, 'GAC': 50, 'GAG': 51,
    'GTA': 52, 'GTT': 53, 'GTC': 54, 'GTG': 55,
    'GCA': 56, 'GCT': 57, 'GCC': 58, 'GCG': 59,
    'GGA': 60, 'GGT': 61, 'GGC': 62, 'GGG': 63
}

kmer_dict_4 = {
    'AAAA': 0, 'AAAT': 1, 'AAAC': 2, 'AAAG': 3,
    'AATA': 4, 'AATT': 5, 'AATC': 6, 'AATG': 7,
    'AACA': 8, 'AACT': 9, 'AACC': 10, 'AACG': 11,
    'AAGA': 12, 'AAGT': 13, 'AAGC': 14, 'AAGG': 15,
    'ATAA': 16, 'ATAT': 17, 'ATAC': 18, 'ATAG': 19,
    'ATTA': 20, 'ATTT': 21, 'ATTC': 22, 'ATTG': 23,
    'ATCA': 24, 'ATCT': 25, 'ATCC': 26, 'ATCG': 27,
    'ATGA': 28, 'ATGT': 29, 'ATGC': 30, 'ATGG': 31,
    'ACAA': 32, 'ACAT': 33, 'ACAC': 34, 'ACAG': 35,
    'ACTA': 36, 'ACTT': 37, 'ACTC': 38, 'ACTG': 39,
    'ACCA': 40, 'ACCT': 41, 'ACCC': 42, 'ACCG': 43,
    'ACGA': 44, 'ACGT': 45, 'ACGC': 46, 'ACGG': 47,
    'AGAA': 48, 'AGAT': 49, 'AGAC': 50, 'AGAG': 51,
    'AGTA': 52, 'AGTT': 53, 'AGTC': 54, 'AGTG': 55,
    'AGCA': 56, 'AGCT': 57, 'AGCC': 58, 'AGCG': 59,
    'AGGA': 60, 'AGGT': 61, 'AGGC': 62, 'AGGG': 63,
    'TAAA': 64, 'TAAT': 65, 'TAAC': 66, 'TAAG': 67,
    'TATA': 68, 'TATT': 69, 'TATC': 70, 'TATG': 71,
    'TACA': 72, 'TACT': 73, 'TACC': 74, 'TACG': 75,
    'TAGA': 76, 'TAGT': 77, 'TAGC': 78, 'TAGG': 79,
    'TTAA': 80, 'TTAT': 81, 'TTAC': 82, 'TTAG': 83,
    'TTTA': 84, 'TTTT': 85, 'TTTC': 86, 'TTTG': 87,
    'TTCA': 88, 'TTCT': 89, 'TTCC': 90, 'TTCG': 91,
    'TTGA': 92, 'TTGT': 93, 'TTGC': 94, 'TTGG': 95,
    'TCAA': 96, 'TCAT': 97, 'TCAC': 98, 'TCAG': 99,
    'TCTA': 100, 'TCTT': 101, 'TCTC': 102, 'TCTG': 103,
    'TCCA': 104, 'TCCT': 105, 'TCCC': 106, 'TCCG': 107,
    'TCGA': 108, 'TCGT': 109, 'TCGC': 110, 'TCGG': 111,
    'TGAA': 112, 'TGAT': 113, 'TGAC': 114, 'TGAG': 115,
    'TGTA': 116, 'TGTT': 117, 'TGTC': 118, 'TGTG': 119,
    'TGCA': 120, 'TGCT': 121, 'TGCC': 122, 'TGCG': 123,
    'TGGA': 124, 'TGGT': 125, 'TGGC': 126, 'TGGG': 127,
    'CAAA': 128, 'CAAT': 129, 'CAAC': 130, 'CAAG': 131,
    'CATA': 132, 'CATT': 133, 'CATC': 134, 'CATG': 135,
    'CACA': 136, 'CACT': 137, 'CACC': 138, 'CACG': 139,
    'CAGA': 140, 'CAGT': 141, 'CAGC': 142, 'CAGG': 143,
    'CTAA': 144, 'CTAT': 145, 'CTAC': 146, 'CTAG': 147,
    'CTTA': 148, 'CTTT': 149, 'CTTC': 150, 'CTTG': 151,
    'CTCA': 152, 'CTCT': 153, 'CTCC': 154, 'CTCG': 155,
    'CTGA': 156, 'CTGT': 157, 'CTGC': 158, 'CTGG': 159,
    'CCAA': 160, 'CCAT': 161, 'CCAC': 162, 'CCAG': 163,
    'CCTA': 164, 'CCTT': 165, 'CCTC': 166, 'CCTG': 167,
    'CCCA': 168, 'CCCT': 169, 'CCCC': 170, 'CCCG': 171,
    'CCGA': 172, 'CCGT': 173, 'CCGC': 174, 'CCGG': 175,
    'CGAA': 176, 'CGAT': 177, 'CGAC': 178, 'CGAG': 179,
    'CGTA': 180, 'CGTT': 181, 'CGTC': 182, 'CGTG': 183,
    'CGCA': 184, 'CGCT': 185, 'CGCC': 186, 'CGCG': 187,
    'CGGA': 188, 'CGGT': 189, 'CGGC': 190, 'CGGG': 191,
    'GAAA': 192, 'GAAT': 193, 'GAAC': 194, 'GAAG': 195,
    'GATA': 196, 'GATT': 197, 'GATC': 198, 'GATG': 199,
    'GACA': 200, 'GACT': 201, 'GACC': 202, 'GACG': 203,
    'GAGA': 204, 'GAGT': 205, 'GAGC': 206, 'GAGG': 207,
    'GTAA': 208, 'GTAT': 209, 'GTAC': 210, 'GTAG': 211,
    'GTTA': 212, 'GTTT': 213, 'GTTC': 214, 'GTTG': 215,
    'GTCA': 216, 'GTCT': 217, 'GTCC': 218, 'GTCG': 219,
    'GTGA': 220, 'GTGT': 221, 'GTGC': 222, 'GTGG': 223,
    'GCAA': 224, 'GCAT': 225, 'GCAC': 226, 'GCAG': 227,
    'GCTA': 228, 'GCTT': 229, 'GCTC': 230, 'GCTG': 231,
    'GCCA': 232, 'GCCT': 233, 'GCCC': 234, 'GCCG': 235,
    'GCGA': 236, 'GCGT': 237, 'GCGC': 238, 'GCGG': 239,
    'GGAA': 240, 'GGAT': 241, 'GGAC': 242, 'GGAG': 243,
    'GGTA': 244, 'GGTT': 245, 'GGTC': 246, 'GGTG': 247,
    'GGCA': 248, 'GGCT': 249, 'GGCC': 250, 'GGCG': 251,
    'GGGA': 252, 'GGGT': 253, 'GGGC': 254, 'GGGG': 255
}

max_len = max(len(seq[0]) for seq in sequence_design_array)


def one_hot_encode(sequence, base_dict, max_len):
    encoding = np.zeros((max_len, len(base_dict)))
    for i, base in enumerate(sequence):
        encoding[i, base_dict[base]] = 1
    return encoding


one_hot_sequence = [one_hot_encode(seq[0], base_dict, max_len) for seq in sequence_design_array]
one_hot_sequence = np.array(one_hot_sequence)
new_sequence = [one_hot_encode(seq[0], base_dict, max_len) for seq in sequence_test_array]
new_sequence = np.array(new_sequence)


def encode_to_2mer(sequence):
    k = 2
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return kmers


def encode_to_3mer(sequence):
    k = 3
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return kmers


def encode_to_4mer(sequence):
    k = 4
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    return kmers


encoded_sequences = []
encodedtest_sequences = []

for seq_list in sequence_design_array:
    encoded_seq_list = [encode_to_2mer(seq) for seq in seq_list]
    encoded_sequences.append(encoded_seq_list)

for seq_list in sequence_test_array:
    encoded_seq_list = [encode_to_2mer(seq) for seq in seq_list]
    encodedtest_sequences.append(encoded_seq_list)

sequence_3mer_encoded = []
sequence_test_3mer_encoded = []

for seq_list in sequence_design_array:
    encoded_seq_list = [encode_to_3mer(seq) for seq in seq_list]
    sequence_3mer_encoded.append(encoded_seq_list)

for seq_list in sequence_test_array:
    encoded_seq_list = [encode_to_3mer(seq) for seq in seq_list]
    sequence_test_3mer_encoded.append(encoded_seq_list)


sequence_4mer_encoded = []
sequence_test_4mer_encoded = []

for seq_list in sequence_design_array:
    encoded_seq_list = [encode_to_4mer(seq) for seq in seq_list]
    sequence_4mer_encoded.append(encoded_seq_list)

for seq_list in sequence_test_array:
    encoded_seq_list = [encode_to_4mer(seq) for seq in seq_list]
    sequence_test_4mer_encoded.append(encoded_seq_list)

max_len2 = max(len(seq[0]) for seq in encoded_sequences)
max_len3 = max(len(seq[0]) for seq in sequence_3mer_encoded)
max_len4 = max(len(seq[0]) for seq in sequence_4mer_encoded)

two_mer_onehot_sequence = [one_hot_encode(seq[0], kmer_dict_2, max_len2) for seq in encoded_sequences]
two_mer_onehot_sequence = np.array(two_mer_onehot_sequence)

new_sequence_2mer = [one_hot_encode(seq[0], kmer_dict_2, max_len2) for seq in encodedtest_sequences]
new_sequence_2mer = np.array(new_sequence_2mer)

three_mer_onehot_sequence = [one_hot_encode(seq[0], kmer_dict_3, max_len3) for seq in sequence_3mer_encoded]
three_mer_onehot_sequence = np.array(three_mer_onehot_sequence)

new_sequence_3mer = [one_hot_encode(seq[0], kmer_dict_3, max_len3) for seq in sequence_test_3mer_encoded]
new_sequence_3mer = np.array(new_sequence_3mer)

four_mer_onehot_sequence = [one_hot_encode(seq[0], kmer_dict_4, max_len4) for seq in sequence_4mer_encoded]
four_mer_onehot_sequence = np.array(four_mer_onehot_sequence)

new_sequence_4mer = [one_hot_encode(seq[0], kmer_dict_4, max_len4) for seq in sequence_test_4mer_encoded]
new_sequence_4mer = np.array(new_sequence_4mer)

#
# def chunk_2(sequence):
#     pairs = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
#     if len(pairs[-1]) == 1:
#         pairs[-1] = sequence[0] + pairs[-1]
#     return pairs
#
#
# for seq_list in sequence_design_array:
#     chunk_2_sequence = [chunk_2(seq) for seq in seq_list]
#     chunk_2_sequence.append(chunk_2_sequence)
#
# print(chunk_2_sequence.shape)
#
# # two_mer_onehot_sequence = [one_hot_encode(seq[0], kmer_dict_2, max_len2) for seq in chunk_2_sequence]
# # two_mer_onehot_sequence = np.array(two_mer_onehot_sequence)
#
#
#
#
# def group_sequence(sequence, group_size=3):
#     groups = [sequence[i:i + group_size] for i in range(0, len(sequence), group_size)]
#     if len(groups[-1]) < group_size:  # 如果最后一组字符不足指定大小，与序列的首字母组成一组
#         missing_chars = group_size - len(groups[-1])
#         groups[-1] += sequence[:missing_chars]
#     return groups
