import numpy as np

tag2label = {'O': 0, 'I-ORG': 1, 'I-LOC': 2, 'I-PER': 3, 'B-LOC': 4, 'B-PER': 5, 'B-ORG': 6}


def compare(y, predict):
    value_y, index_y = ner_vi(y)
    value_pre, index_pre = ner_vi(predict)
    total_ne_y = len(value_y)
    total_ne_pre = len(value_pre)
    count = 0
    for i in range(len(index_pre)):
        if index_pre[i] in index_y and value_pre[i] == value_y[index_y.index(index_pre[i])]:
            count = count + 1
    return count, total_ne_y, total_ne_pre


def ner_vi(y):
    begin = [4, 5, 6]
    end = [1, 2, 3]
    index = []
    value = []
    tmp_value = []
    tmp_index = []
    for i in range(len(y)):
        if y[i] == 0 and len(tmp_value) != 0:
            value.append(tmp_value)
            index.append(tmp_index)
            tmp_value = []
            tmp_index = []
        elif y[i] in begin and len(tmp_value) != 0:
            value.append(tmp_value)
            index.append(tmp_index)
            tmp_value = []
            tmp_index = []
            tmp_value.append(y[i])
            tmp_index.append(i)
        elif y[i] in begin and len(tmp_value) == 0:
            tmp_value.append(y[i])
            tmp_index.append(i)
        elif y[i] in end and len(tmp_value) != 0:
            tmp_value.append(y[i])
            tmp_index.append(i)
    if len(tmp_value) != 0:
        value.append(tmp_value)
        index.append(tmp_index)
    return value, index
