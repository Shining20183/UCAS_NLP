import nltk
import re
import tensorflow as tf
import numpy as np
import keras

class2label = {'Other': 0,
               'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
               'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
               'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
               'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
               'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
               'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
               'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
               'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
               'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}


class Dataloader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.max_len = 0

    def load_data(self):
        text = []
        label = []
        lines = [line.strip() for line in open(self.data_dir)]

        for i in range(0, len(lines), 4):
            sentence = lines[i].split('\t')[1][1:-1].lower()
            sentence = sentence.replace('<e1>', '_e11_').replace('</e1>', '_e12_').replace('<e2>', '_e21_').replace(
                '</e2>', '_e22_')
            sentence = re.sub(r"[^A-Za-z0-9^,!.'+-=_]", " ", sentence)
            relation = lines[i + 1]
            tokens = nltk.word_tokenize(sentence)

            if len(tokens) > self.max_len:
                self.max_len = len(tokens)

            # 将标点符号分词分出去了
            sentence = ' '.join(tokens)
            text.append(sentence)
            label.append(class2label[relation])

        return text, label
