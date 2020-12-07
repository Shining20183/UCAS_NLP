import os
import collections
import codecs
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.mini_frq = mini_frq

        input_file = os.path.join(data_dir, 'input.en.txt')
        vocab_file = os.path.join(data_dir, 'vocab.en.pkl')

        self.preprocess(input_file, vocab_file)
        self.create_batches()
        self.reset_batch_pointer()

    def build_vocab(self, sentences):
        word_counts = collections.Counter()  # 计数生成字典
        if not isinstance(sentences, list):
            sentences = [sentences]
        for sent in sentences:
            word_counts.update(sent)
        vocabulary_inv = ['<START>', '<UNK>', '<END>'] + \
                         [x[0] for x in word_counts.most_common() if x[1] >= self.mini_frq]  # most_common无参数时返回所有元素的列表
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}  # 给词编号
        return [vocabulary, vocabulary_inv]

    def preprocess(self, input_file, vocab_file):
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = f.readlines()
            if lines[0][:1] == codecs.BOM_UTF8:  # 检查是否有BOM
                lines[0] = lines[0][1:]
            lines = [line.strip().split() for line in lines]  # 将串转化为词的列表

        self.vocab, self.words = self.build_vocab(lines)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)  # 将对象持久化保存到文件中

        raw_data = [[0] * self.seq_length +
                    [self.vocab.get(w, 1) for w in line] +  # 找单词在字典中的编号，如果不存在，就返回1
                    [2] * self.seq_length for line in lines]  # 0代表<START>, 2代表<END>
        # 最终生成二维数组，但为什么要×self.seq_length
        self.raw_data = raw_data

    def create_batches(self):
        xdata, ydata = list(), list()
        for row in self.raw_data:  # 这里还不是很明白
            for ind in range(self.seq_length, len(row)):
                xdata.append(row[ind - self.seq_length:ind])
                ydata.append([row[ind]])
        self.num_batches = int(len(xdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = np.array(xdata[:self.num_batches * self.batch_size])
        ydata = np.array(ydata[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

# textloader = TextLoader(data_dir='data/',batch_size=10,seq_length=5)
# x,y=textloader.next_batch()
# print(x.shape)
# print(y.shape)