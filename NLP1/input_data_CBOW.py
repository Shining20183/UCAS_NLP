import os
import collections
import codecs
from six.moves import cPickle
import numpy as np


class TextLoader():
    def __init__(self, data_dir='data/', batch_size=100, win_size=2, mini_frq=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.win_size = win_size
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

        win_seqs = []
        label_indices = []
        for line in lines:
            win_seq = [[self.vocab.get(w,1) for w in line[max((ix-self.win_size),0):(ix + self.win_size+1)] ] for ix,x in enumerate(line)]
            # win_seq = [line[max((ix-self.win_size),0):(ix + self.win_size+1)] for ix,x in enumerate(line)]
            label_indice = [ix if ix<self.win_size else self.win_size for ix,x in enumerate(line)]
            win_seqs += win_seq
            label_indices += label_indice
        self.batch_indices = zip(win_seqs,label_indices)


    def create_batches(self):
        batch_labels = [(x[:y]+x[(y+1):],x[y]) for x,y in self.batch_indices]
        batch_labels = [(x,y) for x,y in batch_labels if len(x) == 2*self.win_size]
        xdata, ydatat = [list(x) for x in zip(*batch_labels)]
        ydata = [[i] for i in ydatat]
        self.num_batches = int(len(xdata)/self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        xdata = np.array(xdata[:self.num_batches*self.batch_size])
        ydata = np.array(ydata[:self.num_batches*self.batch_size])
        self.x_batches = np.split(xdata,self.num_batches,0)
        self.y_batches = np.split(ydata,self.num_batches,0)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

# data_loader = TextLoader()
# x,y = data_loader.next_batch()
#
# print(x.shape)
# print(y.shape)
