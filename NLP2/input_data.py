import collections
import numpy as np

# zhong_punc = {'，','。','！','；','？'}
tag2label = {'O': 0, 'I-ORG': 1, 'I-LOC': 2, 'I-PER': 3, 'B-LOC': 4, 'B-PER': 5, 'B-ORG': 6}


class Dataloader():
    def __init__(self, data_dir, batch_size, min_freq, is_train, max_len):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.min_freq = min_freq
        self.max_len = 0
        self.max_len = max_len
        if is_train:
            self.id_data, self.sequence_length = self.train_process()
        else:
            self.id_data, self.sequence_length = self.test_process()
        self.create_batches()
        self.reset_batch_pointer()

    def read_data(self):
        data = []
        with open(self.data_dir, encoding='utf-8') as f:
            lines = f.readlines()
            text, tag = [], []
            for line in lines:
                # print(line)
                if line != '\n':
                    [char, label] = line.strip().split()
                    if char.isdigit():
                        char = '<NUM>'
                    text.append(char)
                    tag.append(label)
                else:
                    if len(text) != 0:
                        data.append((text, tag))
                        text, tag = [], []
            if len(text) != 0:
                data.append((text, tag))
        return data

    def build_vocab(self, data):
        word_counts = collections.Counter()
        for text, tag in data:
            word_counts.update(text)
        vocabulary_inv = ['<PAD>'] + [x[0] for x in word_counts.most_common() if x[1] >= self.min_freq] + ['<UNK>']
        # print(vocabulary_inv)
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        # print(vocabulary)
        self.vocab_size = len(vocabulary_inv)
        return [vocabulary, vocabulary_inv]

    def pad_sentag(self, data):
        pad_data = []
        sequence_length = []
        for sen, tag in data:
            if len(sen) < self.max_len:
                pad_num = self.max_len - len(sen)
                sequence_length.append(len(sen))
                pad_data.append([sen + ['<PAD>'] * pad_num, tag + ['O'] * pad_num])
        # print(pad_data)
        return pad_data, sequence_length

    def sentence2id(self, sentence, vocabulary):
        sentence_id = []
        for char in sentence:
            if char not in vocabulary:
                print("not in dict:", char)
                char = '<UNK>'
            sentence_id.append(vocabulary[char])
        # print(sentence_id)
        return sentence_id

    def train_process(self):
        data = self.read_data()
        vocabulary, vocabulary_inv = self.build_vocab(data)
        np.save('tmpdata/vocabulary.npy', vocabulary)
        data, sequence_length = self.pad_sentag(data)
        id_data = []
        for text, tag in data:
            id_data.append((self.sentence2id(text, vocabulary), [tag2label[i] for i in tag]))
        # print(id_data)
        return id_data, sequence_length

    def test_process(self):
        data = self.read_data()
        vocabulary = np.load('tmpdata/vocabulary.npy', allow_pickle=True).item()
        # print(vocabulary)
        self.vocab_size = len(vocabulary)
        data, sequence_length = self.pad_sentag(data)
        id_data = []
        for text, tag in data:
            id_data.append((self.sentence2id(text, vocabulary), [tag2label[i] for i in tag]))
        return id_data, sequence_length

    def create_batches(self):
        tmpdata = np.array(self.id_data)
        tmpsl = np.array(self.sequence_length)
        self.num_batches = int(len(tmpdata) / self.batch_size)
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        xdata = tmpdata[:, 0][:self.num_batches * self.batch_size]
        ydata = tmpdata[:, 1][:self.num_batches * self.batch_size]
        sl = tmpsl[:self.num_batches * self.batch_size]

        self.x_batches = np.split(xdata, self.num_batches, 0)
        self.y_batches = np.split(ydata, self.num_batches, 0)
        self.sl_batches = np.split(sl, self.num_batches, 0)

    def next_batch(self):
        x, y, sl = self.x_batches[self.pointer], self.y_batches[self.pointer], self.sl_batches[self.pointer]
        self.pointer += 1
        return x, y, sl

    def reset_batch_pointer(self):
        self.pointer = 0
