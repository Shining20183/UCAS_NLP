from input_data import Dataloader
import tensorflow as tf
import numpy as np
import keras
from keras.optimizers import Adadelta

embedding_dim = 300
max_len = 0
dropout_rate = 0.5


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[2],), trainable=True)

    def call(self, inputs):
        v = tf.tanh(inputs)
        vu = tf.tensordot(v, self.W, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        output = tf.tanh(output)
        print('output:', output.shape)
        return output


class AttLSTM(tf.keras.Model):
    def __init__(self):
        super(AttLSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=95, mask_zero=True)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.attention = SelfAttention()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(19, activation='softmax')

        self.forward_layer = tf.keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=dropout_rate)
        self.backward_layer = tf.keras.layers.LSTM(embedding_dim, activation='relu', return_sequences=True,
                                                   go_backwards=True, dropout=dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(self.forward_layer, backward_layer=self.backward_layer)

    def call(self, inputs):
        outs = self.embedding(inputs)
        outs = self.dropout1(outs)
        outs = self.bilstm(outs)
        outs = self.attention(outs)
        outs = self.dense1(outs)
        outs = self.dense2(outs)
        return outs


train_dataloader = Dataloader('SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
train_text, train_label = train_dataloader.load_data()
max_len = train_dataloader.max_len

tokenizer = keras.preprocessing.text.Tokenizer(oov_token='_oov_')
tokenizer.fit_on_texts(train_text)
train_text = tokenizer.texts_to_sequences(train_text)
train_text = keras.preprocessing.sequence.pad_sequences(train_text, max_len, padding='post')
train_label = np.array(train_label)
train_label = keras.utils.to_categorical(train_label, 19)

print('train_text shape:', train_text.shape)

# train_dataset = tf.data.Dataset.from_tensor_slices((train_text, train_label))

vocab_size = len(tokenizer.word_index) + 1

test_dataloader = Dataloader('SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
test_text, test_label = test_dataloader.load_data()
test_text = tokenizer.texts_to_sequences(test_text)
test_text = keras.preprocessing.sequence.pad_sequences(test_text, max_len, padding='post')
test_label = np.array(test_label)
# print(test_label)

model = AttLSTM()
Optimizer = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.9)
model.compile(optimizer=Optimizer, loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.Accuracy()])
model.fit(train_text, train_label, batch_size=10, epochs=30)

# model.summary()

print('predict')
prediction = model.predict(test_text)
prediction = tf.argmax(prediction, axis=1)
print(prediction)

with open('model_pre.txt', 'w') as fin:
    for i in prediction.numpy():
        fin.write(str(i) + '\n')

correct = tf.cast(tf.equal(prediction.numpy(), test_label), dtype=tf.int32)
correct = tf.reduce_sum(correct)
total = len(prediction.numpy())
acc = correct / total
print('correct:', correct)
print('total:', total)
print('accuracy:', acc)
