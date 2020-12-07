from input_data import Dataloader
import tensorflow as tf
import argparse
import numpy as np
from util import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', type=str, default='data/test.txt')
    parser.add_argument('--test_data_dir', type=str, default='data/test.txt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--word_dim', type=int, default=128)
    parser.add_argument('--tag_sum', type=int, default=7)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--min_freq', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--dropout_keep', type=float, default=0.8)

    args = parser.parse_args()
    train_data_loader = Dataloader(args.train_data_dir, args.batch_size, args.min_freq, True, args.max_len)
    args.vocab_size = train_data_loader.vocab_size

    input_data = tf.placeholder(tf.int32, [args.batch_size, args.max_len])
    target = tf.placeholder(tf.int32, [args.batch_size, args.max_len])
    sequence_length = tf.placeholder(tf.int32, [args.batch_size])

    embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
    embeddings = tf.nn.l2_normalize(embeddings, 1)

    input_data_embedding = tf.nn.embedding_lookup(embeddings, input_data)
    input_data_embedding = tf.nn.dropout(input_data_embedding, args.dropout_keep)

    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(args.word_dim, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(args.word_dim, forget_bias=1.0, state_is_tuple=True)

    (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell,
                                                                    input_data_embedding,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length)
    bilstm_output = tf.concat([output_fw, output_bw], axis=2)

    W = tf.get_variable(name='W', shape=[args.batch_size, 2 * args.word_dim, args.tag_sum], dtype=tf.float32,
                        initializer=tf.zeros_initializer())
    b = tf.get_variable(name='b', shape=[args.batch_size, args.max_len, args.tag_sum], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

    bilstm_output = tf.tanh(tf.matmul(bilstm_output, W) + b)

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_output, target,
                                                                          tf.tile(np.array([args.max_len]),
                                                                                  np.array([args.batch_size])))
    transition_params_ = tf.placeholder(shape=transition_params.shape, dtype=tf.float32)
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_output, transition_params_,
                                                                tf.tile(np.array([args.max_len], dtype=np.int32),
                                                                        np.array([args.batch_size], dtype=np.int32)))

    loss = tf.reduce_mean(-log_likelihood)
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    train_op = optimizer.minimize(loss)

    # train model
    sess = tf.Session()
    print('---------- train ----------')
    sess.run(tf.global_variables_initializer())
    for e in range(args.num_epochs):
        print('epoch:', e)
        train_data_loader.reset_batch_pointer()
        for b in range(train_data_loader.num_batches):
            x, y, sl = train_data_loader.next_batch()
            # print(sl)
            feed = {input_data: x, target: y, sequence_length: sl}
            tf_transition_params, _ = sess.run([transition_params, train_op], feed)
            # sess.run(train_op, feed)
            if b % 100 == 0:
                train_loss = sess.run(loss, feed)
                print('iter', b, ' loss:', train_loss)

    # test model
    print('---------- test ----------')
    test_data_loader = Dataloader(args.test_data_dir, args.test_batch_size, args.min_freq, False, args.max_len)
    test_data_loader.reset_batch_pointer()
    total_y_num = 0
    total_pre_num = 0
    total_true_num = 0
    for b in range(test_data_loader.num_batches):
        x, y, sl = test_data_loader.next_batch()

        feed = {input_data: x, target: y, transition_params_: tf_transition_params, sequence_length: sl}
        tf_viterbi_sequence, tf_viterbi_score = sess.run([viterbi_sequence, viterbi_score], feed)
        for pre, y_ in zip(tf_viterbi_sequence, y):
            true_num, y_num, pre_num = compare(y_, pre)
            total_true_num += true_num
            total_y_num += y_num
            total_pre_num += pre_num

    # 准确率 = 交集 / 模型抽取出的实体
    # 召回率 = 交集 / 数据集中的所有实体
    # f值 = 2×(准确率×召回率) / (准确率 + 召回率)
    if total_pre_num == 0:
        # print('total pre num is 0')
        accuracy = 0
    else:
        accuracy = total_true_num / total_pre_num
    if total_y_num == 0:
        # print('total y num is 0')
        recall = 0
    else:
        recall = total_true_num / total_y_num
    if total_pre_num == 0 or total_y_num == 0:
        # print('f1 is 0')
        f1 = 0
    else:
        f1 = 2 * accuracy * recall / (accuracy + recall)

    with open('result.txt', 'w') as fin:
        fin.write('------------test result-------------')
        fin.write('total true num:' + str(total_true_num) + '\n')
        fin.write('total y num:' + str(total_y_num) + '\n')
        fin.write('total predict num:' + str(total_pre_num) + '\n')
        fin.write('accuracy:' + str(accuracy) + '\n')
        fin.write('recall:' + str(recall) + '\n')
        fin.write('f1-score:' + str(f1) + '\n')

    print('total true num:' + str(total_true_num))
    print('total y num:' + str(total_y_num))
    print('total predict num:' + str(total_pre_num))
    print('accuracy:' + str(accuracy))
    print('recall:' + str(recall))
    print('f1-score:' + str(f1))


if __name__ == '__main__':
    main()
