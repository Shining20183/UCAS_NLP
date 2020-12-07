from input_data_CBOW import *

import numpy as np
import tensorflow as tf
import argparse
import time
import math

def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir',type=str,default='data/')
    parse.add_argument('--batch_size',type=int,default=120)
    parse.add_argument('--win_size',type=int,default=3)
    parse.add_argument('--num_epoch',type=int,default=20)
    parse.add_argument('--word_dim',type=int,default=100)
    parse.add_argument('--learning_rate',type=float,default=0.001)

    args = parse.parse_args()
    data_loader = TextLoader(data_dir=args.data_dir,batch_size=args.batch_size,win_size=args.win_size)
    args.vocab_size = data_loader.vocab_size

    graph = tf.Graph()
    with graph.as_default():
        input_data = tf.placeholder(tf.int32,[args.batch_size,2*args.win_size])
        target = tf.placeholder(tf.int32,[args.batch_size,1])

        weight = tf.Variable(tf.truncated_normal([args.vocab_size,args.word_dim], stddev=1.0 / np.sqrt(args.word_dim)))
        bias = tf.Variable(tf.zeros([args.vocab_size]))

        embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
        embed = tf.zeros([args.batch_size, args.word_dim])
        for e in range(2*args.win_size):
            embed+=tf.nn.embedding_lookup(embeddings,input_data[:,e])

        loss = tf.reduce_mean(tf.nn.nce_loss(weight,bias,target,embed,int(args.batch_size/2),args.vocab_size))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss)

        # 输出词向量
        embeddings_norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / embeddings_norm

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for e in range(args.num_epoch):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start=time.time()
                x,y=data_loader.next_batch()
                feed={input_data:x,target:y}
                train_loss,_=sess.run([loss,optimizer],feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(
                    b, data_loader.num_batches,
                    e, train_loss, end - start))
            # 保存词向量至nnlm_word_embeddings.npy文件
            np.save('cbow_word_embeddings.en', normalized_embeddings.eval())


if __name__ == '__main__':
    main()









