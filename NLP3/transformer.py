"""
ref:https://www.tensorflow.org/tutorials/text/transformer

"""
# 导入模块
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import time
import sacrebleu

# 配置超参数
parser = argparse.ArgumentParser()
parser.add_argument('--buffer_size', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=40)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--dff', type=int, default=512)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--dropout_rate', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--is_test', type=int, default=0)
args = parser.parse_args()


# 加载数据集
# zh-> en
def load_data(data):
    zh_path = 'data/' + data + '.zh.tok'
    en_path = 'data/' + data + '.en.tok'
    zh_file = open(zh_path, 'r', encoding='UTF-8')
    zh_lines = zh_file.readlines()
    en_file = open(en_path, 'r', encoding='UTF-8')
    en_lines = en_file.readlines()
    # 切分传入Tensor的第一个维度，生成相应的dataset，形成中文和英文的映射
    examples = tf.data.Dataset.from_tensor_slices((zh_lines, en_lines))
    zh_file.close()
    en_file.close()
    return examples


train_examples = load_data('train')  # 加载训练数据集
test_examples = load_data('test')  # 加载测试数据集
# 从训练数据集创建自定义子词分词器（subwords tokenizer）
tokenizer_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (zh.numpy() for zh, en in train_examples), target_vocab_size=2 ** 13)
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for zh, en in train_examples), target_vocab_size=2 ** 13)
print('finish build vocab')

# buffer_size = 20000
# batch_size = 64
def encode(lang1, lang2):
    """
    function:为输入和目标添加开始和结束token
    encode:如果单词不在词典中，则分词器（tokenizer）通过将单词分解为子词来对字符串进行编码。
    如：原句是Transformer is awesome.经过编码后：
        7915 ----> T
        1248 ----> ran
        7946 ----> s
        7194 ----> former
        13 ----> is
        2799 ----> awesome
        7877 ----> .
    """
    lang1 = [tokenizer_zh.vocab_size] + tokenizer_zh.encode(
        lang1.numpy()) + [tokenizer_zh.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2

# max_length = 40
def filter_max_length(x, y, max_length=args.max_length):
    # 过滤掉长度大于40个标记的样本
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)
def tf_encode(zh, en):
    result_zh, result_en = tf.py_function(encode, [zh, en], [tf.int64, tf.int64])
    result_zh.set_shape([None])
    result_en.set_shape([None])
    return result_zh, result_en


# padded_shapes = ([None], ())
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(args.buffer_size)
train_dataset = train_dataset.padded_batch(args.batch_size,
                                           padded_shapes=tf.compat.v1.data.get_output_shapes(train_dataset))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_examples.map(tf_encode)
test_dataset = test_dataset.filter(filter_max_length)
test_dataset = test_dataset.padded_batch(args.batch_size,
                                         padded_shapes=tf.compat.v1.data.get_output_shapes(test_dataset))
print('finish load data')
def get_angles(position, i, d_model):
    return position / np.power(10000., 2. * (i // 2.) / np.float(d_model))

def position_encoding(pos, d_model):
    # 位置编码，将位置为pos的词嵌入到d_model的向量空间中
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 第2i也即偶数项使用sin，
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1即奇数项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
# mask将为0的地方填充为1，其他地方填充为0
def create_padding_mask(seq):
    # 添加额外的维度来将填充加到注意力对数（logits）。
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# look ahead mask
# 逐步遮挡（显露）序列, 1 2 3 ...
def create_look_ahead_mask(size):
    '''
       eg.
       x = tf.random.uniform((1, 3))
       temp = create_look_ahead_mask(x.shape[1])
       temp:<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
               array([[0., 1., 1.],
                      [0., 0., 1.],
                      [0., 0., 0.]], dtype=float32)>
       '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
# scale dot-product attention
def scaled_dot_product_attention(q, k, v, mask):
    """计算注意力权重。
     q, k, v 必须具有匹配的前置维度。
     k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
     虽然 mask 根据其类型（填充或前瞻）有不同的形状，
     但是 mask 必须能进行广播转换以便求和。
     参数:
       q: 请求的形状 == (..., seq_len_q, depth)
       k: 主键的形状 == (..., seq_len_k, depth)
       v: 数值的形状 == (..., seq_len_v, depth_v)
       mask: Float 张量，其形状能转换成
             (..., seq_len_q, seq_len_k)。默认为None。
     返回值:
       输出，注意力权重
     """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # seq_len_q个位置分别对应v上的加权求和
    output = tf.matmul(attention_weights, v)
    return output, attention_weights  # 输出表示注意力权重和 V（数值）向量的乘积。


# 多头注意力
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.depth = self.d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 将Q、K、V拆分到多个头
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # (batch_size, seq_len_q, d_model)
        k = self.split_heads(k, batch_size)  # (batch_size, seq_len_q, d_model)
        v = self.split_heads(v, batch_size)  # (batch_size, seq_len_q, d_model)
        # scaled_attention, (batch_size, num_heads, seq_len_q, depth_v)
        # attention_weights, (batch_size, num_heads, seq_len_q, seq_len)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth_v)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        # 返回新的同样长度的向量
        return output, attention_weights


# 点式前馈网络由两层全联接层组成，两层之间有一个 ReLU 激活函数。
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    '''
    EncoderLayer
    包括：多头注意力（有填充遮挡）和点式前馈网络（Point wise feed forward networks）。
    out1 = LayerNormalization()
    out2 = LayerNormalization()
    '''

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    '''
    Decoder block包含：
    1.遮挡的多头注意力（前瞻遮挡和填充遮挡）
    2.多头注意力（用填充遮挡）。V（数值）和 K（主键）接收编码器输出作为输入。Q（请求）接收遮挡的多头注意力子层的输出。
    3. 点式前馈网络
    out1 = LayerNormalization( x +（MultiHeadAttention(x, x, x)=>dropout）)
    out2 = LayerNormalization( out1 +（MultiHeadAttention(enc_output, enc_output out1)=>dropout）)
    out3 = LayerNormalization( out2 + (ffn => dropout) )
    '''
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_blocks1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_blocks2 = self.mha2(enc_output, enc_output, out1,
                                                padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_blocks1, attn_weights_blocks2

class Encoder(tf.keras.layers.Layer):
    '''
    编码器包括：
    输入嵌入（Input Embedding）
    位置编码（Positional Encoding）
    N 个编码器层（encoder layers）
    输入经过嵌入（embedding）后，该嵌入与位置编码相加。
    该加法结果的输出是编码器层的输入。编码器的输出是解码器的输入。
   '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def __call__(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        # 将嵌入和位置编码相加。
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # (batch_size, input_seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    '''
    Decoder包括：
    输出嵌入（Output Embedding）
    位置编码（Positional Encoding）
    N 个解码器层（decoder layers）
    目标（target）经过一个嵌入后，该嵌入和位置编码相加。该加法结果是解码器层的输入。解码器的输出是最后的线性层的输入。
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = position_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    # x:(batch_size, target_seq_len)
    # enc_output:(batch_size, input_seq_len, d_model)
    def __call__(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights  # x: (batch_size, target_seq_len, d_model)

class Transformer(tf.keras.Model):
    '''
    创建Transformer
    包括编码器，解码器和最后的线性层。
    解码器的输出是线性层的输入，返回线性层的输出。
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def __call__(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training,
                                  enc_padding_mask)  # 编码器，将其输出作为解码器的输入， (batch_size, inp_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # 解码器的输出作为线性层的输入 (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights  # 返回线性层的输出

# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8

input_vocab_size = tokenizer_zh.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2

# dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(args.d_model)  # 学习速率调度程序
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# 计算损失
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # print('loss_function/real:',real)
    # print('loss_function/pred:',pred)
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# 累计一个epoch中的batch的loss，最后求平均，得到一个epoch的loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
# 累计一个epoch中的batch的acc，最后求平均，得到一个epoch的acc
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
transformer = Transformer(args.num_layers, args.d_model, args.num_heads, args.dff, input_vocab_size,
                          target_vocab_size,
                          pe_input=input_vocab_size, pe_target=target_vocab_size, dropout_rate=args.dropout_rate)

def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)
    # 在解码器的第二个注意力模块使用,该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)
    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    # 广播机制，look_ahead_mask==>(batch_size, 1, tar_seq_len, tar_seq_len)
    # dec_target_padding_mask ==> (batch_size, 1, tar_seq_len, tar_seq_len)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


'''
创建检查点的路径和检查点管理器（manager）,每 n 个周期（epochs）保存一次检查点。
'''
# 检查点保存或查找路径
checkpoint_path = './checkpoints/train'
# 检查点
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('restored lasted checkpoint')

# epochs = 20
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

'''
 @tf.function 将追踪-编译 train_step 到 TF 图中，以便更快地执行。
 该函数专用于参数张量的精确形状。为了避免由于可变序列长度或可变批次大小（最后一批次较小）导致的再追踪，
 使用 input_signature 指定更多的通用形状。
'''
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    # print('train_step/inp.shape:',inp.shape)
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        '''
        target被分成了 tar_inp 和 tar_real
        tar_inp 作为输入传递到解码器。tar_real 是位移了 1 的同一个输入
        在 tar_inp 中的每个位置，tar_real 包含了应该被预测到的下一个token
        '''
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        # print('train_step/predictions:', predictions)
        # print('train_step/tar_real:', tar_real)

        loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)

# 模型评估
def evaluate(inp_sentencce):
    start_token = [tokenizer_zh.vocab_size]  # 开始token
    end_token = [tokenizer_zh.vocab_size]  # 结束token
    # 输入语句是中文，并增加开始和结束标记
    inp_sentencce = start_token + tokenizer_zh.encode(inp_sentencce) + end_token
    encoder_input = tf.expand_dims(inp_sentencce, 0)
    # 输入 transformer 的第一个词为英语的开始标记
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(args.max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        # predictions： (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask,
                                                     dec_padding_mask)
        # 从 seq_len 维度选择最后一个词
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        # 如果 predicted_id 等于结束标记，就返回结果
        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights
        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights


# 与evaluate方法的功能相同
def evaluate2(inp_sentencce_id):
    # start_token = [tokenizer_zh.vocab_size]
    # end_token = [tokenizer_zh.vocab_size]
    #
    # inp_sentencce = start_token + tokenizer_zh.encode(inp_sentencce) + end_token
    encoder_input = tf.expand_dims(inp_sentencce_id, 0)

    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(args.max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask,
                                                     dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0), attention_weights

def translate(sentence):
    result, attention_weight = evaluate(sentence)
    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
    return predicted_sentence  # 返回预测的句子


# 训练模式
if args.is_train == 1:
    print('begin train')
    for epoch in range(args.epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp->zh, tar->en
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            if batch % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        if (epoch + 1) % 3 == 0:
            ckpt_save_path = ckpt_manager.save()  # 保存检查点
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
# 测试模式
if args.is_test == 1:
    print('begin test')
    syss = []
    refs = []
    for (batch, (inp, tar)) in enumerate(test_dataset):
        for (zhsen, ensen) in zip(inp, tar):
            predict, _ = evaluate2(zhsen)
            sys = tokenizer_en.decode([i for i in predict if i < tokenizer_en.vocab_size])
            syss.append(sys)
            ref = tokenizer_en.decode([i for i in ensen if i < tokenizer_en.vocab_size])
            refs.append(ref)
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(syss, refs)  # 计算bleu分数
    print(bleu.score)
