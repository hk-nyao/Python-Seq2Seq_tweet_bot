"""Sequence-to-sequence model with an attention mechanism."""


import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from  lib.data_utils import *
from lib.my_seq2seq import *

class Seq2SeqModel(object):
  """
  Attention Seq2Seqモデル（複数のバケツ）

  このクラスは、エンコーダは多層のRNN、デコーダはAttentionベースで実装されている。
  これは、以下の論文の提案モデルと同じである：
  http://arxiv.org/abs/1412.7449 -
  完全なモデルを実装するには、詳細またはSeq2Seqライブラリを参照のこと。
  このクラスはまた、LSTM Cellに加えて、GRU Cellを使用することができ、より大きな出力
  ボキャブラリサイズを処理するためにsoftmaxをサンプリングすることもできる。
  このモデルの単層版で、双方向エンコーダを備えたものは以下に記載がある：
  http://arxiv.org/abs/1409.0473
  サンプリングされたsoftmaxは次の論文の第３章で説明されている：
  http://arxiv.org/abs/1412.2007
  """

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=1024, forward_only=False, beam_search = True, beam_size=10, attention=True):
    """
    モデルの生成を行う。
    引数:
      source_vocab_size: 入力側のボキャブラリサイズ
      target_vocab_size: 出力側のボキャブラリサイズ
      buckets: エンコーダとデコーダの文章の最大文字数のタプルの配列。[Input,Output]
      e.g. buckets = [(5,10),(10,15),(15,20),...]
      size: モデルの各層のユニット数
      num_layers: モデルの層の数
      max_gradient_norm: グラディエントは、このノルムを最大限にクリップする
      gradients will be clipped to maximally this norm.
      batch_size: 学習中に使用されるバッチサイズ。
        モデル構築はbatch_sizeから独立しているので、これが例えばデコードのために
        都合がよい場合、初期化のあとに変更される。
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: 学習開始時の学習率。
      learning_rate_decay_factor: 過学習であると判断した場合、必要に応じてこの値ぶん
                                  learning_rateから減少させる。
      use_lstm: Trueの場合、デフォルトのGRU Cellの代わりにLSTM Cellを使用する。
      num_samples: sampled softmaxのサンプル数。
      forward_only: Trueの場合、モデルに逆方向のパスを構築しない。
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    #もしsampled softmaxを使用する場合は、output_projectionが必要になる。
    output_projection = None
    softmax_loss_function = None
    # Sampled softmaxは、ボキャブラリサイズよりも小さいサンプルをサンプリングする
    if num_samples > 0 and num_samples < self.target_vocab_size:
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.target_vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, labels, inputs, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss
    # RNN用の内部多層　Cellを作成する
    print('###### tf.get_variable_scope().reuse : {}'.format(tf.get_variable_scope().reuse))
    def gru_cell():
      #　contrib.rnn.core_rnn_cell　→　core_rnn_cell (1.7.0現在)
      return tf.contrib.rnn.GRUCell(size, reuse=tf.get_variable_scope().reuse)#tf.get_variable_scope().reuse
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(size, reuse=tf.get_variable_scope().reuse)#tf.get_variable_scope().reuse
    single_cell = gru_cell
    if use_lstm:
      single_cell = lstm_cell
    cell = single_cell()
    if num_layers > 1:
      cell_1 = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple=False)
      cell_2 = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)], state_is_tuple=False)

    # seq2seq関数: 入力の為のembedding layerとAttentionを使用する
    print('##### num_layers: {} #####'.format(num_layers))
    print('##### {} #####'.format(output_projection))
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
        if attention:
            print("Attention Model")
            return embedding_attention_seq2seq(
               encoder_inputs, decoder_inputs, cell_1, cell_2,
               num_encoder_symbols=source_vocab_size,
               num_decoder_symbols=target_vocab_size,
               embedding_size=size,
               output_projection=output_projection,
               feed_previous=do_decode,
               beam_search=beam_search,
               beam_size=beam_size )
        else:
            print("Simple Model")
            return embedding_rnn_seq2seq(
              encoder_inputs, decoder_inputs, cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=size,
              output_projection=output_projection,
              feed_previous=do_decode,
              beam_search=beam_search,
              beam_size=beam_size )


    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    # bucketの最後の要素が最大サイズである
    for i in xrange(buckets[-1][0]):
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # ターゲットは１つだけずらしたデコーダ入力である
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # 学習のoutputとloss
    if forward_only:
        if beam_search:
              self.losses = []
              self.outputs, self.beam_path, self.beam_symbol = decode_model_with_buckets(
                  self.encoder_inputs, self.decoder_inputs, targets,
                  self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                  softmax_loss_function=softmax_loss_function)
        else:
              # print self.decoder_inputs
              self.outputs, self.losses = model_with_buckets(
                  self.encoder_inputs, self.decoder_inputs, targets,
                  self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                  softmax_loss_function=softmax_loss_function)
              # output projectionを使用する場合は、デコードのための出力を投影する必要がある
              if output_projection is not None:
                    for b in xrange(len(buckets)):
                      self.outputs[b] = [
                          tf.matmul(output, output_projection[0]) + output_projection[1]
                          for output in self.outputs[b]
                      ]


    else:
      self.outputs, self.losses = model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    self.train_loss_summaries = []
    self.forward_only_loss_summaries = []
    for i in range(len(self.losses)):
        self.train_loss_summaries.append(tf.summary.scalar("train_loss_bucket_{}".format(i), self.losses[i]))
        self.forward_only_loss_summaries.append(tf.summary.scalar("forward_only_loss_bucket_{}".format(i),
                                                                  self.losses[i]))

    # モデルを学習するための Gradients と SGD の更新
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only, beam_search):
    """
    モデルの学習を実行するために、与えられた入力を供給する。
    引数:
      session: 使用するtensorflowのsession
      encoder_inputs: エンコーダの入力に使われるnumpyのintベクトルのリスト。
      decoder_inputs: デコーダの入力に使われるnumpyのintベクトルのリスト。
      target_weights: ターゲットの重みとして使われるnumpyのfloatベクトルのリスト。
      bucket_id: 使用するモデルのバケツ
      forward_only: 逆方向の構築を行うか。
    戻り値:
      3つの戻り値（gradient_norm, 平均perplexity, outputs）。
    Raises:
      ValueError: バケツサイズとencoder_inputs,decoder_inputs,target_weightsが
                  一致していないとき。
    """
    # バケツサイズとそれぞれのサイズが一致しているか
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id],  # Loss for this batch.
                     self.train_loss_summaries[bucket_id]]  # Summary op for Train Loss
    else:
        if beam_search:
            output_feed = [self.beam_path[bucket_id]]  # Loss for this batch.
            output_feed.append(self.beam_symbol[bucket_id])
        else:
            output_feed = [self.losses[bucket_id],
                           self.forward_only_loss_summaries[bucket_id]] # Summary op for forward only loss

        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])
    # print bucket_id
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[3], None  # Gradient norm, loss, no outputs.
    else:
      if beam_search:
          return outputs[0], outputs[1], outputs[2:]  # No gradient norm, loss, outputs.
      else:
          return None, outputs[0], outputs[1], outputs[2:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """
    指定されたバケツからランダムなバッチデータを取得し、学習するための準備を行う。
    step()でデータを供給するには、batch-majorのベクトルのリストでなければいけないが、
    ここでのデータはリストでないmajorケースを含む。
    したがって、このメソッドの主な動作は、データケースを適切なフォーマットで供給するために
    再インデックスすることである。
    引数:
      data: サイズがlen(self.buckets)のタプル。
            各要素には、batchの作成に使用する入力データと出力データのペアのリストが含まれている。
      bucket_id: batchを取得するためのバケツ。
    戻り値:
      3つの戻り値(encoder_inputs, decoder_inputs, target_weights)。
      これらはstep()を呼び出すための引数として使われる。
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # データからエンコーダとデコーダの入力をランダムに取得し、
    # 必要に応じてそれらをパディング(バケツサイズに対応した空白埋めを)し、エンコーダの出力を逆にして"GO"をデコーダに追加する。
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # エンコーダの入力はパディングされた後に反転される。
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # デコーダの入力には余計に"GO"シンボルがついており、それからパッドが埋め込まれる。
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # 今度は、上記で選択したデータからbatch-majorベクトルを作成する。
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # バッチエンコーダの入力は、エンコーダの入力を再インデックスするだけでよい
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # バッチエンコーダの入力は再インデックスした decoder_inputsであり、重みを作成する。
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # パディングするターゲットの　target_weightsを0にする
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # 対応するターゲットがPADシンボルの場合は、重みを0にする
        # 対応するターゲットは、decoder_inputが1だけ前方にシフトされている。
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
