import config
import os
import sys
import math
import numpy as np
import tensorflow as tf
import data_processer
import lib.seq2seq_model as seq2seq_model


def show_progress(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def read_data_into_buckets(enc_path, dec_path, buckets):
    """
    ツイートとリプライを読み込んでその長さに応じたバケツに入れる
    Args:
      enc_path: ツイートインデックスのpath
      dec_path: リプライインデックスのpath
      buckets:  バケツのリスト（[5,10],[10,15],[20,25]..など）
    Returns:
      data_set: data_set[i]はbuckets[i]の[tweet, reply]と中身は同じ
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(enc_path, mode="r") as enc_file, tf.gfile.GFile(dec_path, mode="r") as dec_file:
        tweet = enc_file.readline()
        reply = dec_file.readline()
        counter = 0
        while tweet and reply:
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in tweet.split()]
            target_ids = [int(x) for x in reply.split()]
            target_ids.append(data_processer.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                # ツイートとリプライの長さに基づき、ペアが入るバケツを見つける
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            tweet = enc_file.readline()
            reply = dec_file.readline()
    for bucket_id in range(len(buckets)):
        print("{}={}=".format(buckets[bucket_id], len(data_set[bucket_id])))
    return data_set


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
# 学習させるSeq2Seqモデルを作って保存する
def create_or_restore_model(session, buckets, forward_only, beam_search, beam_size):

    """Create model and initialize or load parameters"""

    model = seq2seq_model.Seq2SeqModel(source_vocab_size=config.MAX_ENC_VOCABULARY,
                                       target_vocab_size=config.MAX_DEC_VOCABULARY,
                                       buckets=buckets,
                                       size=config.LAYER_SIZE,
                                       num_layers=config.NUM_LAYERS,
                                       max_gradient_norm=config.MAX_GRADIENT_NORM,
                                       batch_size=config.BATCH_SIZE,
                                       learning_rate=config.LEARNING_RATE,
                                       learning_rate_decay_factor=config.LEARNING_RATE_DECAY_FACTOR,
                                       beam_search=beam_search,
                                       attention=True,
                                       forward_only=forward_only,
                                       beam_size=beam_size)

    print("model initialized")
    ckpt = tf.train.get_checkpoint_state(config.GENERATED_DIR)
    # the checkpoint filename has changed in recent versions of tensorflow
    checkpoint_suffix = ".index"
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def next_random_bucket_id(buckets_scale):
    print("output buckets scale:{}".format(buckets_scale))
    # 0以上1未満の一様乱数を生成
    n = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > n])
    return bucket_id


# 学習させる
def train():
    # GPUで動かすときはこのコードを使用する
    # per_process_gpu_memory_fraction：使用するメモリの最大値を指定する引数
    # 1を100%として、0.666は66%のメモリを使用する
    # BFC = Best-Fit with Coalescing
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allocator_type = 'BFC'

    # with tf.Session(config=tf_config) as sess:
    with tf.Session() as sess:

        # 学習用とテスト用データをバケツに入れる
        show_progress("Setting up data set for each buckets...")
        train_set = read_data_into_buckets(config.TWEETS_TRAIN_ENC_IDX_TXT, config.TWEETS_TRAIN_DEC_IDX_TXT, config.buckets)
        valid_set = read_data_into_buckets(config.TWEETS_VAL_ENC_IDX_TXT, config.TWEETS_VAL_DEC_IDX_TXT, config.buckets)
        show_progress("done\n")

        # Seq2Seqを生成。学習時はbeam searchはOFF
        show_progress("Creating model...")
        beam_search = False
        model = create_or_restore_model(sess, config.buckets, forward_only=False, beam_search=beam_search, beam_size=config.beam_size)
        show_progress("done\n")

        # list of # of data in ith bucket
        train_bucket_sizes = [len(train_set[b]) for b in range(len(config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
        # This is for choosing randomly bucket based on distribution
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        show_progress("before train loop")
        # Train Loop
        steps = 0
        previous_perplexities = []
        writer = tf.summary.FileWriter(config.LOGS_DIR, sess.graph)

        while True:
            bucket_id = next_random_bucket_id(train_buckets_scale)
            # print(bucket_id)

            # Get batch
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            # show_progress("Training bucket_id={0}...".format(bucket_id))

            # Train!
            #  _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
            #                                        forward_only=False, beam_search=beam_search)
            _, average_perplexity, summary, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                           bucket_id,
                                                           forward_only=False,
                                                           beam_search=beam_search)

            # show_progress("done {0}\n".format(average_perplexity))

            steps = steps + 1
            if steps % 2 == 0:
                writer.add_summary(summary, steps)
                show_progress(".")
            if steps % 50 != 0:
                continue

            # check point
            checkpoint_path = os.path.join(config.GENERATED_DIR, "seq2seq.ckpt")
            show_progress("Saving checkpoint...")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            show_progress("done\n")

            perplexity = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
            print("global step %d learning rate %.4f perplexity %.2f"
                  % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_perplexities) > 2 and perplexity > max(previous_perplexities[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_perplexities.append(perplexity)

            for bucket_id in range(len(config.buckets)):
                if len(valid_set[bucket_id]) == 0:
                    print("  eval: empty bucket %d" % bucket_id)
                    continue
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(valid_set, bucket_id)
                # _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                # bucket_id, True, beam_search=beam_search)
                _, average_perplexity, valid_summary, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True, beam_search=beam_search)
                writer.add_summary(valid_summary, steps)
                eval_ppx = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
                print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


if __name__ == '__main__':
    train()
