import re
import tensorflow as tf
import config
# For Japanese tokenizer
import MeCab
# For sanitize dusts in each sentence(original)
import sanitize
import sys
import os
from tensorflow.python.platform import gfile

#各テキストファイルは、存在してもしなくても自動的に上書きする仕様になっています

# The data format
#
# (A) data/tweets1M.txt
#  tweet_get.pyによって作られる、ツイートとリプライのペアが入っているファイル
#  奇数行にはツイート、偶数行にはリプライを入れ、２行で１ペアとして保存する
#  例)
#   1行目: おはよー！今日から新学期だね
#   2行目: 今日も一日がんばるぞい！
#
# 以下は、data_processer.pyの実行によって生成されるファイル
#
# (B) chatbot_generated/tweets_enc.txt
#  tweets1M.txtからツイートだけを集めて保存したファイル
#
# (C) chatbot_generated/tweets_dec.txt
#  tweets1M.txtからリプライだけを集めて保存したファイル
#
# (D) chatbot_generated/tweets_train_[enc|dec].txt
#  学習データとして割り当てられた、ツイートとリプライのペアが入っているファイル
#
# (E) chatbot_generated/tweets_val_[enc|dec].txt
#  テストデータとして割り当てられた、ツイートとリプライのペアが入っているファイル
#
# (F) chatbot_generated/vocab_enc.txt
#  tweets_encに出現したボキャブラリを行毎に保存したファイル
#  出現回数の降順に並べている
#
# (G) chatbot_generated/vocab_dec.txt
#  tweets_decに出現したボキャブラリを行毎に保存したファイル
#  出現回数の降順に並べている
#
# (H) chatbot_generated/tweets_[train|val]_[dec|enc]_idx.txt
#  tweets_[train|val]_[enc|dec].txtの文章から、
#  文章中の単語をid化したものを保存したファイル
#

texts = ["tweets_enc","tweets_dec",
         "tweets_train_enc", "tweets_train_dec",
         "tweets_val_enc", "tweets_val_dec",
         "tweets_train_enc_idx", "tweets_train_dec_idx",
         "tweets_val_enc_idx", "tweets_val_dec_idx",
         "vocab_enc", "vocab_dec"]

DIGIT_RE = re.compile(r"\d")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

tagger = MeCab.Tagger("-Owakati")


def japanese_tokenizer(sentence):
    assert type(sentence) is str
    # Mecab doesn't accept binary, but Python string (utf-8).
    result = tagger.parse(sentence)
    return result.split()

def split_tweets_replies(tweets_path, enc_path, dec_path):
    """Read data from tweets_paths and split it to tweets and replies.
    Args:
      tweets_path: original tweets data
      enc_path: path to write tweets
      dec_path: path to write replies
    Returns:
      None
    """
    i = 1
    with gfile.GFile(tweets_path, mode="rb") as f, gfile.GFile(enc_path, mode="w+") as ef, gfile.GFile(dec_path,
                                                                                                       mode="w+") as df:
        for line in f:
            if not isinstance(line, str):
                line = line.decode('utf-8')
            line = sanitize.sanitize_text(line)

            # Odd lines are tweets
            if i % 2 == 1:
                ef.write(line)
                ef.write("\n")
            # Even lines are replies
            else:
                df.write(line)
                df.write("\n")
            i = i + 1

def num_lines(file):
    """Return # of lines in file
    Args:
      file: Target file.
    Returns:
      # of lines in file
    """
    return sum(1 for _ in open(file))


def create_train_validation(source_path, train_path, validation_path, train_ratio=0.75):
    """Split source file into train and validation data
    Args:
      source_path: source file path
      train_path: Path to write train data
      validation_path: Path to write validatio data
      train_ratio: Train data ratio
    Returns:
      None
    """
    nb_lines = num_lines(source_path)
    nb_train = int(nb_lines * train_ratio)
    counter = 0
    with gfile.GFile(source_path, "r") as f, gfile.GFile(train_path, "w") as tf, gfile.GFile(validation_path,
                                                                                             "w") as vf:
        for line in f:
            if counter < nb_train:
                tf.write(line)
            else:
                vf.write(line)
            counter = counter + 1


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
def sentence_to_token_ids(sentence, vocabulary, tokenizer=japanese_tokenizer, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    # return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words] #mark added .decode by Ken
    return [vocabulary.get(w, UNK_ID) for w in words]  # added  by Ken


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=japanese_tokenizer, normalize_digits=True):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
        with gfile.GFile(target_path, mode="wb") as tokens_file:  # edit w to wb
            counter = 0
            for line in data_file:
#                line = tf.compat.as_bytes(line)  # added by Ken
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                # line is binary here
                line = line.decode('utf-8')
                token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                  normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        # Dictionary of (word, idx)
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# From https://github.com/1228337123/tensorflow-seq2seq-chatbot
def create_vocabulary(source_path, vocabulary_path, max_vocabulary_size, tokenizer=japanese_tokenizer):
    """Create vocabulary file. Please see comments in head for file format
    Args:
      source_path: source file path
      vocabulary_path: Path to write vocabulary
      max_vocabulary_size: Max vocabulary size
      tokenizer: tokenizer used for tokenize each lines
    Returns:
      None
    """
    with gfile.GFile(source_path, mode="r") as f:
        counter = 0
        vocab = {}  # (word, word_freq)
        for line in f:
            counter += 1
            words = tokenizer(line)
            if counter % 5000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            for word in words:
                # Normalize numbers. Not sure if it's necessary.
                word = re.sub(DIGIT_RE, "0", word)
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")
        print("\n")


if __name__ == '__main__':

    #各ファイルがなければ自動的に生成する
    os.chdir(config.GENERATED_DIR)
    path = os.getcwd()
    for i, text in enumerate(texts):
        if os.path.isfile(text) == False:
            f = open(text+".txt",'w')
            f.close()

    print("Splitting into tweets and replies...")
    print("ツイート数：{}".format(int(len(open(config.TWEETS_TXT).readlines())/2.0)))
    split_tweets_replies(config.TWEETS_TXT, config.TWEETS_ENC_TXT, config.TWEETS_DEC_TXT)
    print("Done")

    print("Splitting into train and validation data...")
    create_train_validation(config.TWEETS_ENC_TXT, config.TWEETS_TRAIN_ENC_TXT, config.TWEETS_VAL_ENC_TXT)
    create_train_validation(config.TWEETS_DEC_TXT, config.TWEETS_TRAIN_DEC_TXT, config.TWEETS_VAL_DEC_TXT)
    print("Done")

    print("Creating vocabulary files...")
    create_vocabulary(config.TWEETS_ENC_TXT, config.VOCAB_ENC_TXT, config.MAX_ENC_VOCABULARY)
    create_vocabulary(config.TWEETS_DEC_TXT, config.VOCAB_DEC_TXT, config.MAX_DEC_VOCABULARY)
    print("Done")

    print("Creating sentence idx files...")
    data_to_token_ids(config.TWEETS_TRAIN_ENC_TXT, config.TWEETS_TRAIN_ENC_IDX_TXT, config.VOCAB_ENC_TXT)
    data_to_token_ids(config.TWEETS_TRAIN_DEC_TXT, config.TWEETS_TRAIN_DEC_IDX_TXT, config.VOCAB_DEC_TXT)
    data_to_token_ids(config.TWEETS_VAL_ENC_TXT, config.TWEETS_VAL_ENC_IDX_TXT, config.VOCAB_ENC_TXT)
    data_to_token_ids(config.TWEETS_VAL_DEC_TXT, config.TWEETS_VAL_DEC_IDX_TXT, config.VOCAB_DEC_TXT)
    print("Done")
