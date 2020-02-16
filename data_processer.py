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

"""
    各テキストファイルは、存在してもしなくても自動的に上書きする仕様になっています

    ○tweet_get.pyによって作成

    (A) data/tweets1M.txt
        "ツイート"と"リプライ"のペアが入っているファイル (以降、元データと称する)
        奇数行にはツイート、偶数行にはリプライが入り、２行で１ペアとなっている
        例)
            1行目: おはよー！今日から新学期だね
            2行目: 今日も一日がんばるぞい！


    ○data_processer.pyによって作成
    
    (B) chatbot_generated/tweets_enc.txt
        元データから"ツイート"だけを集めて保存したファイル

    (C) chatbot_generated/tweets_dec.txt
        元データから"リプライ"だけを集めて保存したファイル

    (D) chatbot_generated/tweets_train_[enc|dec].txt
        元データから、学習用に一定割合のデータを保存したファイル

    (E) chatbot_generated/tweets_val_[enc|dec].txt
        元データから、テスト用に一定割合のデータを保存したファイル

    (F) chatbot_generated/vocab_enc.txt
        (B) に出現したボキャブラリ(単語)を行毎に保存したファイル
        出現回数の降順に並べている

    (G) chatbot_generated/vocab_dec.txt
        (C) に出現したボキャブラリを行毎に保存したファイル
        出現回数の降順に並べている

    (H) chatbot_generated/tweets_[train|val]_[dec|enc]_idx.txt
        (D)(E)のデータから、文章中の単語をid化したものを保存したファイル
"""

texts = ["tweets_enc", "tweets_dec",
         "tweets_train_enc", "tweets_train_dec",
         "tweets_val_enc", "tweets_val_dec",
         "tweets_train_enc_idx", "tweets_train_dec_idx",
         "tweets_val_enc_idx", "tweets_val_dec_idx",
         "vocab_enc", "vocab_dec"]

DIGIT_RE = re.compile(r"\d")

# padding(バケツサイズより文字数が少ない場合の単語埋め)
_PAD = "_PAD"
# 文の始まり
_GO = "_GO"
# 文の終わり
_EOS = "_EOS"
# 未知語
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

tagger = MeCab.Tagger("-Owakati")


# MeCabによって、文章を分かち書きする関数
def japanese_tokenizer(sentence):
    # MeCabはstr(UTF-8)は受け入れるがbinaryは受け入れないので、確認する
    assert type(sentence) is str
    result = tagger.parse(sentence)
    return result.split()


# ツイートとリプライのペアが入ったファイルを読み込み、
# それぞれ(B)と(C)に分けて保存する関数
def split_tweets_replies(tweets_path, enc_path, dec_path):
    """
    Args:
      tweets_path: 元データが保存されているPath
      enc_path:    (B)を保存するPath
      dec_path:    (C)を保存するPath
    Returns:
      None
    """
    i = 1
    with gfile.GFile(tweets_path, mode="rb") as f, \
            gfile.GFile(enc_path, mode="w+") as ef, gfile.GFile(dec_path, mode="w+") as df:

        for line in f:
            if not isinstance(line, str):
                line = line.decode('utf-8')
            line = sanitize.sanitize_text(line)

            # 奇数行はツイート
            if i % 2 == 1:
                ef.write(line + "\n")
            # 偶数行はリプライ
            else:
                df.write(line + "\n")
            i += 1


# ファイルの行数を返す関数
def num_lines(file):
    return sum(1 for _ in open(file))


# 元データを学習用とテスト用、すなわち(D)と(E)に分ける関数
def create_train_validation(source_path, train_path, validation_path, train_ratio=0.75):
    """
    Args:
      source_path:     元データが保存されているPath
      train_path:      (D)を保存するPath
      validation_path: (E)を保存するPath
      train_ratio:     学習用に保存するデータの割合
    Returns:
      None
    """
    nb_lines = num_lines(source_path)
    nb_train = int(nb_lines * train_ratio)
    counter = 0
    with gfile.GFile(source_path, "r") as f, \
            gfile.GFile(train_path, "w") as tf, gfile.GFile(validation_path, "w") as vf:

        for line in f:
            if counter < nb_train:
                tf.write(line)
            else:
                vf.write(line)
            counter = counter + 1


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
# 文を分かち書きし、ボキャブラリファイルによって単語をid化する関数
def sentence_to_token_ids(sentence, vocabulary, tokenizer=japanese_tokenizer, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    # else:
    #    words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # 「2020年」などの文字の数字部分を、全て"0"に正規化してからid化
    return [vocabulary.get(re.sub(DIGIT_RE, "0", w), UNK_ID) for w in words]
    # return [vocabulary.get(w, UNK_ID) for w in words]  # added  by Ken


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
# 学習用およびテスト用データ(D)~(E)を、Seq2Seqに入力できるよう
# 文を分かち書きし、ボキャブラリファイルによって単語をid化する関数
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=japanese_tokenizer, normalize_digits=True):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
        with gfile.GFile(target_path, mode="wb") as tokens_file:  # edit w to wb
            counter = 0
            for line in data_file:
                # line = tf.compat.as_bytes(line)  # added by Ken
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                # line is binary here
                line = line.decode('utf-8')
                token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
# 単語ファイルを読み込み、単語だけのリストと(単語、ID)の辞書を返す関数
def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        # (単語, ID) の辞書
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


# From https://github.com/1228337123/tensorflow-seq2seq-chatbot
# ツイート(B)とリプライ(C)のみが入ったファイルを読み込み、
# それぞれの単語ファイル、すなわち(F)(G)を作り保存する関数
def create_vocabulary(source_path, vocabulary_path, max_vocabulary_size, tokenizer=japanese_tokenizer):
    """
    Args:
      source_path:         (B)と(C)が保存されているPath
      vocabulary_path:     (F)と(G)を保存するPath
      max_vocabulary_size: 単語サイズ(何種類まで単語を保存するか)
      tokenizer:           文を単語に分けるためのトークナイザー
    Returns:
      None
    """
    with gfile.GFile(source_path, mode="r") as f:
        counter = 0
        vocab = {}  # (単語, 出現数)
        for line in f:
            counter += 1
            words = tokenizer(line)
            if counter % 5000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            for word in words:
                # 「2020年」などの文字の数字部分を、全て"0"に正規化
                # 必要かどうかは分からないとしている
                word = re.sub(DIGIT_RE, "0", word)
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        # 固定ラベル(START_VOCAB)と頻度順のボキャブラリを連結し、ファイルに書き出す
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")
        print("\n")


if __name__ == '__main__':

    # 各ファイルがなければ自動的に生成する
    if not os.path.exists(config.GENERATED_DIR):
        os.makedirs(config.GENERATED_DIR)
    os.chdir(config.GENERATED_DIR)
    path = os.getcwd()
    for i, text in enumerate(texts):
        if not os.path.isfile(text):
            f = open(text+".txt", 'w')
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
