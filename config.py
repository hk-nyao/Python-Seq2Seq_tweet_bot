import os
# from sys import platform

# APIの個人情報
CONSUMER_KEY = "xxxxxxxxxxxxxxxxxxxx"
CONSUMER_SECRET = "xxxxxxxxxxxxxxxxxxxx"
ACCESS_TOKEN = "xxxxxxxxxxxxxxxxxxxx"
ACCESS_TOKEN_SECRET = "xxxxxxxxxxxxxxxxxxxx"


GENERATED_DIR = os.getenv("HOME") + "/Seq2Seq_local/chatbot_generated"
LOGS_DIR = os.getenv("HOME") + "/Seq2Seq_local/chatbot_train_logs"
DATA_DIR = os.getenv("HOME") + "/Seq2Seq_local/data"


is_fast_build = False
beam_search = True
beam_size = 20

if is_fast_build:
    TWEETS_TXT = DATA_DIR + "/tweets.short.txt"

else:
    TWEETS_TXT = DATA_DIR + "/tweets1M.txt"

if is_fast_build:
    MAX_ENC_VOCABULARY = 5
    NUM_LAYERS = 2
    LAYER_SIZE = 2
    BATCH_SIZE = 2
    buckets = [(5, 10), (8, 13)]

else:
    MAX_ENC_VOCABULARY = 100000
    NUM_LAYERS = 3
    LAYER_SIZE = 1024
    BATCH_SIZE = 128
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 60)]

MAX_DEC_VOCABULARY = MAX_ENC_VOCABULARY

# 学習率
LEARNING_RATE = 0.5
# 学習率減衰係数
LEARNING_RATE_DECAY_FACTOR = 0.99
# 最大勾配ノルム
MAX_GRADIENT_NORM = 5.0

TWEETS_TXT = "{0}/tweets1M.txt".format(DATA_DIR)

TWEETS_ENC_TXT = "{0}/tweets_enc.txt".format(GENERATED_DIR)
TWEETS_DEC_TXT = "{0}/tweets_dec.txt".format(GENERATED_DIR)

TWEETS_TRAIN_ENC_TXT = "{0}/tweets_train_enc.txt".format(GENERATED_DIR)
TWEETS_TRAIN_DEC_TXT = "{0}/tweets_train_dec.txt".format(GENERATED_DIR)

TWEETS_VAL_ENC_TXT = "{0}/tweets_val_enc.txt".format(GENERATED_DIR)
TWEETS_VAL_DEC_TXT = "{0}/tweets_val_dec.txt".format(GENERATED_DIR)

TWEETS_TRAIN_ENC_IDX_TXT = "{0}/tweets_train_enc_idx.txt".format(GENERATED_DIR)
TWEETS_TRAIN_DEC_IDX_TXT = "{0}/tweets_train_dec_idx.txt".format(GENERATED_DIR)
TWEETS_VAL_ENC_IDX_TXT = "{0}/tweets_val_enc_idx.txt".format(GENERATED_DIR)
TWEETS_VAL_DEC_IDX_TXT = "{0}/tweets_val_dec_idx.txt".format(GENERATED_DIR)

VOCAB_ENC_TXT = "{0}/vocab_enc.txt".format(GENERATED_DIR)
VOCAB_DEC_TXT = "{0}/vocab_dec.txt".format(GENERATED_DIR)
