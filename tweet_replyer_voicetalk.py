#自身のツイートを取得し、そのテキストを機械音声で発話させる
#メソッドをまとめたファイル
#tweet_listener.pyと同時に実行することで、リアルタイムでTwitter botと会話できる

import os
import tensorflow as tf
import tweepy
import time
import predict
import sqlite3
import pickle
import tweet_listener
import config
import tweet_get
import subprocess
import types
import base64
import json
import requests


WAV_FILES = []

consumer_key = config.CONSUMER_KEY
consumer_secret = config.CONSUMER_SECRET
access_token = config.ACCESS_TOKEN
access_token_secret = config.ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


def select_next_tweet():
    #データベースオブジェクト(conn)を作り、データベースにアクセスする
    conn = sqlite3.connect(tweet_listener.DB_NAME)
    #cursorオブジェクト(c)を作ることで、SQLコマンドをこのオブジェクトから実行できるようにする
    c = conn.cursor()
    #sql文を実行。データベースに変更が生じる
    c.execute("select sid, data, bot_flag from tweets where processed = 0")
    for row in c:
        sid = row[0]
        data = pickle.loads(row[1])
        bot_flag = row[2]
        return sid, data, bot_flag
    return None, None, None

"""
yield():returnと違い、関数を一時停止して、
        次に呼び出した時はyieldの次の行から始まる
1GB の巨大なテキストファイルがあるとします。
このファイルを読み込み、データを渡してくれる関数を作るとします。
普通にやろうとすると、受け渡し用のメモリが 1GB になってしまいますが、
yield を使えば、少量、たとえば 1 行づつデータを読み込み、
その都度 yield すればいいので、メモリの使用量は僅かで済みます。
"""

def mark_tweet_processed(status_id):
    conn = sqlite3.connect(tweet_listener.DB_NAME)
    c = conn.cursor()
    c.execute("update tweets set processed = 1 where sid = ?", [status_id])
    conn.commit()
    conn.close()


def tweets():
    while True:
        status_id, tweet, bot_flag = select_next_tweet()
        if status_id is not None:
            yield(status_id, tweet, bot_flag)
        time.sleep(1)

#リプライを投稿するメソッド
def post_reply(api, bot_flag, reply_body, screen_name, status_id):
    #リプライにある未知語(unknown)の数
    unk_count = reply_body.count('_UNK')
    #リプライの文章
    reply_body = reply_body.replace('_UNK', '😄')
    #自動ツイートの場合
    if bot_flag == tweet_listener.SHOULD_TWEET:
        if unk_count > 0:
            return
        reply_text = reply_body
        print("My Tweet:{0}".format(reply_text))
        if not reply_text:
            return
        api.update_status(status=reply_text)
    #リプライの場合
    else:
        if not reply_body:
            reply_body = "😓(お返事が生成できませんでした)"
        reply_text = "@" + screen_name + " " + reply_body
        print("Reply:{0}".format(reply_text))
        api.update_status(status=reply_text,
                          in_reply_to_status_id=status_id)

#GPUを使用して、seq2seqのモデルから返答を予測し、リプライとして返すメソッド
def twitter_bot():
    # Only allocate part of the gpu memory when predicting.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=tf_config) as sess:
        predictor = predict.EasyPredictor(sess)

        for tweet in tweets():
            status_id, status, bot_flag = tweet
            print("Processing {0}...".format(status.text))
            screen_name = status.author.screen_name
            replies = predictor.predict(status.text)
            if not replies:
                print("no reply")
                continue
            reply_body = replies[0]
            #ツイート：機械音声（男）、リプライ：機械音声（声優）で発話させたい場合
            #tweet_get.open_jtalk(status.text,"man")
            #tweet_get.rospeex(replies[0])
            #talk(WAV_FILES)
            #WAV_FILES.clear()
            if reply_body is None:
                print("No reply predicted")
            else:
                try:
                    post_reply(api, bot_flag, reply_body, screen_name, status_id)
                except tweepy.TweepError as e:
                    # duplicate status
                    if e.api_code == 187:
                        pass
                    else:
                        raise
            mark_tweet_processed(status_id)


if __name__ == '__main__':
    twitter_bot()
