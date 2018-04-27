#自身のbotがツイートを取得し、データベースに保存するための
#メソッドをまとめたファイル

import os
import sqlite3
import pickle
import tweepy
from datetime import datetime, timedelta
import config

#データベースを表す名前
DB_NAME = 'tweets.db'

SHOULD_TWEET = 1
"""
pickleは、pythonのオブジェクトをバイト列に変換してくれるライブラリである。
メリットは、データ構造(リストとかクラスのインスタンスとか)を外部に保存できる点である。
今回のプログラムでは、ツイートオブジェクトを外部に保存するために用いている
"""

#ツイートを保存するデータベースを作るための関数
def create_tables():
    #データベースオブジェクト(conn)を作り、データベースにアクセスする
    conn = sqlite3.connect(DB_NAME)
    #cursorオブジェクト(c)を作ることで、SQLコマンドをこのオブジェクトから実行できるようにする
    cur = conn.cursor()
    print("Start creating database...")
    #テーブルが存在しない場合は、新しく作る
    sql = 'create table if not exists tweets(sid integer primary key, data blob not null, processed integer not null default 0, bot_flag integer not null default 0)'
    #sql文を実行。データベースに変更が生じる
    cur.execute(sql)
    #sql文を実行して生じた変更をセーブする
    conn.commit()
    #データベースを閉じる
    conn.close()
    print("Done")

#ツイートとそのIDを引数にとり、ツイートをバイト例に変換したものを
#データベースに保存する関数
def insert_tweet(status_id, tweet, bot_flag=0):
    #データベースオブジェクト(conn)を作り、データベースにアクセスする
    conn = sqlite3.connect(DB_NAME)
    #cursorオブジェクト(c)を作ることで、SQLコマンドをこのオブジェクトから実行できるようにする
    cur = conn.cursor()
    print("Start inserting tweet...")
    #ツイートをバイト列に変換したものを、binary_dataに返す
    #HIGHESTプロトコルは、データを早く変換するための定数
    binary_data = pickle.dumps(tweet, pickle.HIGHEST_PROTOCOL)
    #ツイートをバイトに変換したものをデータベースに保存する
    cur.execute("insert into tweets (sid, data, bot_flag) values (?, ?, ?)", [status_id, sqlite3.Binary(binary_data), bot_flag])
    conn.commit()
    conn.close()
    print("Done")

#Streamingを使う理由は、Restと違って永続的にデータを取得できるから
#botとしてユーザーのツイートにすぐ反応する必要があるので、こちらを使用している
#その為には元々のStreamListenerを継承しなければならず、クラスを作成している
class StreamListener(tweepy.StreamListener):
    #初期化メソッド
    def __init__(self, api):
        #継承するAPIに自分のAPIを設定
        self.api = api
        self.next_tweet_time = self.get_next_tweet_time()

    #ツイートを取得するためのメソッド
    def on_status(self, status):
        print("{0}: {1}".format(status.text, status.author.screen_name))

        screen_name = status.author.screen_name
        #自分のツイートは無視する
        if screen_name == self.api.me().screen_name:
            print("Ignored my tweet")
            return True
        #他人が自身のツイートにリプライした時
        elif status.text.startswith("@{0}".format(self.api.me().screen_name)):
            # Save mentions
            print("Saved mention")
            insert_tweet(status.id, status)
            return True
        else:
            if self.next_tweet_time < datetime.today():
                print("Saving normal tweet as seed")
                self.next_tweet_time = self.get_next_tweet_time()
                insert_tweet(status.id, status, bot_flag=SHOULD_TWEET)
            print("Ignored this tweet")
            return True

    @staticmethod
    def get_next_tweet_time():
        return datetime.today() + timedelta(hours=4)

    @staticmethod
    def on_error(status_code):
        print("Error occured")
        print(status_code)
        return True

################################################################################

#自身のツイッターアカウントをAPIに接続するためのメソッド
def tweet_listener():
    consumer_key = config.CONSUMER_KEY
    consumer_secret = config.CONSUMER_SECRET
    access_token = config.ACCESS_TOKEN
    access_token_secret = config.ACCESS_TOKEN_SECRET

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    #タイムラインのツイートを取得し続ける
    while True:
        try:
            stream = tweepy.Stream(auth=api.auth,
                                   listener=StreamListener(api))
            print("listener starting...")
            stream.userstream()
        except Exception as e:
            print(e)
            print(e.__doc__)

if __name__ == '__main__':
    create_tables()
    tweet_listener()
