import tensorflow as tf
import tweepy
import time
import predict
import config

from datetime import datetime, timedelta
from threading import Thread
from queue import Queue

consumer_key = config.CONSUMER_KEY
consumer_secret = config.CONSUMER_SECRET
access_token = config.ACCESS_TOKEN
access_token_secret = config.ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


"""
yield(): returnと違い、関数を一時停止して、次に呼び出した時はyieldの次の行から始まる
         大量のデータの受け渡しを、メモリを消費せず少量ずつ行いたいときなどに有効
"""


# Streamingを使う理由は、Restと違って永続的にデータを取得できるから
# botとしてユーザーのツイートにすぐ反応する必要があるので、こちらを使用している
# その為にStreamListenerを継承したクラスを作成する
class MyListener(tweepy.StreamListener):
    # 初期化メソッド
    def __init__(self, api, queue):
        self.next_tweet_time = self.get_next_tweet_time()
        self.api = api
        self.queue = queue

    # ツイートを取得するためのメソッド
    def on_status(self, status):
        print("{0}: {1}".format(status.text, status.author.screen_name))

        # 自分のツイートは無視する
        if status.user.id == self.api.me().id:
            print("Ignored my tweet")
            return True
        # 他人が自身のツイートにリプライした時、queueにリプライを保存
        elif status.text.startswith('@'+self.api.me().screen_name):
            print("Saved mention")
            self.queue.put(status)
            return True

    @staticmethod
    def get_next_tweet_time():
        return datetime.today() + timedelta(hours=4)

    @staticmethod
    def on_error(status_code):
        print("Error: {}".format(status_code))
        return True


# APIを通じてユーザーからのリプライを取得し、queueに保存するためのThread
class StreamReceiverThread(Thread):
    def __init__(self, api, queue):
        super(StreamReceiverThread, self).__init__()
        self.daemon = True
        self.api = api
        self.queue = queue

    def run(self):
        listener = MyListener(self.api, self.queue)
        stream = tweepy.Stream(auth, listener)
        # タイムラインのツイートを取得し続ける
        while True:
            try:
                # userstreamは使えなくなったため、filterによるキーワード検索が必須
                # そのため、自分のユーザー名で検索をかけている
                stream.filter(track=['@'+self.api.me().screen_name])
            except Exception as e:
                print(e)
                print(e.__doc__)
                time.sleep(60)
                stream = tweepy.Stream(auth, listener)


# queueに保存されたリプライに返信するためのThread
class ProcessingThread(Thread):
    def __init__(self, queue):
        super(ProcessingThread, self).__init__()
        self.daemon = True
        self.queue = queue

    # queueに保存されたリプライを非同期に取得し、seq2seqで返信を生成
    def run(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(config=tf_config) as sess:
            # predictor = 学習済みのseq2seq
            predictor = predict.EasyPredictor(sess)
            while True:
                reply = self.queue.get()
                make_reply_text(reply, predictor)


# リプライを投稿するメソッド
def post_reply(reply_body, screen_name, status_id):
    # リプライの文章
    reply_body = reply_body.replace('_UNK', '😄')
    reply_text = "@" + screen_name + " " + reply_body
    print("Reply:{0}".format(reply_text))
    # リプライを投稿
    api.update_status(status=reply_text, in_reply_to_status_id=status_id)


# GPUを使用して、seq2seqのモデルから返答を予測し、リプライとして返すメソッド
def make_reply_text(tweet, predictor):

    print("Processing {0}...".format(tweet.text))
    screen_name = tweet.author.screen_name
    # 返答を予測
    replies = predictor.predict(tweet.text)
    if not replies:
        print("no reply")
    reply_body = replies[0]
    if reply_body is None:
        print("No reply predicted")
    else:
        try:
            # twitterに返答を投稿
            post_reply(reply_body, screen_name, tweet.id)
        except tweepy.TweepError as e:
            print(e)
            print(e.__doc__)


# リアルタイムにリプライに返信するメソッド
def start_streaming():
    # ツイートを保存するためのqueueを生成
    q = Queue()
    # ツイート取得用Threadとリプライ用Threadを生成
    p_thread = ProcessingThread(q)
    s_thread = StreamReceiverThread(api, q)
    p_thread.start()
    s_thread.start()
    print("start streaming")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    start_streaming()
