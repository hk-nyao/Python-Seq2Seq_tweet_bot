# -*- coding:utf-8 -*-

import config
import tweepy
import time

from sanitize import sanitize_text

# APIにログインする為に必要なユーザー情報
auth = tweepy.OAuthHandler(config.CONSUMER_KEY, config.CONSUMER_SECRET)
auth.set_access_token(config.ACCESS_TOKEN, config.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


# ツイートを取得し、テキストファイルに保存する
def get_tweet():
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    with open(config.TWEETS_TXT, "a") as text_file:
        # ツイートを何ペア取得するか
        tweets = int(input("取得したいツイート数を入力\n<< "))
        while True:
            # APIの残り取得制限数を表示
            jsn = api.rate_limit_status()['resources']['application']['/application/rate_limit_status']

            print("API limits: {}".format(jsn))
            # API制限に引っかかっている場合は10秒ごとに再チェック
            while jsn["remaining"] == 0:
                time.sleep(10)
                jsn = api.rate_limit_status()['resources']['application']['/application/rate_limit_status']

            # クエリで@検索をかけたあと、リプライになっているツイート(status)を抽出
            # リツイート、bot、URL、稼ぎ目的等のツイートは除外
            result = api.search(q='@ -RT -bot -amp -副業 -応募 -プレゼント -抽選 -キャンペーン -source:twittbot.net -filter:links',
                                lang='ja', result_type='recent', count=100)
            for i, status in enumerate(result):
                # 誰かにリプライされていればリプライ先のツイートidが入り、なければnullが入る
                reply_id = status.in_reply_to_status_id
                # statusのツイートid
                orig_id = status.id
                if reply_id is not None:
                    # リプライ先のツイートを取得
                    try:
                        results = api.get_status(reply_id)
                        # ツイートをコーパスに整形
                        orig_text = sanitize_text(str(results.text))
                        reply_text = sanitize_text(str(status.text))
                        print("{}: {}".format(orig_id, orig_text))
                        print("{}: {}".format(reply_id, reply_text))
                        # 整形後、2文字以上のやりとりならテキストファイルに保存
                        if len(orig_text) >= 2 and len(reply_text) >= 2:
                            text_file.write(orig_text + "\n" + reply_text + "\n")
                            tweets -= 1
                            print("残りツイート数 = {}\n".format(tweets))
                            # API制限に引っかかるのをなるべく避けるため、0.5秒待つ
                            time.sleep(0.5)
                            # 指定したツイート数を取得したらプログラムを終了
                            if tweets == 0:
                                exit()
                    # ユーザーがprivateだとTweepErrorを投げるのでpassする
                    except tweepy.error.TweepError:
                        pass


# mainメソッド
if __name__ == '__main__':
    get_tweet()
