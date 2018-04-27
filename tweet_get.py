# -*- coding:utf-8 -*-

# Twitter APIでツイートとリプライを取得し、TWEETS_TXTに保存する
# ツイートとリプライをしゃべらせることもできる

import tweepy
import re
import sys
import subprocess
import time
from datetime import datetime
import types
import base64
import json
import requests


# config.py
import config
# saniztize.py
import sanitize

# APIにログインする為に必要なユーザー情報
consumer_key = config.CONSUMER_KEY
consumer_secret = config.CONSUMER_SECRET
access_token = config.ACCESS_TOKEN
access_token_secret = config.ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

TWEETS_TXT = config.TWEETS_TXT
URL = "http://rospeex.nict.go.jp/nauth_json/jsServices/VoiceTraSS"
WAV_FILES = []


# ツイートを取得し、テキストファイルに保存する
def get_tweet():
    with open (TWEETS_TXT,"a") as text_file:
                    # ツイートを何ペア取得するか
                    tweets=int(input("取得したいツイート数を入力\n<<"))
                        # クエリで@検索をかけたあと、リプライになっているツイートを抽出
                        # リツイート、bot、URLを含むツイートは除外
                    while(True):
                        # APIの残り取得制限数を表示
                        print("API limits:{}".format(api.rate_limit_status()))
                        result = api.search(q='@ -RT -bot -amp -source:twittbot.net -filter:links' ,lang='ja', count=100)
                        for i, status in enumerate(result):
                            # リプライ元のツイートid
                            reply_id = status.in_reply_to_status_id
                            # リプライをしているユーザーid
                            orig_id = status.id
                            if reply_id is not None:
                                # リプライ元のツイートを取得
                                # ツイートが保護されていたらTweepErrorを投げられる
                                try:
                                    results = api.get_status(reply_id)
                                except tweepy.error.TweepError:
                                    None
                                else:
                                    # ツイートをコーパスに整形
                                    result_text = sanitize.sanitize_text(str(results.text))
                                    reply_text = sanitize.sanitize_text(str(status.text))
                                    print(result_text)
                                    print(reply_text)

                                    #2文字以上のやりとりならテキストファイルに保存
                                    if len(result_text)>=2:
                                        if len(reply_text)>=2:
                                            text_file.write(result_text)
                                            text_file.write("\n")
                                            text_file.write(reply_text)
                                            text_file.write("\n")
                                            tweets = tweets-1
                                            time.sleep(0.5)
                                            #open_jtalk(result_text,"man")
                                            #rospeex(reply_text)
                                            #talk(WAV_FILES)
                                            #WAV_FILES.clear()
                                        # APIの取得制限に引っかからないようにsleepする
                                        print(reply_id)
                                        print("残りツイート数 = {}\n".format(tweets))
                                        if(tweets==0):
                                            exit();


            # id確認
            # f.write('ユーザーid: ')
            # f.write(status.user.screen_name)
            # f.write('\nツイートid: ')
            # f.write(str(id))
            # f.write('\nリプライid: ')
            # f.write(str(replyid))
            # f.write('\nテキスト: ')
            # f.write(status.text)
            # f.write('\n\n')

def rospeex(text):
    databody = {"method": "speak",
                "params": ["1.1",
                           {"language": "ja", "text": text,
                            "voiceType": "F128", "audioType": "audio/x-wav"}]}
    response = requests.post(URL, data=json.dumps(databody))
    tmp = json.loads(response.text)
    wav = base64.decodestring(tmp["result"]["audio"].encode("utf-8"))
    with open("saiyu.wav", "wb") as f:
        f.write(wav)
    WAV_FILES.append("saiyu.wav")

def open_jtalk(text,voice):
    global wr
    wav = voice+".wav"
    if voice=="mei":
        htsvoice=['-m','/usr/share/hts-voice/mei/mei_normal.htsvoice']
    elif voice=="miku":
        htsvoice=['-m','/usr/share/hts-voice/miku/miku.htsvoice']
    elif voice=="man":
        htsvoice=['-m','/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice']
    open_jtalk=['open_jtalk']
    mech=['-x','/var/lib/mecab/dic/open-jtalk/naist-jdic']
    speed=['-r','1.3']
    outwav=['-ow',wav]
    cmd=open_jtalk+mech+htsvoice+speed+outwav
    c = subprocess.Popen(cmd,stdin=subprocess.PIPE)
    c.stdin.write(text.encode())
    c.stdin.close()
    c.wait()
    WAV_FILES.append(wav)
#    wr = subprocess.Popen(aplay)

def talk(wav_files):
    for i, wav in enumerate(wav_files):
        aplay = ['aplay',wav]
        subprocess.call(aplay)


# mainメソッド
if __name__=='__main__':
    get_tweet()
