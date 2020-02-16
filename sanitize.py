# -*- coding:utf-8 -*-

import re
import sys

# config.py
import config

tweets = "{0}/tweets1M.txt".format(config.DATA_DIR)
output = "{0}/tweets100M.txt".format(config.DATA_DIR)


# 取得したツイート数の確認
def count_tweet():
    print("ツイート数：{}".format(int(len(open(config.TWEETS_TXT).readlines()) / 2.0)))


# 正規表現によるコーパス作成を行うメソッド
def sanitize_text(text):
    dust = "[^a-zA-Z0-9０−９一-龥ぁ-んァ-ン!！?？〜~ー…:;：；.・、。「」]"
    spacing = "\n | \s+ | \s+.\s+"
    single = "\s+.\s+"
    url = r"(https?|ftp)(:\/\/[-_\.!~*'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)"
    hashtag = r"[#|＃]%s"
    reply = "@[a-zA-Z(),_,0-9]+"
    # 顔文字など、カッコごと中身を消す正規表現
    bracket = "\[（(].+?\[)）]"
    # 同じ文字が4文字以上続くものにマッチ
    repeat = r"(.)\1{3,}"

    # 改行文字(\n)や複数のスペースを、半角スペースに置き換える
    if re.search(spacing, text) is not None:
        text = re.sub(spacing, ' ', text)
    # URL、ハッシュタグ、リプライidが存在した場合、消す
    if re.search(url, text) is not None:
        text = re.sub(url, '', text)
    if re.search(hashtag, text) is not None:
        text = re.sub(hashtag, '', text)
    if re.search(reply, text) is not None:
        text = re.sub(reply, '', text)
    # カッコを中身ごと消す
    if re.search(bracket, text) is not None:
        text = re.sub(bracket, ' ', text)
    # 日本語または?か!で文が終了し、それ以外に余計なものがついている場合、消す
    if re.search(dust, text) is not None:
        text = re.sub(dust, ' ', text)
    # 左右が空白かつ1文字のものを半角スペースに置き換える
    if re.search(single, text) is not None:
        text = re.sub(single, ' ', text)
    # 同じ文字が4つ以上続いた場合、全て3つの文字とする
    if re.search(repeat, text) is not None:
        text = re.sub(repeat, r"\1\1\1", text)

    return text.strip()


if __name__ == "__main__":
    with open(tweets, "r") as tweets:
        with open(output, "w") as output:
            line = tweets.readline()
            while line:
                output.write(sanitize_text(line) + "\n")
                line = tweets.readline()
