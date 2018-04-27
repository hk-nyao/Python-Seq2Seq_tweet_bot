# -*- coding:utf-8 -*-

import re
import sys

# config.py
import config

tweets = "{0}/tweets1M.txt".format(config.DATA_DIR)
output = "{0}/tweets100M.txt".format(config.DATA_DIR)


#正規表現によるゴミの除去をおこなうメソッド
def sanitize_text(text):
    #[]内以外の文字にmatchする
    tab = "\n"
    url = r"(https?|ftp)(:\/\/[-_\.!~*'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)"
    hashtag = r"[#|＃]%s"
    reply = "@[a-zA-Z(),_,0-9]+"
    dust = "[^a-zA-Z0-9０−９一-龥ぁ-んァ-ン!！?？〜~ー…:;：；.・、。「」]"
    #（）内の文字ごと（）を消す正規表現
    kakko = "\(.+?\)"
    kakko2 = "\（.+?\）"
    space = "\s+"
    dust2 = "\s.\s"
    #last = ".$"

    #改行文字(\n)が存在した場合、消す
    if re.search(tab,text) is not None:
        #print("existed")
        text = re.sub(tab,'　',text)
    #URLが存在した場合、消す
    if re.search(url,text) is not None:
        #print("URL existed")
        text = re.sub(url,'',text)
    #ハッシュタグを消す
    if re.search(hashtag,text) is not None:
        text = re.sub(hashtag,'',text)
    #リプライidを消す
    if re.search(reply,text) is not None:
        text = re.sub(reply,'',text)
    #半角のカッコを中身ごと消す
    if re.search(kakko, text) is not None:
        text = re.sub(kakko, ' ', text)
    #全角のカッコを中身ごと消す
    if re.search(kakko2, text) is not None:
        text = re.sub(kakko2, ' ', text)
    #日本語または?か!で文が終了し、それ以外に余計なものがついている場合、消す
    if re.search(dust, text) is not None:
        text = re.sub(dust, ' ', text)
    #複数のスペースを一つの半角スペースに置き換える
    if re.search(space,text) is not None:
        text = re.sub(space, ' ', text)
    #左右が空白であり、1文字のものを消す
    while(re.search(dust2,text) is not None):
        text = re.sub(dust2, ' ', text)
    #末尾かつ左が空白であり、1文字のものを消す（おはよう　！なら「！」が消える）
    #if re.search(last,text) is not None:
    #    text = re.sub(last, '', text)

    return text.strip()

if __name__ == "__main__":
  with open(tweets,"r") as tweets:
      with open(output,"w") as output:
          line = tweets.readline()
          while(line):
              output.write(sanitize_text(line))
              output.write("\n")
              line = tweets.readline()
