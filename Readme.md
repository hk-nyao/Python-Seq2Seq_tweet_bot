
# 目次
* はじめに
* 使ってみたい場合
* 各プログラムの役割
* デモ
* 動作環境


# はじめに
 * Twitterの規約が更新され、API Keyの取得方法が変わったため、本プログラムによるデータ取得は困難になっています。 *

## コードの参照元
本プログラムは以下のサイトを参考に製作しました.<br><https://github.com/higepon/tensorflow_seq2seq_chatbot><br>
参考元からの主な変更点は以下の通りです.<br>
1. tensorflow 1.7.0で動作するよう、一部のメソッド参照先の書き換えを行なった
2. ツイートの取得プログラムを自作し、より細かい文章のサニタイズを行なった
3. 機械音声ライブラリのOpen_jtalkやRospeexを利用し、文章だけでなく機械音声による発話をできるようにした
4. いくつかの改善点（ファイルが存在しないときに自動で生成しない）を修正した

## 注意点
自作プログラムの完成度は低く、データの取得やサニタイズ処理には改良の余地があります.<br>また、Tensorflowはバージョンアップのサイクルが早く、過去のバージョンで使用できた関数の参照場所が変わっていて使用できなくなった、といった問題が発生しやすいです.


# 使ってみたい場合
## 事前準備
* 電話番号が紐付けされたTwitter アカウントから、Twitter API Keyを取得する.

## 学習データの取得〜コーパス生成
* tweet_get.pyを実行し、コンソールに取得したいデータ数（ツイート数）を入力する.<br>取得したツイートは"/data/tweets1M.txt"に保存される.

* data_processer.pyを実行する.<br>tweets1M.txtをもとに、学習に必要なテキストファイルが"/chatbot_generated"に生成される.

## モデルの学習
* train.pyを実行する.<br>学習のチェックポイントは"/chatbot_generated"に保存される.

## モデルの実行
* tweet_listener.pyおよびtweet_replyer_voicetalk.pyを実行する.<br>APIを取得したアカウントにリプライを送ると、学習モデルによって生成された文章がリプライとして送られてくる.

# 各プログラムの説明
* <b>config.py:</b><br>ファイルの参照パスやAPIキーの設定を保存する

* <b>tweet_get.py:</b><br>Twitter APIを通して、Twitterに流れるツイートとそのリプライを"/data/tweets1M.txt"に保存する

* <b>line.py:</b><br>tweets1M.txtのツイート数を表示する

* <b>data_processer.py:</b><br>tweets1M.txtから、学習に使用するテキストファイルを生成し"/chatbot_generated"に保存する

* <b>/lib内のpyファイル:</b><br>Seq2Seqモデルの設定を保存する

* <b>train.py</b>:<br>/chatbot_generatedの各テキストファイルを読み込みSeq2Seqモデルの学習を行う.<br>学習は50stepsずつ行われ、チェックポイントは/chatbot_generatedに保存される
　　　　　　　　
* <b>predict.py:</b><br>生成されたSeq2Seqモデルを用いて会話が行える

* <b>tweet_listener.py:</b><br>APIを取得したユーザーのtimelineのツイートとリプライを取得し、データベースに保存する

* <b>tweet_replyer_voicetalk.py:</b><br>データベースにリプライが保存されている場合、Seq2Seqモデルによって返答を生成し、ユーザーにリプライを送る.<br>リプライとその返答を機械音声で発話させることもできる

# デモ
準備中

# 動作環境
  * Ubuntu 18.04 LTS
  * Python 3.6.5
  * Tensorflow 1.7.0
　
