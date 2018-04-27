
# 目次
* はじめに
* このプログラムでできること
* テキストファイルの役割
* デモ
* 引き継ぐ場合
* 動作環境


# はじめに
  <コードの参照元>
  <https://github.com/1228337123/tensorflow-seq2seq-chatbot>
  を元に、tensorflow 1.7.0で動作するよう改変・追加を行ったものです。
  コードに関して不明な点があれば、元コードを参照すると良いかもしれません。

  <引き継ぎについて>
  プログラムの完成度は低く、特にサニタイズ処理やモデル構築には改良の余地があります。
  また、Tensorflowはバージョンアップのサイクルが早く、過去のバージョンで使用できた関数の参照場所が
  変わっていて使用できなくなった、といった問題が発生しやすいです。


# このプログラムでできること
  1. tweet_get.py - Twitter APIを通して、Twitterに流れるツイートとそのリプライを/data/tweets1M.txtに保存する。

  2. data_processer.py - tweets1M.txtから、学習に使用する複数のテキストファイルを生成し、/chatbot_generatedに保存する。

  3. train.py - /chatbot_generatedの各テキストファイルを読み込み、Seq2Seqモデルの学習を行う。学習中は50stepsずつ行い、チェックポイントは/chatbot_generatedに保存される。
　　　　　　　　
  4. predict.py - 生成されたSeq2Seqモデルを読み込み、コンソール上で会話が行える。

  5. tweet_listener.py - APIを使用しているユーザーの、Twitter上の（自分のものも含む）ツイートとリプライを取得し、データベースに保存する。

  6. tweet_replyer.py - データベースに他人からのリプライが保存されている場合、Seq2Seqモデルによって返答を生成し、当該ユーザーにリプライを送る。



# テキストファイルの役割
  1. tweets1M.txt -  Twitter上のツイートとリプライの文章を、学習に使用できる形にしたもの。
                  奇数行はツイート、偶数行はリプライとなっている。


  2. tweets_enc.txt - tweets1M.txtから、ツイート（奇数行）だけを取り出したもの。

  3. tweets_dec.txt - tweets1M.txtから、リプライ（偶数行）だけを取り出したもの。

  4. tweets_train_enc.txt - tweets_enc.txtを、学習データとテストデータに分けた時の学習データ。

  5. tweets_train_dec.txt - tweets_dec.txtを、学習データとテストデータに分けた時の学習データ。

  6. tweets_val_enc.txt - tweets_enc.txtを、学習データとテストデータに分けた時のテストデータ。

  7. tweets_val_dec.txt - tweets_dec.txtを、学習データとテストデータに分けた時のテストデータ。

  8. tweets_train_enc_idx.txt - tweets_train_enc.txtの文章をid化したもの。

  9. tweets_train_enc_idx.txt - tweets_train_dec.txtの文章をid化したもの。

  10. tweets_val_enc_idx.txt - tweets_val_enc.txtの文章をid化したもの。

  11. tweets_val_enc_idx.txt - tweets_val_dec.txtの文章をid化したもの。

  12. vocab_enc.txt - tweets_enc.txtに出現したボキャブラリ（単語）を1行に1つずつ並べたもの。

  13. vocab_enc.txt - tweets_dec.txtに出現したボキャブラリ（単語）を1行に1つずつ並べたもの。



# デモ

# 引き継ぐ場合

# 動作環境
  * Ubuntu 16.04 LTS
  * Python 3.6.4
  * Tensorflow 1.7.0
　
