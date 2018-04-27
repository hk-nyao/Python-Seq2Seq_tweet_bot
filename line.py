import re
import config
# For sanitize dusts in each sentence(original)
import sanitize
import sys
import os

if __name__ == '__main__':
    print("ツイート数：{}".format(int(len(open(config.TWEETS_TXT).readlines())/2.0)))

