from mongo_funcs import read
import sys
sys.path.insert(0, '/home/cc/pytweet/analysis')
sys.path.insert(0, '/home/cc/pytweet/')
import pickle
import json
import pandas as pd
from src import *
import tweepy
from rake_nltk import Rake
from nltk import bigrams
from processing import get_source_user, smooth
import datetime
from mongo_funcs import delete,write
from infer import Infer
## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def main(argv):

#  keywords = [ f for k in keywords for f in k.split()]
  infer = Infer(argv[3], 16, argv[4])
  tweet_list = []
  init = initialize.Initializer()
  init.initialize()
  stream = init.get_stream()
  global api
  api = init.api

  data = get_source_user(api, argv[6:], int(argv[0]), int(argv[1]), int(argv[2]))
  time_stamp = str(datetime.datetime.utcnow())
  result = []
  for d in data:
    sent = 2 * infer.infer_logreg(d[1])
    if sent >= 0:
      result.append({"source":d[0], "time": d[2], "text": d[1], "sentiment": sent })

  new_result = smooth(result, 2, int(argv[5]))
#  delete("follow_ft")
  write(new_result, "follow")     
  print("Wrote:", len(new_result))
  print("NON-SMOOTHED:", len(result))

if __name__ == "__main__":
   main(sys.argv[1:])




