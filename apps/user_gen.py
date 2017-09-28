import sys
import pickle
import json
import pandas as pd
import src
import tweepy
from rake_nltk import Rake
from nltk import bigrams
import datetime
import analysis
## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def main(argv):

#  keywords = [ f for k in keywords for f in k.split()]
  infer = analysis.Infer(argv[3], 16, argv[4])
  tweet_list = []
  init = src.Initializer()
  init.initialize()
  stream = init.get_stream()
  global api
  api = init.api

  data = analysis.get_source_user(api, argv[6:], int(argv[0]), int(argv[1]), int(argv[2]))
  time_stamp = str(datetime.datetime.utcnow())
  result = []
  for d in data:
    sent = 2 * infer.infer_logreg(d[1])
    if sent >= 0:
      result.append({"source":d[0], "time": d[2], "text": d[1], "sentiment": sent , "frequency": sent,  "type":d[0]})

  new_result = analysis.smooth(result, 2, int(argv[5]))
#  delete("follow_ft")
  analysis.write(new_result, "follow_ft")
  analysis.write_smooth("follow_ft")     
  print("Wrote:", len(new_result))
  print("NON-SMOOTHED:", len(result))

if __name__ == "__main__":
   main(sys.argv[1:])




