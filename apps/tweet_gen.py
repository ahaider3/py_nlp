from mongo_funcs import read
import sys
sys.path.insert(0, '/home/cc/pytweet/')
import pickle
import json
import pandas as pd
from src import *
import tweepy
from rake_nltk import Rake
from nltk import bigrams
## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def main():
  data = read()
  res = []
  corpus = ""
  for d in data:
    corpus += d["description"] + "\n" + d["title"]
  r = Rake()
  r.extract_keywords_from_text(corpus)
  keywords = r.get_ranked_phrases()[:10]
  keywords = [ ' '.join(big) for k in keywords for big in list(bigrams(k.split()))]

#  keywords = [ f for k in keywords for f in k.split()]
  tweet_list = []
  init = initialize.Initializer()
  init.initialize()
  stream = init.get_stream()
  global api
  api = init.api
  query = keywords
  max_tweets = 10000
  stream.filter(track=query)
    



if __name__ == "__main__":
   main()




