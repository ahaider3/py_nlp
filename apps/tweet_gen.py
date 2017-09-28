import sys
sys.path.append("/home/cc/aws/py_nlp")
import pickle
import json
import pandas as pd
import tweepy
from src.initializer.initialize import *
#import src
import analysis
from rake_nltk import Rake
from nltk import bigrams
## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def main():
  data = analysis.read()
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
  init = Initializer()
  init.initialize()
  stream = init.get_stream()
  global api
  api = init.api
  query = keywords
  max_tweets = 10000
  min_range = min(100, len(query))
  stream.filter(track=query[:min_range])
    



if __name__ == "__main__":
   main()




