import sys
sys.path.insert(0, '/home/cc/pytweet/')
import pickle
import json
import pandas as pd
from src import *
import tweepy

## use a pandas df to store (screen_name, location , tweet_id) -- aggregate this data as I stream

def main(argv):
    tweet_list = []
    init = initialize.Initializer()
    init.initialize()
    stream = init.get_stream()
    global api
    api = init.api
    query = ["Donald Trump", "Global Warming", "Paris Agreement", "America"]
    max_tweets = 10000
    #tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]
    stream.filter(track=query)
    



if __name__ == "__main__":
    main(sys.argv)
