import json
import sys
from mongo_funcs import read, write, delete
from rake_nltk import Rake
from collections import defaultdict
sys.path.append('/home/cc/pytweet/analysis/')
from infer import Infer

MAX_TWEETS=10


def get_text(tweet):
  return tweet['text']

def get_coords(tweet):
  if 'coordinates' in tweet and tweet['coordinates']:
    return tweet['coordinates']['coordinates']

def get_place(tweet):
  if 'place' in tweet and tweet['place']:
    return tweet['place']['country']


def get_user_loc(tweet):
  if 'user' in tweet and 'location' in tweet['user']:
    return tweet['user']['location']

def main(argv):

  infer = Infer(argv[0], 16, argv[1])

  cursor_news = read()
  corpus = ""
  id_to_index = {}
  news = []
  sources = {}
  for ind, d in enumerate(cursor_news):
    corpus += d["description"] + "\n" + d["title"]

    d['score'] = infer.infer_logreg(d['description']) + " " +  infer.infer_logreg(d['title'])   


    id_to_index[d["_id"]] = ind
    news.append(d)


  final_data = []
  for d in news:
    d.pop("_id")
    final_data.append(d)
  delete("source_ratings")
  write(final_data, "source_ratings")     
        
  print("wrote", len(final_data))




if __name__ == "__main__":
  main(sys.argv[1:])
