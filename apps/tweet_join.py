import json
import sys
from mongo_funcs import read, write, delete
from rake_nltk import Rake
from collections import defaultdict
sys.path.append('/home/cc/pytweet/analysis/')
from infer import Infer
from nltk import bigrams
MAX_TWEETS=1000


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

  infer = Infer(argv[1], 16, argv[2])

  f = open(argv[0], "r")
  raw = f.readlines()
  data = []
  for r in raw:
    try:
      data.append(json.loads(r))
    except:
      pass
  print(len(data))
  cursor_news = read()
  corpus = ""
  id_to_index = {}
  news = []
  for ind, d in enumerate(cursor_news):
    corpus += d["description"] + "\n" + d["title"]
    d["tweets"] = list()
    d['location'] = list()
    d['sentiments'] = list()

    id_to_index[d["_id"]] = ind
    news.append(d)


  r = Rake()
  r.extract_keywords_from_text(corpus)
  keywords = r.get_ranked_phrases()[:10]
  keywords = [ ' '.join(big) for k in keywords for big in list(bigrams(k.split()))]

  keyword_to_id = defaultdict(list)
  # don't split keywords
#  keywords = [ f for k in keywords for f in k.split()]
  for keyword in keywords:
    for d in news:
      if keyword in d["description"] or keyword in d["title"]:
        keyword_to_id[keyword].append(d["_id"])
  num_joined = 0
  for d in [d_ for d_ in data if "text" in d_]:
    for keyword in keywords:
      if keyword in d['text']:
        ids = keyword_to_id[keyword]
        for id_ in ids:
          ind = id_to_index[id_]
          if len(news[ind]['tweets']) < MAX_TWEETS:
            text = get_text(d)
            news[ind]['tweets'].append(text)
            news[ind]['sentiments'].append(int(round( 2 * infer.infer_logreg(text))))
            news[ind]['location'].append(get_user_loc(d))

            num_joined += 1
  final_data = []
  for d in news:
    d.pop("_id")
    final_data.append(d)
  delete("tweets")
  write(final_data, "tweets")     
        
  print("wrote", len(final_data), "joined", num_joined)




if __name__ == "__main__":
  main(sys.argv[1:])
