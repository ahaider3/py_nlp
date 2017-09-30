import json
import sys


from rake_nltk import Rake
from collections import defaultdict
from nltk import bigrams
MAX_TWEETS=1000
import pickle
import analysis
import src

argv = sys.argv[1:]

infer = analysis.Infer(argv[1], 16, argv[3])

m = pickle.load(open(argv[4]))
m_inv = {v[0]:k for k, v in m.items()}
init = src.Initializer()
init.initialize()
stream = init.get_stream()
global api
api = init.api



infer_tag = analysis.InferTimeline(argv[1], 16, argv[2],m_inv, api)

f = open(argv[0], "r")
raw = f.readlines()
data = []
print('DONE')

for r in raw:
  try:
    data.append(json.loads(r))
  except:
    pass
print(len(data))
cursor_news = analysis.read()
corpus = ""
id_to_index = {}
news = []
for ind, d in enumerate(cursor_news):
  corpus += d["description"] + "\n" + d["title"]
  d["tweets"] = list()
  d['location'] = list()
  d['sentiments'] = list()
  d['tagsource'] = list()
  id_to_index[d["_id"]] = ind
  news.append(d)


r = Rake()
r.extract_keywords_from_text(corpus)
keywords = r.get_ranked_phrases()[:10]
keywords = [ ' '.join(big) for k in keywords for big in list(bigrams(k.split()))]
print('DONE')

keyword_to_id = defaultdict(list)
for keyword in keywords:
  for d in news:
    if keyword in d["description"] or keyword in d["title"]:
      keyword_to_id[keyword].append(d["_id"])
num_joined = 0
print('DONE')
sets = [set() for i in range(len(news))]
assert(len(data) > 0)
print(len(data))
for d in [d_ for d_ in data if "text" in d_]:
  for keyword in keywords:
    if keyword in d['text']:
      ids = keyword_to_id[keyword]
      for id_ in ids:
        ind = id_to_index[id_]
        text = analysis.get_text(d)
        if text not in sets[ind] and len(news[ind]['tweets']) < MAX_TWEETS:
          sets[ind].add(text)
          news[ind]['tweets'].append(text)
          news[ind]['sentiments'].append(int(round( 2 * infer.infer_logreg(text))))

#          news[ind]['location'].append(get_user_loc(d))
          news[ind]['tagsource'].append(infer_tag.infer(d))

for ind in range(len(news)):
  d = defaultdict(int)
#  var = {'cnn':0, 'FoxNews':0, 'BreitbartNews': 0}
#  var_freq = {'cnn':0, 'FoxNews':0, 'BreitbartNews': 0}
  var = {k:0 for k,v in m.items()}
  var_freq = {k:0 for k in m}
  for idx, tag in enumerate(news[ind]['tagsource']):
    if tag is not None:
      var[tag] += news[ind]['sentiments'][idx]
      var_freq[tag] += 1

  score = max(var.items(), lambda k: k[1])[1]
  common = sum([j[1] for j in var_freq.items()])

  news[ind]['score'] = score
  news[ind]['type'] = common
  print("COMMON:", common)

  
final_data = []
for d in news:
  d.pop("_id")
  final_data.append(d)
#analysis.delete("tweets")
analysis.write(final_data, "tweets")     

