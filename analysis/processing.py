import numpy as np
from nltk.tokenize import TweetTokenizer
from datetime import datetime
from collections import defaultdict

def pad(seq_len, vectors, num_feats):

  pad_vecs = np.zeros([len(vectors), seq_len, num_feats])
  for i, vec in enumerate(vectors):
    for ind, j in enumerate(range(0, len(vec), num_feats)):
      if ind == seq_len:
        break
      pad_vecs[i][ind] = vec[j:j+num_feats]	
  return pad_vecs



def read_subj(subj_path, obj_path):

  f_subj = open(subj_path, "r")
  f_obj = open(obj_path, "r")
  subj_lines = [ (1, l) for l in f_subj.readlines()]
  obj_lines = [ (0, l) for l in f_obj.readlines()]
  return subj_lines + obj_lines
  

    
def get_source_user(api, sources, num_headlines=10, branch_off=10, num_posts=10):

#  keywords = [ f for k in keywords for f in k.split()]

  statuses = [(src, api.user_timeline(screen_name=src, count=num_headlines, include_rts=False)) 
		for src in sources]
#    tweets = api.user_timeline(target, target,target)
  rts_id = [ (src, api.retweets(tweet.id, branch_off)) for src, status in statuses for tweet in status]
  flat_rt_id = [(src, tweet.author.screen_name) for src, tweets in rts_id for tweet in tweets]
#  with_users = [ (rt[1].author.screen_name, api.get_user(rt[1].author.screen_name, 
#                                                        rt[1].author.screen_name,
#                                                        rt[1].author.screen_name).location,
#                                        rt[0])
#  
#                                        for rt in flat_rt_id]
  # 1) add subj analysis 2) predict given string of tweets the news site a user most agrees with
  # follower and follower count to determine how influential and if that is related with the news site
  texts = [ (src, tweet.text, str(tweet.created_at)) for src, sn in flat_rt_id for tweet in api.user_timeline(screen_name=sn, count=num_posts, include_rts=False)]
  print(texts[0][2])
  
  return texts

def _smooth(data, smooth_factor):
  new_data = []
  for i in range(0, len(data), smooth_factor):
    temp = data[i:i+smooth_factor]
    avg = sum([t['sentiment'] for t in temp])/float(len(temp))
    data[i]['sentiment'] = avg
    new_data.append(data[i])
  return new_data

def smooth(data, smooth_idx, smooth_factor):
  source_set = set([d['source'] for d in data])
  for d in data:
    d['date'] = datetime.strptime(d['time'], "%Y-%m-%d %X")

  source_dict = defaultdict(list)

  for d in data:
    source_dict[d['source']].append(d)

  result = []
  for key in source_dict:
    res = sorted(source_dict[key], key=lambda k: k['date'])
    source_dict[key] = _smooth(res, smooth_factor)
    print(key, len(source_dict[key]))
    result += source_dict[key]

  return result
 
