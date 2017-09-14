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
  statuses = []
  for src in sources:
    try:
      statuses.append((src, api.user_timeline(screen_name=src, count=num_headlines, include_rts=False)) )
    except:
      pass


#  statuses = [		for src in sources]
#    tweets = api.user_timeline(target, target,target)
  rts_id = []
  for src, status in statuses:
    for tweet in status:
      try:
        rts_id.append((src, api.retweets(tweet.id, branch_off)))
      except: 
        pass

  flat_rt_id = []

  for src, tweets in rts_id:
    for tweet in tweets:
      try:
        flat_rt_id.append((src, tweet.author.screen_name))
      except:
        pass
#  with_users = [ (rt[1].author.screen_name, api.get_user(rt[1].author.screen_name, 
#                                                        rt[1].author.screen_name,
#                                                        rt[1].author.screen_name).location,
#                                        rt[0])
#  
#                                        for rt in flat_rt_id]
  # 1) add subj analysis 2) predict given string of tweets the news site a user most agrees with
  # follower and follower count to determine how influential and if that is related with the news site

  texts = []
  for src, sn in flat_rt_id:
    try:
      tweets = api.user_timeline(screen_name=sn, count=num_posts, include_rts=False)
    except:
      pass
    for tweet in tweets:
      texts.append((src, tweet.text, str(tweet.created_at))) 


  
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
 
def string_to_vec(str_list, wv):

  result = [wv[s] for s in str_list if s in wv.vocab]
  if len(result) > 0:
    return np.concatenate(result)


def vectorize(data, word_vectors, SEQ_LENGTH=16):
  tknzr = TweetTokenizer(strip_handles=True)
  cleaned_data = []
#  cleaned_data = [(sent, tknzr.tokenize(tweet)) for sent, tweet in data]
  for sent, tweet in data:
    try: 
      tknzd = tknzr.tokenize(tweet)
      cleaned_data.append((sent, tknzd))
    except:
      pass
  print("NUM_PASSED:", len(data) - len(cleaned_data), "TOTAL:", len(data))

  sent_vec = [(sent,string_to_vec(s, word_vectors)) for sent, s in cleaned_data]
  sent_vec = [s for s in sent_vec if s[1] is not None]

  to_np = onehot(sent_vec)
  onehot_vecs = [to_np[s[0]] for s in sent_vec]
  padded_vecs = pad(SEQ_LENGTH, [s[1] for s in sent_vec], 300)
  print(len(onehot_vecs), len(padded_vecs))
  assert(len(onehot_vecs) == len(padded_vecs))

  return padded_vecs, onehot_vecs


  


def onehot(data):

  set_ = set()
  for d in data:
    set_.add(d[0])

  els = list(set_)

  d = {}
  
  for ind, el in enumerate(els):
    arr = np.zeros([1]).astype(np.float32)
    arr[0] = float(ind)
    d[el] = arr
  print(d)
  return d

  

