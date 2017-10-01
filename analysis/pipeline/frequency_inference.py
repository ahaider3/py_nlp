import json
import sys

import analysis
from rake_nltk import Rake
from collections import defaultdict

import src
from nltk import bigrams
MAX_TWEETS=1000
from datetime import datetime
from multiprocessing import Process, Queue
import pickle

#        res = self.model(json.loads(data))
#        self.res.append(res)
#        if not (self.counter % self.freq):
#          self.output(self.res)
#          self.res = []
         
#        print(res)

def write_func(data, rt):
  smoothed_data = defaultdict(list)
  for d in data:
    src = d['type']
    smoothed_data[src].append(d)
  cleaned_data = []
  for k,v in smoothed_data.items():
    cleaned_data.append({"type": k, "frequency": len(v), "time": rt})
      
    
  print(cleaned_data)
  analysis.write(cleaned_data, "tweet_frequency")
  analysis.write_smooth()


argv = sys.argv[1:]
queue = Queue()
m = pickle.load(open(argv[2], "r"))
m_inv = {v[0]:k for k,v in m.items()}
infer_tag = analysis.InferTimeline(argv[0], 16, argv[1],m_inv)

start = datetime.now()
curr_res = []

while True:
  try:
    init = src.PredictInitializer()
    init.initialize(infer_tag.basic_infer, write_func, 120*5, datetime.now())
    stream = init.get_stream()
    global api
    api = init.api
    stream.filter(track=["donald trump"])
  except:
    if len(init.l.res) > 0:
      curr_res += init.l.res
      if (datetime.now() - start).total_seconds() > 120*5:
        write_func(curr_res, str(start))
        curr_res = []
        start = datetime.now()
    continue



from tweet_extraction import *
