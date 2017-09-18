import json
import sys

sys.path.append('/home/cc/pytweet/analysis/io')
sys.path.append('/home/cc/pytweet/analysis/twitter')
sys.path.insert(0, '/home/cc/pytweet/')

from mongo_funcs import read, write, delete
from rake_nltk import Rake
from collections import defaultdict
sys.path.append('/home/cc/pytweet/analysis/')
from src import *
from infer import InferTimeline, Infer
from nltk import bigrams
MAX_TWEETS=1000
from datetime import datetime
from tweet_extraction import *
from multiprocessing import Process, Queue


#        res = self.model(json.loads(data))
#        self.res.append(res)
#        if not (self.counter % self.freq):
#          self.output(self.res)
#          self.res = []
         
#        print(res)
delete("tweet_frequency")

def write_func(data, rt):
  smoothed_data = defaultdict(list)
  for d in data:
    src = d['type']
    smoothed_data[src].append(d)
  cleaned_data = []
  for k,v in smoothed_data.items():
    cleaned_data.append({"type": k, "frequency": len(v), "time": rt})
      
    
  print(cleaned_data)
  write(cleaned_data, "tweet_frequency")


argv = sys.argv[1:]
queue = Queue()
m = {0:'cnn', 1:'BreitbartNews', 2:'FoxNews'}
infer_tag = InferTimeline(argv[0], 16, argv[1],m)
init = initialize.PredictInitializer()
init.initialize(infer_tag.basic_infer, 5, write_func, queue)
stream = init.get_stream()

start = datetime.now()
curr_res = []
while True:
  try:
    
    init = initialize.PredictInitializer()
    init.initialize(infer_tag.basic_infer, 5, write_func, queue)
    stream = init.get_stream()
    global api
    api = init.api
    stream.filter(track=["donald trump"])
  except:

    curr_res += init.l.res
    if (datetime.now() - start).total_seconds() > 120:
      write_func(curr_res, str(start))
      curr_res = []
      start = datetime.now()
    continue



