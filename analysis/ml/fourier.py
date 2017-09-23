import sys 
sys.path.append("/home/cc/pytweet/analysis/io/")
from mongo_funcs import *
import numpy as np
from datetime import datetime
from collections import defaultdict

def write_smooth():
  data = list(read("tweet_frequency"))
  out_data = []
  d_times = defaultdict(list)
  for d in data:
    r = {}
    r['time'] = d['time']
    r['rt'] = datetime.strptime(d['time'].split('.')[0], '%Y-%m-%d %X')
    r['freq'] = d['frequency']
    r['type'] = d['type']
    d_times[d['type']].append(r)
  for d in d_times:
    d_times[d] = sorted(d_times[d], key= lambda t: t['rt'])
    ys = [y['freq'] for y in d_times[d]]
    mean = float(sum(ys))/len(ys)
    ys = [y for y in ys]
    rfft = np.fft.rfft(ys)
    rfft[10:] = 0
  
    irfft = np.fft.irfft(rfft)
    print(len(irfft), len(d_times[d]))
  #  assert(len(irfft) == len(d_times[d]))
    for ind, k in enumerate(irfft):
      d_times[d][ind]['irfft'] = float(k)
      
  
  for d in d_times:
    out_data += d_times[d]
  
  delete("tweet_frequency_smooth")
  write(out_data, "tweet_frequency_smooth")
  print("WROTE SMOOTH:", len(out_data))
  
