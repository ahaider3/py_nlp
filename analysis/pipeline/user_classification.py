import sys
sys.path.append('/home/cc/pytweet/analysis/io')
sys.path.append('/home/cc/pytweet/analysis/')
sys.path.append('/home/cc/pytweet/analysis/ml')


from processing import vectorize
from mongo_funcs import read
from logreg import logreg
from svm import svm
from forest import random_forest
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from collections import defaultdict

  

data = list(read("follow"))
#s = np.random.choice(len(list(data)), 10000)
#data = [data[d] for d in s]

train_data = []

freqs = defaultdict(int)

for d in data:
  train_data.append((d['source'], d['text']))
  freqs[d['source']] += 1

print("Frequencies:", freqs)

wv = KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)

x, y = vectorize(train_data, wv)

#print(y.shape)
y = np.array([arr[0] for arr in y])
test_samples = np.random.choice(len(x), int(len(x) * .1))
train_samples = [i for i in range(len(x)) if i not in test_samples]

print(len(x), len(y))
assert(len(x) == len(y))
x = x.reshape((len(x), 16*300))
#logreg(x[train_samples], y[train_samples], sys.argv[2], (x[test_samples], y[test_samples]))

random_forest(x[train_samples], y[train_samples], sys.argv[2], (x[test_samples], y[test_samples]))



