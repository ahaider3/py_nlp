import analysis
from nltk.tokenize import TweetTokenizer
import sys
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict
from sklearn import linear_model
import numpy as np
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))
data = list(analysis.read("total_news"))
source_word = defaultdict(list)
kv = KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
tknzr = TweetTokenizer(strip_handles=True)
for d in data:
  words = tknzr.tokenize(d['title'])
  word_vecs = [kv[w] for w in words if w in kv.vocab and w not in stops]
  source_word[d['source']] += word_vecs

def train(labels):
  
  total_data = []
  for l in labels:
    for vec in source_word[l]:
      total_data.append((l, vec))

  keymap = {key:ind for ind, key in enumerate(source_word.keys())}
  train_samples = np.random.choice(len(total_data), int(.80 * len(total_data)))
  y, X = zip(*[ (keymap[k[0]], k[1]) for k in total_data])
  X_np, y_np = np.array(X), np.array(y)
  
  X_train, y_train = X_np[train_samples], y_np[train_samples]
  test_samples = [i for i in range(len(total_data)) if i not in train_samples]
  X_test, y_test = X_np[test_samples], y_np[test_samples]
  
#  print("-----LOGISTIC REGRESSION-------")
#  analysis.logreg(X_train, y_train, "temp", (X_test, y_test))
  
  print("------ RANDOM FOREST------")
  res = analysis.random_forest(X_train, y_train, "temp", (X_test, y_test))
  print(labels, res)
  return res
  
#  print("------ SVM------")
#  analysis.svm(X_train, y_train, "temp", (X_test, y_test))
  
#  print("------ Adaboost------")
#  analysis.adaboost(X_train, y_train, "temp", (X_test, y_test))


labels = [(source, source_1) for source in source_word.keys() for source_1 in source_word.keys() if source_1 != source]
scores = defaultdict(float)
for label in labels:
  scores[label[0]]+=train(label)
vals = [v for k, v in scores.items()]
mean = np.mean(vals)
scores_final = []
for k, v in scores.items():
  d = {}
  d['source'] = k
  d['score'] = (-1 * (v-mean))
  scores_final.append(d)
analysis.delete("news_ratings")
analysis.write(scores_final, "news_ratings")
print("WROTE:", scores_final)
print("DISTRIBUTION:", [(k, len(v)) for k, v in source_word.items()])
