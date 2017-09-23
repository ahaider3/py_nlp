import tensorflow as tf
import analysis
from nltk.tokenize import TweetTokenizer
import sys
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict
from sklearn import linear_model
import numpy as np
from nltk.corpus import stopwords

def evaluate(vects, label, vects_train, label_train):

  maxes = list(np.argmax(vects, axis=1))
  num_equal = 0
  for tup in zip(label, maxes):
    if tup[0] == tup[1]:
      num_equal+=1
  maxes_train = list(np.argmax(vects_train, axis=1))
  num_equal_train = 0
  for tup in zip(label_train, maxes_train):
    if tup[0] == tup[1]:
      num_equal_train+=1
#  print(vects.shape)
#  print(maxes)
  assert(len(maxes) == len(vects))
  print("TEST EVAL:", float(num_equal)/len(maxes), "TRAIN_ACC:", float(num_equal_train)/len(maxes_train))


stops = set(stopwords.words('english'))
data = list(analysis.read("total_news"))
source_word = defaultdict(list)
kv = KeyedVectors.load_word2vec_format(sys.argv[1], binary=True)
tknzr = TweetTokenizer(strip_handles=True)
for d in data:
  words = tknzr.tokenize(d['title'])
  word_vecs = [kv[w] for w in words if w in kv.vocab and w not in stops]
  source_word[d['source']] += word_vecs

labels=source_word.keys()
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

x_plhr, y_plhr, loss, train_op, pred = analysis.build_ffn(64, 300, [256,128,64,32,16,len(labels)], len(labels))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10000000):
    sub_sample = np.random.choice(len(X_train), 64)
    onehot = np.zeros((64, len(labels)))
    for ind, idx in enumerate(list(y_train[sub_sample])):
      onehot[ind][idx] = 1

    curr_loss, _ = sess.run([loss, train_op], feed_dict={x_plhr: X_train[sub_sample], y_plhr: onehot})
    if not i % 100:
      print(curr_loss, "AT STEP:", i)
    if not i % 1000:
      vects = sess.run(pred, feed_dict={x_plhr: X_test})
      vects_train = sess.run(pred, feed_dict={x_plhr: X_train})

      evaluate(vects, list(y_test), vects_train, list(y_train))

