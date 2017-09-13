import sys
sys.path.append('/home/cc/pytweet/analysis')
import tensorflow as tf
import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from processing import pad
from processing import read_subj

from graph_builders import build_logreg

SEQ_LENGTH=16
STATE_SIZE=256
BATCH_SIZE=1
NUM_ITER = int(1e8)
NUM_CLASSES = 2

def read_csv(data_path):
  f = open(data_path, "r")
  lines = f.readlines()
  tweet_sen_tup = [(line.split(',')[0], line.split(',')[5]) for line in lines]
  return tweet_sen_tup


def string_to_vec(str_list, wv):

  result = [wv[s] for s in str_list if s in wv.vocab]
  if len(result) > 0:
    return np.concatenate(result)

def main(argv):

  data = read_subj(argv[0], argv[1])
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
  word_vectors = KeyedVectors.load_word2vec_format(argv[2], binary=True)
  # convert to word vectors
  num_els_1 = sum([len(s[1]) for s in cleaned_data])

  sent_vec = [(sent,string_to_vec(s, word_vectors)) for sent, s in cleaned_data]
  sent_vec = [s for s in sent_vec if s[1] is not None]
  num_els = sum([len(s[1]) for s in sent_vec])
  
  print(num_els/float(len(sent_vec)), num_els_1)
  print(len(word_vectors['computer']), len(word_vectors['pizza']))
  padded_vecs = pad(SEQ_LENGTH, [s[1] for s in sent_vec], 300)
 # x_plhr, y_plhr, loss, train_op = build_rnn(SEQ_LENGTH, BATCH_SIZE, STATE_SIZE, 300)
  x_plhr, y_plhr, loss, train_op = build_logreg(SEQ_LENGTH, BATCH_SIZE, STATE_SIZE, 300, NUM_CLASSES)

  sentiments = np.array([int(s[0]) for s in sent_vec]).reshape((len(sent_vec), 1)).astype(np.int32)
  

  with tf.Session() as sess:
    num_samples = len(padded_vecs)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(NUM_ITER):
      samples = np.random.choice(num_samples, BATCH_SIZE)
      fd = {}
      fd[x_plhr] = padded_vecs[samples]
      one_hot = np.zeros([NUM_CLASSES], dtype=np.float32)

 #     fd[y_plhr] = sentiments[samples]
      one_hot[sentiments[samples][0]] = 1.0
      fd[y_plhr] = one_hot.reshape(1, -1)
      loss_curr, _ = sess.run([loss, train_op], feed_dict=fd)
      if not i %10000:
        saver.save(sess, argv[3], global_step = i)
      if not i % 1000:
        print(loss_curr)

      
      


if __name__ == "__main__":
  main(sys.argv[1:])
