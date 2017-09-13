import sys
import tensorflow as tf
import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from processing import pad
from graph_builders import build_multi_rnn

SEQ_LENGTH=16
STATE_SIZE=256
BATCH_SIZE=1
NUM_ITER = int(1e6)

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

  data = read_csv(argv[0])
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
  word_vectors = KeyedVectors.load_word2vec_format(argv[1], binary=True)
  # convert to word vectors
  num_els_1 = sum([len(s[1]) for s in cleaned_data])

  sent_vec = [(sent,string_to_vec(s, word_vectors)) for sent, s in cleaned_data]
  sent_vec = [s for s in sent_vec if s[1] is not None]
  num_els = sum([len(s[1]) for s in sent_vec])
  
  print(num_els/float(len(sent_vec)), num_els_1)
  print(len(word_vectors['computer']), len(word_vectors['pizza']))
  padded_vecs = pad(SEQ_LENGTH, [s[1] for s in sent_vec], 300)
 # x_plhr, y_plhr, loss, train_op = build_rnn(SEQ_LENGTH, BATCH_SIZE, STATE_SIZE, 300)
  x_plhr, y_plhr, loss, train_op = build_multi_rnn(SEQ_LENGTH, BATCH_SIZE, STATE_SIZE, 300)

  sentiments = np.array([[float(s[0].strip("\""))] * SEQ_LENGTH for s in sent_vec]).reshape((len(sent_vec), SEQ_LENGTH, 1))
  

  with tf.Session() as sess:
    num_samples = len(padded_vecs)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(NUM_ITER):
      samples = np.random.choice(num_samples, BATCH_SIZE)
      fd = {}
      fd[x_plhr] = padded_vecs[samples]
      fd[y_plhr] = sentiments[samples].reshape([BATCH_SIZE * SEQ_LENGTH, 1])
      loss_curr, _ = sess.run([loss, train_op], feed_dict=fd)
      if not i %100:
        saver.save(sess, argv[2], global_step = i)
      print(loss_curr)

      
      


if __name__ == "__main__":
  main(sys.argv[1:])
