import tensorflow as tf
import sys
import sys
import tensorflow as tf
import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from processing import pad

SEQ_LENGTH=16
STATE_SIZE=64
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
  sess = tf.Session()
  print(argv[0])
  saver = tf.train.import_meta_graph(argv[0])
  saver.restore(sess, argv[1])
  graph = tf.get_default_graph()
  x_plhr = graph.get_tensor_by_name('Placeholder:0')
  pred = tf.get_collection('pred')
  print(x_plhr)
  print([n.name for n in tf.get_default_graph().as_graph_def().node])
  print(pred)
 
   
  test_data = read_csv(argv[2])
  tknzr = TweetTokenizer(strip_handles=True)
  cleaned_data = []
  for sent, tweet in test_data:
    try: 
      tknzd = tknzr.tokenize(tweet)
      cleaned_data.append((sent, tknzd))
    except:
      pass
  print("NUM_PASSED:", len(test_data) - len(cleaned_data), "TOTAL:", len(test_data))
  word_vectors = KeyedVectors.load_word2vec_format(argv[3], binary=True)
  sent_vec = [(sent,string_to_vec(s, word_vectors)) for sent, s in cleaned_data]
  sent_vec = [s for s in sent_vec if s[1] is not None]
  
  padded_vecs = pad(SEQ_LENGTH, [s[1] for s in sent_vec], 300)
  sentiments = np.array([[float(s[0].strip("\""))] * SEQ_LENGTH for s in sent_vec]).reshape((len(sent_vec), SEQ_LENGTH, 1))
  num_samples = len(padded_vecs)
  fd = {}
  samples = np.random.choice(num_samples, BATCH_SIZE)
  fd[x_plhr] = padded_vecs[samples]
  sub_sent = sentiments[samples]
#  fd[y_plhr] = sentiments[samples].reshape([BATCH_SIZE * SEQ_LENGTH, 1])
  curr_sent = sess.run(pred, feed_dict=fd)
  print(curr_sent[0])
  result = curr_sent[0].reshape([BATCH_SIZE, SEQ_LENGTH, 1])
  pred_l = []
  for i in range(BATCH_SIZE):
    pred_l.append((np.mean(result[i]),np.mean(sub_sent[i])))
  print(pred_l)
#  result = np.append(curr_sent[0], sentiments, axis=1)
#  avg_res = []
  
#  print(result)


 

if  __name__ == "__main__":

  main(sys.argv[1:])
