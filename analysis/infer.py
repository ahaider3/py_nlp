import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from processing import pad


NUM_FEAT=300

class Infer(object):


  def __init__(self, kv_path, seq_len, model_path):

    self.kv = KeyedVectors.load_word2vec_format(kv_path, binary=True)
    self.tknzr = TweetTokenizer(strip_handles=True)
    self.seq_len = seq_len

    self.sess = tf.Session()
    self.saver = tf.train.import_meta_graph(model_path + ".meta")
    self.saver.restore(self.sess, model_path)
    graph = tf.get_default_graph()
    self.x_plhr = graph.get_tensor_by_name('Placeholder:0')
    self.pred = tf.get_collection('pred')



  def infer(self, text):
    tknzd = self.tknzr.tokenize(text)
    wv = self.string_to_vec(tknzd, self.kv)
    if wv is not None:
      wv_padded = pad(self.seq_len, [wv], NUM_FEAT)

      result = self.sess.run(self.pred, feed_dict={self.x_plhr: wv_padded})[0]
      result = result.reshape([1, self.seq_len, 1])
      return np.mean(result[0])
    return -1
  def infer_logreg(self, text):
    tknzd = self.tknzr.tokenize(text)
    wv = self.string_to_vec(tknzd, self.kv)
    if wv is not None:
      wv_padded = pad(self.seq_len, [wv], NUM_FEAT)

      result = self.sess.run(self.pred, feed_dict={self.x_plhr: wv_padded})[0]
      result = result.reshape([1, -1])
      return np.argmax(result)
    return -1
   

  def string_to_vec(self, str_list, wv):

    result = [wv[s] for s in str_list if s in wv.vocab]
    if len(result) > 0:
      return np.concatenate(result)


    

