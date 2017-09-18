import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from processing import pad, get_recent_tweets
import pickle
from scipy.stats import mode

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


    
class InferTimeline(object):


  def __init__(self, kv_path, seq_len, model_path, num_to_label, api=None):

    self.kv = KeyedVectors.load_word2vec_format(kv_path, binary=True)

    self.tknzr = TweetTokenizer(strip_handles=True)
    self.seq_len = seq_len
    self.label_map = num_to_label
    self.api = api
    model_file = open(model_path, "r")
    self.model = pickle.load(model_file)




  def infer(self, tweet):
    tweets = get_recent_tweets(self.api, tweet, num_posts=1)
    if tweets is None or not len(tweets):
      return
    tkns = [self.tknzr.tokenize(text) for text in tweets]

    vec_tweet = [self.string_to_vec(text_vec, self.kv) for text_vec in tkns]
    vec_tweet = [v for v in vec_tweet if v is not None]
    print('CHECK1', len(tweets), vec_tweet)
    if vec_tweet is not None and len(vec_tweet):
      vec_tweet = pad(self.seq_len, vec_tweet, NUM_FEAT)
      vec_tweet = vec_tweet.reshape((len(vec_tweet), NUM_FEAT * self.seq_len))
      pred = self.model.predict(vec_tweet)
      pred = [int(p) for p in pred]
      print( 'check', pred)
      m = int(mode(pred)[0])
      print(self.label_map[m])
      return self.label_map[m]
    
  def basic_infer(self, tweet):
    text = tweet['text']
    tkns = self.tknzr.tokenize(text)
    vec_tweet = self.string_to_vec(tkns, self.kv)
    if vec_tweet is not None and len(vec_tweet):
      vec_tweet = pad(self.seq_len, [vec_tweet], NUM_FEAT)
      vec_tweet = vec_tweet.reshape((1, NUM_FEAT * self.seq_len))
      pred = self.model.predict(vec_tweet)
      pred = [int(p) for p in pred]
      m = int(mode(pred)[0])
      return {"text":text, "type":self.label_map[m], "time": tweet['created_at']}

  def string_to_vec(self, str_list, wv):

    result = [wv[s] for s in str_list if s in wv.vocab]
    if len(result) > 0:
      return np.concatenate(result)

