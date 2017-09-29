import analysis
import numpy as np
import sys
import analysis
import pickle
import codecs
lr = 1e-1
seq_length=25
hidden_size = 100

#news_data = list(analysis.read("total_news"))

#txt = ""
#for d in news_data:
#  txt += d["old_title"] + " " + d["old_description"] + " "
#f = codecs.open(sys.argv[1], encoding='utf-8')
#txt = f.read()
txt = open(sys.argv[1]).read()
data = txt
vocab = list(set(txt))
print("DATASET SIZE:", len(data), "VOCAB SIZE:", len(vocab))
print(vocab)
#sys.exit()
char_ix = {c:idx for idx, c in enumerate(vocab)}
ix_char = {idx:c for idx, c in enumerate(vocab)}
print(vocab)
lstm = analysis.LSTM(len(vocab), hidden_size)
adagrad = analysis.Adagrad_LSTM(lr, len(vocab), hidden_size)
#inputs = [i for i in range(len(vocab))]
#outputs = [min(25, i+1) for i in inputs]

smooth_loss = -np.log(1./len(vocab)) * seq_length

p = 0
for i in range(10000000): 
  if p + seq_length + 1 >= len(data) or not i:
    hprev = np.zeros((hidden_size, 1))
    p = 0
    c0 = np.zeros((1, hidden_size))
    h0 = np.zeros((1, hidden_size))

  inputs = [char_ix[c] for c in data[ p: p +  seq_length]]
  outputs = [char_ix[c] for c in data[ p+1: p+1+seq_length]]
  
  inX = []

  for ind, i_ in enumerate(inputs):
    arr = np.zeros(len(vocab))
    arr[i_] = 1
    inX.append(arr)

  inX = np.array(inX).reshape((seq_length, 1, len(vocab)))
#  inX = np.array(inputs).reshape((seq_length, 1, len(vocab)))
#  if not i % 100:
#    idxs = rnn.inference(inputs[0], 200, hprev)
#    out = ''.join(ix_char[ix] for ix in idxs)
#    print '----\n %s \n----' % (out, )
  Hout, _, _, cache = lstm.forward(inX, c0=c0, h0=h0)
  wrand = np.random.randn(*Hout.shape)
  deriv = lstm.compute_apply(wrand, cache)
  loss = 1
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if not i % 100:

    print(smooth_loss)

#    chars = [ix_char[idx] for idx in ix_char]
#    out = ''.join(chars)
#  if not i % 1000:
#    pickle.dump(rnn, open(sys.argv[1], "w"))
  

  adagrad.update([lstm.WLSTM, c0, h0], deriv)

  p += seq_length
