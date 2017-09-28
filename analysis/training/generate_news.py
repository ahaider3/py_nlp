import analysis
import numpy as np
import sys
import analysis
import pickle
import codecs
lr = 1e-1
seq_length=25
hidden_size = 100

news_data = list(analysis.read("total_news"))

txt = ""
for d in news_data:
  txt += d["old_title"] + " " + d["old_description"] + " "
#f = codecs.open(sys.argv[1], encoding='utf-8')
#txt = f.read()
data = txt
vocab = list(set(txt))
print("DATASET SIZE:", len(data), "VOCAB SIZE:", len(vocab))
print(vocab)
#sys.exit()
char_ix = {c:idx for idx, c in enumerate(vocab)}
ix_char = {idx:c for idx, c in enumerate(vocab)}
print(vocab)
rnn = analysis.RNN(len(vocab), hidden_size, len(vocab))
adagrad = analysis.Adagrad(lr, len(vocab), hidden_size, len(vocab))
inputs = [i for i in range(len(vocab))]
outputs = [min(25, i+1) for i in inputs]

smooth_loss = -np.log(1./len(vocab)) * seq_length

p = 0
for i in range(10000000): 
  if p + seq_length + 1 >= len(data) or not i:
    rnn.clear_state()
    hprev = np.zeros((hidden_size, 1))
    p = 0
  inputs = [char_ix[c] for c in data[ p: p +  seq_length]]
  outputs = [char_ix[c] for c in data[ p+1: p+1+seq_length]]

  if not i % 100:
    idxs = rnn.inference(inputs[0], 200, hprev)
    out = ''.join(ix_char[ix] for ix in idxs)
    print '----\n %s \n----' % (out, )
  loss, params, derivs, hprev = rnn.forward_back_pass(inputs, outputs, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if not i % 100:

    print(smooth_loss)

#    chars = [ix_char[idx] for idx in ix_char]
#    out = ''.join(chars)
  if not i % 1000:
    pickle.dump(rnn, open(sys.argv[1], "w"))
  

  adagrad.update(params, derivs)

  p += seq_length
