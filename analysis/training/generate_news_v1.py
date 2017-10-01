import analysis
import numpy as np
import sys
import analysis
import pickle
import codecs
lr = 1e-4
seq_length=25
hidden_size = 100

news_data = analysis.read("tweets_1")
print("READ FROM DB:")
txt = ""
for d in news_data:
  txt += d["old_title"] + " "
#f = codecs.open(sys.argv[1], encoding='utf-8')
#txt = f.read()
#txt = open(sys.argv[1]).read()
data = txt
vocab = list(set(txt))
print("DATASET SIZE:", len(data), "VOCAB SIZE:", len(vocab))
print(vocab)
#sys.exit()
char_ix = {c:idx for idx, c in enumerate(vocab)}
ix_char = {idx:c for idx, c in enumerate(vocab)}
print(vocab)
lstm = analysis.LSTM_V2(len(vocab), hidden_size)
fc = analysis.FC(hidden_size, len(vocab))
adagrad = analysis.SGD(lr)
#inputs = [i for i in range(len(vocab))]
#outputs = [min(25, i+1) for i in inputs]

smooth_loss = -np.log(1./len(vocab)) * seq_length
c0 = np.zeros((1, hidden_size))
h0 = np.zeros((1, hidden_size))

p = 0
for i in range(10000000): 
  if p + seq_length + 1 >= len(data) or not i:
    hprev = np.zeros((hidden_size, 1))
  
    p = 0

  inputs = [char_ix[c] for c in data[ p: p +  seq_length]]
  outputs = [char_ix[c] for c in data[ p+1: p+1+seq_length]]
  caches = []
  probs = []

  loss = 0.0
  c = c0
  h = h0
  for inp, out in zip(inputs, outputs): 

#    for ind, i_ in enumerate(inputs):
#      arr = np.zeros(len(vocab))
#      arr[i_] = 1
#      inX.append(arr)
    inX = np.zeros((1, len(vocab))).astype(np.float32)
    inX[0, inp] = 1. # one hot vector
#    inX = np.array(inX).reshape((seq_length, 1, len(vocab)))
#    inX = np.array(inputs).reshape((seq_length, 1, len(vocab)))
#    if not i % 100:
#      idxs = rnn.inference(inputs[0], 200, hprev)
#      out = ''.join(ix_char[ix] for ix in idxs)
#      print '----\n %s \n----' % (out, )
#    Hout, c0, h0, cache = lstm.forward(inX, c0=c0, h0=h0)
    prob, c, h, cache = lstm.forward(inX, c0=c, h0=h)
    loss += analysis.cross_entropy(prob, out)
    caches.append(cache)
    probs.append(prob)
  print("LOSS:", loss/seq_length, loss)
  dc_next, dh_next = np.zeros_like(c0), np.zeros_like(h0)
  accum_grads = {}
  for prob, out, cache in reversed(list(zip(probs, outputs, caches))):
    grad, dc_next, dh_next = lstm.backward(prob, out, dc_next, dh_next, cache)
    for g in grad.keys():
      if g not in accum_grads:
        accum_grads[g] = np.zeros_like(grad[g])
      accum_grads[g] += grad[g]
  for k, v in accum_grads.items():
    accum_grads[k] = np.clip(v, -5., 5.)
  
  if not i % 10:
    chars_out = []
    inp = inputs[0]
    chars_out.append(inp)
    c_inf = np.copy(c)
    h_inf = np.copy(h)
    for k in range(100):
      inX = np.zeros((1, len(vocab)))
      inX[0, inp] = 1.
      prob, c_inf, h_inf, _ = lstm.forward(inX, c_inf, h_inf)
      ind = np.argmax(prob, axis=1)[0]
      chars_out.append(ind)
      inp = ind
    s_out = ''.join([ix_char[c] for c in chars_out])
    print("GUESS AT:", i, " IS:", s_out)
     

  adagrad.update(lstm.params, accum_grads)


  p += seq_length
#   res_output = fc.forward(Hout) # (seq_len, 1, output)
#   loss =0
#   ps = {}
#   for t in range(seq_length):
#     ps[t] = np.exp(res_output[t])/np.sum(np.exp(res_output[t]))
#     loss += -np.log(ps[t][outputs[t],0])
#   wrand = np.random.randn(*Hout.shape)
#   deriv = lstm.compute_apply(wrand, cache)
#   smooth_loss = smooth_loss * 0.999 + loss * 0.001
#   if not i % 10:
#
#     print(smooth_loss)
#
#      chars = [ix_char[idx] for idx in ix_char]
#      out = ''.join(chars)
#    if not i % 1000:
#      pickle.dump(rnn, open(sys.argv[1], "w"))
#   
#
#   adagrad.update([lstm.WLSTM, c0, h0], deriv)
#
