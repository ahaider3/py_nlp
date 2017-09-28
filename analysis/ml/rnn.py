import numpy as np


class RNN(object):

  def __init__(self, input_size, hidden_size, output_size):
    self.Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
    self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    self.Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
    self.bh = np.zeros((hidden_size, 1)) # hidden bias
    self.by = np.zeros((output_size, 1)) # output bias
    self.hprev = np.zeros((hidden_size, 1))
    self.hidden_size = hidden_size
    self.input_size = input_size

  def forward_back_pass(self, inputs, targets, hprev):

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    for t in range(len(inputs)):
      arr = np.zeros((self.input_size, 1))
      xs[t] = arr
      xs[t][inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh)
      ys[t] = np.dot(self.Why, hs[t]) + self.by
      ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
      loss += -np.log(ps[t][targets[t], 0])

    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
      dy = np.copy(ps[t])
      dy[targets[t]] -= 1
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext
      dhraw = (1 - hs[t] * hs[t]) * dh
      dbh += dhraw
      dWxh += np.dot(dhraw, xs[t].T)
      dWhh += np.dot(dhraw, hs[t-1].T)
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam)

    self.hprev = hs[len(inputs) -1]
    return loss, [self.Wxh, self.Whh, self.Why, self.bh, self.by], [dWxh, dWhh, dWhy, dbh, dby], hs[len(inputs)-1]

  def inference(self, init_char, num_chars, h):

    x = np.zeros((self.input_size, 1))
    x[init_char] = 1
    output_chars = []
#    h = np.copy(self.hprev)
    for t in range(num_chars):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      ix = np.random.choice(range(self.input_size), p=p.ravel())
      x = np.zeros((self.input_size,1))
      x[ix] = 1
      output_chars.append(ix)
    return output_chars

  def clear_state(self):
    self.hprev = np.zeros((self.hidden_size,1))

class Adagrad(object):

  def __init__(self, lr, input_size, hidden_size, output_size):

    self.lr = lr
    self.mWxh = np.zeros((hidden_size, input_size)) 
    self.mWhh = np.zeros((hidden_size, hidden_size))
    self.mWhy = np.zeros((output_size, hidden_size))
    self.mbh = np.zeros((hidden_size, 1))
    self.mby = np.zeros((input_size, 1)) 
    self.mem = [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]

  def update(self, model_params, model_grads):

    for param, dparam, mparam in zip(model_params, model_grads, self.mem):

      mparam += dparam * dparam
      param += -self.lr * dparam / np.sqrt(mparam+ 1e-8)

    

