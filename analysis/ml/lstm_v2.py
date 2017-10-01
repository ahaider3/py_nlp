import numpy as np
import analysis


class LSTM_V2(object):


  def __init__(self, input_size, hidden_size):
    self.ih = input_size + hidden_size
    self.Wf = np.random.randn(self.ih, hidden_size) / np.sqrt(self.ih/2.)
    self.Wi = np.random.randn(self.ih, hidden_size) / np.sqrt(self.ih/2.)
    self.Wc = np.random.randn(self.ih, hidden_size) / np.sqrt(self.ih/2.)
    self.Wo = np.random.randn(self.ih, hidden_size) / np.sqrt(self.ih/2.)
    self.Wy = np.random.randn(hidden_size, input_size) / np.sqrt(input_size/2.)
    self.bf = np.zeros((1,hidden_size))
    self.bi = np.zeros((1,hidden_size))
    self.bc = np.zeros((1, hidden_size))
    self.bo = np.zeros((1, hidden_size))
    self.by = np.zeros((1,input_size))
    self.params = {'Wf':self.Wf, 'Wi': self.Wi, 'Wc':self.Wc, 'Wo':self.Wo,
              'Wy':self.Wy, 'bf': self.bf, 'bc':self.bc, 'bo':self.bo,
              'by':self.by, 'bi': self.bi}


  def forward(self, X, c0, h0):

    input_ = np.column_stack((h0, X))
    hf = analysis.sigmoid(np.dot(input_, self.Wf) + self.bf)
    hi = analysis.sigmoid(np.dot(input_, self.Wi) + self.bi)
    ho = analysis.sigmoid(np.dot(input_, self.Wo) + self.bo)
    hc = analysis.tanh(np.dot(input_, self.Wc) + self.bc)
   
    c = hf * c0 + hi * hc
    h = ho * analysis.tanh(c)

    y = np.dot(h,self.Wy) + self.by
    prob = analysis.softmax(y)
    cache = {'hf':hf, 'hi':hi, 'ho':ho, 'hc':hc, 'c':c, 'h':h, 'y':y, 'c_old': c0, 'X': input_}

    return prob, c, h, cache

  def backward(self, prob, ytrain, dc_next, dh_next, cache):
  
    hf, hi, ho, hc = cache['hf'], cache['hi'], cache['ho'], cache['hc']
    c, h, y, c_old = cache['c'], cache['h'], cache['y'], cache['c_old']

    X = cache['X']

    dy = prob.copy()
#    print(np.shape(dy))
    dy[0, ytrain] -= 1
    dWy = h.T.dot(dy)
    dby = dy
    
    dh = dy.dot(self.Wy.T) + ho

    dho = analysis.tanh(c) * dh
    dho = analysis.dsigmoid(ho) * dho
    
    dc = ho * dh * analysis.dtanh(c)
    dc = dc + dc_next

    dhf =  c_old * dc
    dhf = analysis.dsigmoid(hf) * dhf

    dhi = hc * dc
    dhi = analysis.dsigmoid(hi) * dhi

    dhc = hi * dc
    dhc = analysis.dtanh(hc) * dhc

    dWf = X.T.dot(dhf)
    dbf = dhf
    dXf = dhf.dot(self.Wf.T)

    dWi = X.T.dot(dhi)
    dbi = dhi
    dXi = dhi.dot(self.Wi.T)

    dWo = X.T.dot(dho)
    dbo = dho
    dXo = dho.dot(self.Wo.T)
    
    dWc = X.T.dot(dhc)
    dbc = dhc
    dXc = dhc.dot(self.Wc.T)

    dX = dXo + dXc + dXi + dXf

    dh_next = dX[:, :np.shape(dho)[1]]
    dc_next = hf * hc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    return grad, dc_next, dh_next

class SGD(object):

  def __init__(self, lr):

    self.lr = lr

  def update(self, model_params, model_grads):

    for k, v in model_params.items():
      v += -self.lr * model_grads[k]




