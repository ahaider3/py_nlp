import numpy as np


class LSTM(object):

  def __init__(self, input_size, hidden_size, forget_bias_init = 3):

    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size)/np.sqrt(input_size + hidden_size) 

    WLSTM[0, :] = 0
    if forget_bias_init != 0:
      WLSTM[0, hidden_size: 2*hidden_size] = forget_bias_init

    self.WLSTM = WLSTM


  def forward(self, X, c0=None, h0=None):
  
    n, b, input_size = X.shape

    d = self.WLSTM.shape[1]/4
    
    if c0 is None: c0 = np.zeros((b,d)) # batch size X hidden size
    if h0 is None: h0 = np.zeros((b,d)) # batch size X hidden size
    
    xphpb = self.WLSTM.shape[0] # input size + hidden size + bias
 
    Hin = np.zeros((n, b, xphpb))
    Hout = np.zeros((n, b, d))  # seq len X batch size X output for each t output a hidden
    IFOG = np.zeros((n, b, d* 4))
    IFOGf = np.zeros((n, b, d*4))
    C = np.zeros((n,b,d))
    Ct = np.zeros((n,b,d))

    for t in range(n):
      prevh = Hout[t-1] if t> 0 else h0
      Hin[t,:, 0] = 1 # for this seq set bias to 1
      Hin[t, :, 1:input_size+1] = X[t] # populate inpute
      Hin[t, :, input_size+1:] = prevh

      # compute
      IFOG[t] = Hin[t].dot(self.WLSTM)

      IFOGf[t, :, :3*d] = 1.0/(1.0 + np.exp(-IFOG[t,:,:3*d]))
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:])

      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3*d:] + IFOGf[t,:,d:2*d] * prevc # i_t * c_in_t + f_t * c_t-1
      Ct[t] = np.tanh(C[t])
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t] # c_t * o_T


    cache = {}
    cache['WLSTM'] = self.WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache

  def compute_apply(self, dHout_in, cache, dcn = None, dhn = None):
    WLSTM = self.WLSTM
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias
 
    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects

    if dcn is not None: dC[n-1] += dcn.copy()
    if dhn is not None: dHout[n-1] += dhn.copy()

    for t in reversed(range(n)):
      tanhCt = Ct[t]

      dIFOGf[t, :, 2*d: 3*d] = tanhCt * dHout[t]

      dC[t] += (1-tanhCt**2) * (IFOGf[t, :, 2*d: 3*d] * dHout[t])

      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,:, d:2*d] * dC[t]

      else:
        dIFOGf[t,:, d:2*d] = c0 * dC[t]
        dc0 = IFOGf[t, :, d:2*d] * dC[t]

      dIFOGf[t, :, :d] = IFOGf[t, :, 3*d:] * dC[t]
      dIFOGf[t, :, 3*d:] = IFOG[t,:,:d] * dC[t]


      dIFOG[t,:, 3*d:] = (1-IFOGf[t, :, 3*d:] **2) * dIFOGf[t,:,3*d:]

      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
 
      # backprop matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())
 
      # backprop the identity transforms into Hin
      dX[t] = dHin[t,:,1:input_size+1]
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:]
      else:
        dh0 += dHin[t,:,input_size+1:]
 
    return [dWLSTM, dc0, dh0]

class Adagrad_LSTM(object):

  def __init__(self, lr, input_size, hidden_size):

    self.lr = lr
    self.mWLSTM = np.zeros((input_size + hidden_size + 1, 4 * hidden_size))
    self.mdc0 = np.zeros((1, hidden_size))
    self.mdh0 = np.zeros((1, hidden_size))
    self.mem = [self.mWLSTM, self.mdc0, self.mdh0]

  def update(self, model_params, model_grads):

    for param, dparam, mparam in zip(model_params, model_grads, self.mem):

      mparam += dparam * dparam
      param += -self.lr * dparam / np.sqrt(mparam+ 1e-8)


