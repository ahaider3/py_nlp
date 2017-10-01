import numpy as np
from scipy.special import expit

def sigmoid(x):
  return expit(x)
#  return 1/(1+np.exp(-x))


def dsigmoid(x):

  xp = sigmoid(x)
  return xp * ( 1 - xp)


def softmax(x):
  m = np.max(x)
  e_x = np.exp(x-m)
  return e_x/e_x.sum()


def tanh(x):
  return np.tanh(x)


def dtanh(x):
  return 1  - (tanh(x))**2


def cross_entropy(x, ytrue):

  return -np.log(x[0,  ytrue])
