import numpy as np





class FC(object):



  def __init__(self, nr, nc):

    self.xdim = nr
    self.ydim = nc

    w = np.random.randn(self.xdim, self.ydim) * 0.01
    b = np.random.randn(self.ydim, 1)

    self.weights = w
    self.bias = b


  def forward(self, inX):

    return np.dot(inX, self.weights) + self.bias
    
