import numpy as np

'''
Linear support vector machine with regularization - implementation is based on
the implementation that I (Volodymyr) have done for the cs231n
course assignments (see cs231n.github.io). Some functions such as svm_loss
where written by me, while some boilerplate was given in the assignment, so I
will reuse it.
'''
class SVM:

  def svm_loss(self, W, X, y, reg):
    '''
    Structured SVM loss function, vectorized implementation.
    '''
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
  
    num_classes = W.shape[1]
    num_train = X.shape[0]
  
    res = X.dot(W)
  
    ones_mask = np.zeros(res.shape)
    ones_mask[np.arange(num_train), y] = 1
  
    corrects = ones_mask * res
  
    res += corrects
    res -= corrects.dot(np.ones([num_classes, 1])).repeat(num_classes, axis=1)
    res += np.ones(res.shape)
    res = res * (np.ones(res.shape) - ones_mask)
    res = np.maximum(res, np.zeros(res.shape))
  
    loss = res.sum() / num_train + reg * np.sum(W * W)
  
    X_mask = np.zeros(res.shape)
    X_mask[res > 0] = 1
    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(num_train), y] = -incorrect_counts
    dW = X.T.dot(X_mask) / num_train + 2 * reg * W
    return loss, dW

  def fit(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=4000, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    
    # Set initial weights - out of the blue (but this is slightly better than zeroes).
    self.W = 0.01 * np.random.randn(dim, num_classes)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      # evaluate loss and gradient
      loss, grad = self.svm_loss(self.W, X, y, reg)
      loss_history.append(loss)

      self.W -= grad * learning_rate

      if verbose and it % 10 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[0])
    y_pred = X.dot(self.W).argmax(axis=1)
    return y_pred
  