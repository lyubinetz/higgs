import numpy as np
from helpers import _get_batch_indices

'''
This file contains functions for the neural network implementation.
It is loosely based on cs231n implementation that Volodymyr wrote - in the
course they used a much nicer structure with separate functions for various
layers, but I just stuck my forard and backward passes into the
loss() function. The implementation is made to be as compact as possible.

Some details of the implementation:

- Optimization can be a GD or SGD. We use RMSprop update strategy, which we found
  to be very effective.
- We only use a ReLU activation function - it's the easiest to implement.
- I adapted the API to mimic scipy's with fit() and predict().
'''

class SimpleNet(object):

  def __init__(self, matrix_dims, weight_magnitude=1e-1, reg=0.01, input_size=30):
    '''
    input_size - dimension of input data (can be different depending on featurization)
    matrix_dims - dimensions of matrices used in NN, aka layer sizes
    weight_magnitude - we initialize matrice with Gaussian(0, 1) multiplied by this magnitude
    reg - regularization strength (aka factor we multiply L2 reg with)
    '''
    # In this task we only deal with 2 classes
    classes = 2

    self.reg = reg
    self.params = {}
    self.num_layers = len(matrix_dims) + 1
    all_md = [input_size] + matrix_dims + [classes]

    # W is the weight term, b is a constant term, 1 step transormation is f(WX+b), where f
    # is activation function (ReLU in our case)
    for i in range(self.num_layers):
      # Note that we divide the resulting weights by np.sqrt(all_md[i]) - see
      # http://cs231n.github.io/neural-networks-2/#init - otherwise we found that
      # we cannot train the network for particular sizes. Typically this gets handled
      # well by batch normalization, but since it is nontrivial to implement (the
      # backward pass is hard), this 1 liner is good enough for us :)
      self.params['W' + str(i)] = \
        weight_magnitude * np.random.randn(all_md[i], all_md[i + 1]) / np.sqrt(all_md[i])
      self.params['b' + str(i)] = np.zeros(all_md[i + 1])

  def loss(self, X, y=None):
    '''
    Compute loss and gradients - does both forward and backward pass.
    '''
    # Compute the forward pass
    M = {} # Results of applying FC layer + ReLU
    for i in range(self.num_layers - 1):
      prev = X
      if i > 0:
        prev = M[i - 1]
      Wi, bi = self.params['W' + str(i)], self.params['b' + str(i)]
      Mi = prev.dot(Wi) + bi
      Mi = np.maximum(Mi, np.zeros(Mi.shape))
      M[i] = Mi.copy()

    Wlast, blast = self.params['W' + str(self.num_layers - 1)], \
      self.params['b' + str(self.num_layers - 1)]
    scores = Mi.dot(Wlast) + blast

    # If we just need the predictions
    if y is None:
      return scores

    # Compute softmax loss with L2 regularization. We tried hinge loss too,
    # results were comparable.
    # Disclosure: My implementation used a loop when I implemented this and I
    # looked up how people do it with crazy arrange() usage.
    shifted_by_max = scores - np.max(scores, axis=1, keepdims=True) # This is done for numerical stability, otherwise u might get huge exponents
    z = np.sum(np.exp(shifted_by_max), axis=1, keepdims=True)
    log_probs = shifted_by_max - np.log(z)
    probs = np.exp(log_probs)
    loss = -np.sum(log_probs[np.arange(scores.shape[0]), y]) / scores.shape[0]

    # Add regularization
    for i in range(self.num_layers):
      loss += self.reg * (self.params['W' + str(i)] * self.params['W' + str(i)]).sum()

    # Backward pass: compute gradients, storing them in the grads dictionary
    grads = {}

    # probs is sort of the gradient from the last layer - then we go back and update the weight matrices.
    probs[range(X.shape[0]), y] -= 1
    probs /= X.shape[0]

    grads['W' + str(self.num_layers - 1)] = M[self.num_layers - 2].T.dot(probs) + \
      2 * self.reg * self.params['W' + str(self.num_layers - 1)]
    grads['b' + str(self.num_layers - 1)] = np.sum(probs, axis=0)

    idx = self.num_layers - 2
    hg = probs
    # Go backward through layers
    while idx >= 0:
      # Relu backward
      hg = hg.dot(self.params['W' + str(idx + 1)].T)
      hg[M[idx] <= 0] = 0

      # Affine backward
      prev = X
      if idx > 0:
        prev = M[idx - 1]

      grads['W' + str(idx)] = prev.T.dot(hg) + 2 * self.reg * self.params['W' + str(idx)]
      grads['b' + str(idx)] = np.sum(hg, axis=0)

      idx -= 1

    return loss, grads

  def fit(self, X, y, learning_rate=0.1, num_iters=1000, verbose=False, update_strategy='rmsprop', decay_rate=0.9,
          optimization_strategy = 'gd', mini_batch_size=10000, mini_batch_class_ratio=None):
    '''
    Trains the neural network on dataset (X, y).
    - update_strategy is one of ('fixed', 'rmsprop')
    - optimization_strategy is one of ('gd', 'sgd')
    '''
    if verbose:
      print('Started fitting the neural network!')

    # Run gradient descent to optimize W
    loss_history = []
    ploss = 10000.0 # previous loss
    eps = 0.0000000001
    rmsprop_cache = {}

    for it in range(num_iters):
      # Evaluate loss and gradient
      if optimization_strategy == 'gd':
        loss, grad = self.loss(X, y=y)
      elif optimization_strategy == 'sgd':
        batch_indexes = _get_batch_indices(y, mini_batch_size, mini_batch_class_ratio)
        #batch_indexes = np.random.choice(len(y), mini_batch_size, replace=False)
        loss, grad = self.loss(X[batch_indexes], y[batch_indexes])
      loss_history.append(loss)

      if loss > ploss and optimization_strategy == 'gd':
          # Decrease LR so that we take smaller steps. We only do it when we cannot decrease the
          # loss with current LR.
          learning_rate *= 0.9
      if optimization_strategy == 'sgd':
        # When using SGD, increase the batch-size for a more stable loss and gradient and
        # decrease the learning rate using exponential decay
        #mini_batch_size = min(int(mini_batch_size * 1.0005), y.shape[0])
        learning_rate *= 0.9999
      ploss = loss # Set the previous loss

      # Update gradients
      if update_strategy == 'fixed':
        for p, w in self.params.items():
          dw = grad[p]
          next_w = w - learning_rate * dw
          self.params[p] = next_w
      elif update_strategy == 'rmsprop':
        for p, w in self.params.items():
          dw = grad[p]
          if p not in rmsprop_cache:
            rmsprop_cache[p] = dw**2
          else:
            rmsprop_cache[p] = decay_rate * rmsprop_cache[p] + (1 - decay_rate) * dw**2
          next_w = w - learning_rate * dw / (np.sqrt(rmsprop_cache[p]) + eps)
          self.params[p] = next_w

      if verbose:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    return self.loss(X).argmax(axis=1)
