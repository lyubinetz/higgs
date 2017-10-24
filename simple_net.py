import numpy as np

'''
This file contains functions for the neural network implementation.
It is based on cs231n implementation that Volodymyr wrote.

Some details of the implementation:

- Optimization is a simple GD, but the nice part is a decaying learning rate. Whenever
  we encounter that the loss is worse than before, we probably went the wrong way, and
  LR is multiplied by <1 factor. This seems to do really well in practice.
- We only use a ReLU activation function.
- I adapted the API to mimic scipy's with fit() and predict().
'''

class SimpleNet(object):

  def __init__(self, matrix_dims, weight_magnitude=1e-1, reg=0.01, input_size=30):
    '''
    input_size - dimension of input data
    matrix_dims - dimensions of matrices used in NN, aka layer sizes
    weight_magnitude - we initialize matrice with Gaussian(0, 1) multiplied by this magnitude
    reg - regularization strength
    '''
    # In this task we only deal with 2 classes
    classes = 2

    self.reg = reg
    self.params = {}
    self.num_layers = len(matrix_dims) + 1
    all_md = [input_size] + matrix_dims + [classes]

    for i in range(self.num_layers):
      self.params['W' + str(i)] = weight_magnitude * np.random.randn(all_md[i], all_md[i + 1]) / np.sqrt(all_md[i])
      self.params['b' + str(i)] = np.zeros(all_md[i + 1])

  def loss(self, X, y=None):
    '''
    Compute loss and gradients.
    '''
    # Compute the forward pass
    M = {} # Results of applying affine layer + ReLU
    for i in range(self.num_layers - 1):
      prev = X
      if i > 0:
        prev = M[i - 1]
      Wi, bi = self.params['W' + str(i)], self.params['b' + str(i)]
      Mi = prev.dot(Wi) + bi
      Mi = np.maximum(Mi, np.zeros(Mi.shape))
      M[i] = Mi.copy()

    Wlast, blast = self.params['W' + str(self.num_layers - 1)], self.params['b' + str(self.num_layers - 1)]
    scores = Mi.dot(Wlast) + blast
    
    # If we just need the predictions
    if y is None:
      return scores

    # Compute softmax loss with L2 regularization
    # Disclosure: My implementation used a loop when I implemented this and I
    # looked up how people do it with crazy arrange() usage
    shifted_by_max = scores - np.max(scores, axis=1, keepdims=True) # This is done for numerical stability
    Z = np.sum(np.exp(shifted_by_max), axis=1, keepdims=True)
    log_probs = shifted_by_max - np.log(Z)
    probs = np.exp(log_probs)
    loss = -np.sum(log_probs[np.arange(scores.shape[0]), y]) / scores.shape[0]

    # Add regularization
    for i in range(self.num_layers):
      Wi = self.params['W' + str(i)]
      loss += self.reg * (Wi * Wi).sum()

    # Backward pass: compute gradients, storing them in the grads dictionary
    grads = {}

    # probs is sort of the gradient
    probs[range(X.shape[0]), y] -= 1
    probs /= X.shape[0]

    grads['W' + str(self.num_layers - 1)] = M[self.num_layers - 2].T.dot(probs) + 2 * self.reg * self.params['W' + str(self.num_layers - 1)]
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


  def _get_batch_indices(self, y, batch_size, class_ratio=None):
    '''
    Compute mini-batch indices so that we can use them easily to get the mini-batch data
    :param y: labels vector, needed to compute indices with the given class ratio
    :param batch_size: number of data entries the mini batch should have
    :param class_ratio: proportion of data corresponding to class 0 in the resulting mini-batch. If None,
    same proportion as in input data is kept
    :return: array of indices of size batch_size corresponding to the mini-batch entries that should be used
    '''
    # compute the indices of entries for each class label
    indices_y0 = np.nonzero(y == 0)[0]
    indices_y1 = np.nonzero(y == 1)[0]

    # we keep the same class ratio as in the data
    if class_ratio is None:
      class_ratio = indices_y0.shape[0] * 1.0 / y.shape[0]

    shuffled_indices_y0 = np.random.permutation(len(indices_y0))
    shuffled_indices_y1 = np.random.permutation(len(indices_y1))

    class_0_batch_size = int(np.floor(class_ratio * batch_size))
    class_1_batch_size = int(np.ceil((1 - class_ratio) * batch_size))

    if class_1_batch_size > len(indices_y1):
      class_1_batch_size = len(indices_y1)

    if class_0_batch_size > len(indices_y0):
      class_0_batch_size = len(indices_y0)

    mini_batch_indices = np.r_[indices_y0[shuffled_indices_y0[:class_0_batch_size]],
                         indices_y1[shuffled_indices_y1[:class_1_batch_size]]]

    return mini_batch_indices

  def fit(self, X, y, learning_rate=0.1, num_iters=1000, verbose=False, update_strategy='rmsprop', decay_rate=0.9,
          optimization_strategy = 'gd', mini_batch_size=10000, mini_batch_class_ratio=None):
    '''
    Trains the neural network on dataset (X, y).
    - update_strategy is one of ('fixed', 'decrease_on_mistake', 'rmsprop')
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
        batch_indexes = self._get_batch_indices(y, mini_batch_size, mini_batch_class_ratio)
        #batch_indexes = np.random.choice(len(y), mini_batch_size, replace=False)
        loss, grad = self.loss(X[batch_indexes], y[batch_indexes])
      loss_history.append(loss)

      if loss > ploss: # update_strategy == 'decrease_on_mistake' and
        if optimization_strategy =='gd':
          # Decrease LR so that we take smaller steps
          learning_rate *= 0.8
        elif optimization_strategy == 'sgd':
          # When using SGD, increase the batch-size for a more stable loss and gradient and decrease the learning
          # rate by a lower value
          mini_batch_size = min(int(mini_batch_size * 1.0005), y.shape[0])
          learning_rate *= 0.999
      ploss = loss

      # Update gradients
      if update_strategy in ['fixed', 'decrease_on_mistake']:
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
