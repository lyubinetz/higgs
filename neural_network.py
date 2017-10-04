import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = x.reshape(x.shape[0], -1).dot(w) + b
  cache = (x, w, b)
  return out, cache

def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None

  orig_x_shape = x.shape
  x_cp = np.array(x, copy=True).reshape(x.shape[0], -1)

  dx = dout.dot(w.T)
  dw = x_cp.T.dot(dout)
  db = dout.sum(axis=0)

  dx = dx.reshape(orig_x_shape)
  return dx, dw, db

def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = np.array(x, copy=True)
  out[out <= 0] = 0
  cache = x
  return out, cache

def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  x = cache
  dx = np.array(dout, copy=True)
  dx[x <= 0] = 0
  return dx

def affine_relu_forward(x, w, b):
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache

def affine_relu_backward(dout, cache):
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db

def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  shifted_logits = x - np.max(x, axis=1, keepdims=True)
  Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
  log_probs = shifted_logits - np.log(Z)
  probs = np.exp(log_probs)
  N = x.shape[0]
  loss = -np.sum(log_probs[np.arange(N), y]) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

class NeuralNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  """

  def __init__(self, hidden_dims, input_dim=30, num_classes=2,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.hidden_dims = hidden_dims
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    for i in range(len(hidden_dims)):
        if i == 0:
            inp_sz = input_dim
        else:
            inp_sz = hidden_dims[i - 1]
        self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, (inp_sz, hidden_dims[i]))
        self.params['b' + str(i + 1)] = np.zeros((hidden_dims[i]))

    self.params['W' + str(len(hidden_dims) + 1)] = np.random.normal(0, weight_scale, (hidden_dims[len(hidden_dims) - 1], num_classes))
    self.params['b' + str(len(hidden_dims) + 1)] = np.zeros((num_classes))

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
        self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
        self.dropout_param['mode'] = mode
    if self.use_batchnorm:
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

    scores = None

    cache = {}
    hidden_dims = self.hidden_dims
    inp = X
    for i in range(len(hidden_dims)):
        inp, cache[i] = affine_relu_forward(inp, self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])
    scores, cache[len(hidden_dims)] = affine_forward(inp, self.params['W' + str(len(hidden_dims) + 1)], self.params['b' + str(len(hidden_dims) + 1)])

    # If test mode return early
    if mode == 'test':
        return scores

    loss, grads = 0.0, {}

    loss, loss_grads = softmax_loss(scores, y)
    sum_of_w = 0
    for i in range(self.num_layers):
        sum_of_w += (self.params['W' + str(i + 1)] * self.params['W' + str(i + 1)]).sum()
    loss += 0.5 * self.reg * sum_of_w

    gradx, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = \
        affine_backward(loss_grads, cache[len(hidden_dims)])
    grads['W' + str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]

    for i in range(len(hidden_dims), 0, -1):
        gradx, grads['W' + str(i)], grads['b' + str(i)] = affine_relu_backward(gradx, cache[i - 1])
        grads['W' + str(i)] += self.reg * self.params['W' + str(i)]

    return loss, grads

  def fit(self, X, y, learning_rate=1e-1, num_iters=100, verbose=False):
    if verbose:
      print('Started fitting the neural network!')

    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    
    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in xrange(num_iters):
      # Evaluate loss and gradient
      loss, grad = self.loss(X, y=y)
      loss_history.append(loss)

      # Update gradients
      for p, w in self.params.items():
        dw = grad[p]
        next_w = w - learning_rate * dw
        self.params[p] = next_w

      if verbose:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    return loss(X)
