import numpy as np
from helpers import batch_iter


def compute_mse_loss(y, tx, w):
  '''
  Calculates the MSE loss.
  '''
  return np.power(y - tx.dot(w), 2).sum() / (2 * y.shape[0])

def compute_mean_squared_gradient(y, tx, w):
  '''Calculate the gradient of the the mean squared error function with respect to w
  Parameters:
      y = labels, numpy column vector
      tx = numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
      w = numpy column vector, weights of the model

  Return:
      Gradient value copmuted as -1/len(y) * dot(tx.T, e) where e = y - dot(tx, w)

  Obs:
      <x,y> = inner product of vectors x and y
  '''
  e = y - tx.dot(w)
  return -tx.T.dot(e)/y.shape[0]

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  '''Apply gradient descent method to minimize mean squared error function
  for labels y and training data tx.

  Paramters:
    y = labels, numpy column vector
    tx = data in matrix form (with first column = 1 for bias), one data entry per row,
    numpy multidimensional array
    initial_w =  initial values for the weights, numpy column vector
    max_iters = number of steps for the gradient descent method
          must be >0 to return meaningful loss
    gamma = learning step

  Returns the weights corresponding to the last step'''

  assert max_iters > 0

  w = initial_w
  loss = None
  for n_iter in range(max_iters):
      # compute loss and gradient
      loss = compute_mse_loss(y, tx, w)
      gradient = compute_mean_squared_gradient(y, tx, w)
      # update parameters
      w = w - gamma * gradient

  return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
  '''Apply stochastic gradient descent method to minimize mean squared loss function
  for labels y and training data tx

  Parameters:
    y = labels, numpy column vector
    tx = data in matrix form (with first column = 1 for bias), one data entry per row
    numpy multidimensional array,
    initial_w = initial values for the weights, numpy column vector
    max_iters = number of steps for the stochastic gradient descent method
      must be >0 to return meaningful loss
    gamma = learning step

  Returns the weights corresponding to the last step
  '''

  assert max_iters > 0

  w = initial_w
  for i in range(max_iters):
    for minibatch_y, minibatch_x in batch_iter(y, tx, batch_size=1, num_batches=1):
      # compute loss and gradient
      loss = compute_mse_loss(y, tx, w)
      gradient = compute_mean_squared_gradient(minibatch_y, minibatch_x, w)
      # update parameters
      w = w - gamma * gradient

  return (w, loss)


def least_squares(y, tx):
  '''
  Calculate the least squares solution using normal equations.
  '''
  w = np.linalg.inv((tx.T.dot(tx))).dot(tx.T).dot(y)
  mse_loss = compute_mse_loss(y, tx, w)
  return w, mse_loss

def ridge_regression(y, tx, lambda_):
  '''
  Calculate ridge regression solution using normal equations.
  '''
  xtx = tx.T.dot(tx) # Product of tx and its transpose
  w = np.linalg.inv((xtx + lambda_ * np.identity(xtx.shape[0]))).dot(tx.T).dot(y)
  mse_ridge = compute_mse_loss(y, tx, w) + lambda_ * (w * w).sum()
  return w, mse_ridge

def linear_predict(w, tx):
  return tx.dot(w)


