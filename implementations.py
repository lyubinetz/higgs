import numpy as np
from helpers import batch_iter


def compute_mse_loss(y, tx, w):
  '''
  Calculates the MSE loss.
  '''
  return np.power(y - tx.dot(w), 2).sum() / (2 * y.shape[0])

def compute_mean_squares_gradient(y, tx, w):
  '''Calculate the gradient of the the mean squared error function with respect to w
  Parameters:
      y = labels, numpy column vector
      tx = numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
      w = numpy column vector, weights of the model

  Return:
      Gradient value computed as -1/len(y) * dot(tx.T, e) where e = y - dot(tx, w)
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
      gradient = compute_mean_squares_gradient(y, tx, w)
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
      gradient = compute_mean_squares_gradient(minibatch_y, minibatch_x, w)
      # update parameters
      w = w - gamma * gradient

  return (w, loss)


def least_squares(y, tx):
  '''
  Calculate the least squares solution using normal equations.
  '''
  w = np.linalg.inv((tx.T.dot(tx))).dot(tx.T).dot(y)
  mse_loss = compute_mse_loss(y, tx, w)
  return (w, mse_loss)

def ridge_regression(y, tx, lambda_):
  '''
  Calculate ridge regression solution using normal equations.
  '''
  xtx = tx.T.dot(tx) # Product of tx and its transpose
  w = np.linalg.inv((xtx + lambda_ * np.identity(xtx.shape[0]))).dot(tx.T).dot(y)
  mse_ridge = compute_mse_loss(y, tx, w) + lambda_ * (w * w).sum()
  return (w, mse_ridge)

def linear_predict(w, tx):
  return tx.dot(w)

##################### Logistic regression ##############################

def sigmoid_element(x):
  '''
  Function that applies the sigmoid function to a number, taking into consideration
  the fact that the exponential grows very fast. Therefore, if x < 0, then 
  the expression of the function will be \frac{e^x}{e^x+1}, making sure that the 
  numerator is upper bounded by 1 and denominator by 2. Also, if x > 0,
  then we express the sigmoid function as \frac{1}{1+e^{-x}}, the numerator being 1, 
  and the denominator is upper bounded by 2. Note that both expressions used are 
  equivalent.
  :param x: the element on which we want to apply the sigmoid function
  :return: \sigma(x)
  '''
  if x <= 0:
    return np.exp(x) / (np.exp(x) + 1)
  return 1 / (1 + np.exp(-x))


def sigmoid(t):
  '''
  Sigmoid function, that can be applied to a number as well as a numpy array.
  :param t: a number or a numpy array.
  :return: the sigmoid function applied to the parameter t. If t is a vector, then it is 
  applied element-wise
  '''
  vectorized_sigmoid = np.vectorize(sigmoid_element)
  return vectorized_sigmoid(t)


def calculate_logistic_regression_loss(y, tx, w):
  '''
  Function that computes the loss for logistic regression, as described in the course.

  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :return: the loss of the current model, as a number
  '''

  return np.sum(np.log(1 + np.exp(np.dot(tx, w))) - y * tx.dot(w))

def calculate_logistic_regression_gradient(y, tx, w):
  '''
  Function that calculates the gradient of the logistic regression loss, as presented 
  in course.
  
  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :return: gradient of the logistic regression loss, numpy column vector
  '''
  return tx.T @ (sigmoid(np.dot(tx, w)) - y)

def logistic_regression_learning_by_gradient_descent(y, tx, w, gamma):
  '''
  Function that performs one step of the Gradient Descent in the Logistic Regression 
  case. 
  
  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :param gamma: the learning step
  :return: (loss, w), where the first position represents the loss of the current model 
  defined by w, and the second position represents the vector w defining the new model
  '''
  loss = calculate_logistic_regression_loss(y, tx, w)
  gradient = calculate_logistic_regression_gradient(y, tx, w)
  w -= gamma * gradient
  return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
  '''
  Function that implements Logistic Regression using Gradient Descent method.
  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param initial_w: numpy column vector, weights of the model
  :param max_iters: number of steps for the stochastic gradient descent method. Must be >0 to return meaningful loss
  :param gamma: the learning step
  :return: (w, loss), where the first position represents the vector w defining the trained
   model, and the second position represents the loss of the resulted model.
  '''

  assert max_iters > 0

  y = y.reshape(y.shape[0], 1) # Making sure y is a 2D column array
  w = initial_w
  for iteration in range(max_iters):
    # performing one step of the algorithm
    loss, w = logistic_regression_learning_by_gradient_descent(y, tx, w, gamma)
    if iteration % 100 == 0:
      print("Iteration {it}, loss: {l}".format(it=iteration, l=loss))
  return w, loss

#### Regularized logistic regression

def compute_L2_regularizer(lambda_, w):
  '''
  Function that computes the value lambda_/2 * ||w||^2.
  :param lambda_: the regularization parameter
  :param w: the current weights vector
  :return: the value of the regularizer
  '''
  return 1.0 * lambda_ / 2 * np.sum(w ** 2)


def calculate_reg_logistic_regression_loss(y, tx, w, lambda_):
  '''
  Function that computes the loss for regularized logistic regression, as described in the course.

  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :param lambda_: the regularization parameter
  :return: the loss of the current model, as a number
  '''
  return calculate_logistic_regression_loss(y, tx, w) + compute_L2_regularizer(lambda_, w)


def calculate_reg_logistic_regression_gradient(y, tx, w, lambda_):
  '''
  Function that calculates the gradient of the regularized logistic regression loss, 
  as presented in course.

  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :param lambda_: the regularization parameter
  :return: gradient of the logistic regression loss, numpy column vector
  '''

  # gradient of ||w||_2^2 w.r.t. w is 2w, so we can only add the gradient of the regularizer
  return calculate_logistic_regression_gradient(y, tx, w) + lambda_ * w


def reg_logistic_regression_learning_by_gradient_descent(y, tx, w, gamma, lambda_):
  '''
  Function that performs one step of the Gradient Descent in the Regularized Logistic 
  Regression case. 

  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param w: numpy column vector, weights of the model
  :param gamma: the learning step
  :param lambda_: the regularization parameter
  :return: (loss, w), where the first position represents the loss of the current model 
  defined by w, and the second position represents the vector w defining the new model
  '''
  loss = calculate_reg_logistic_regression_loss(y, tx, w, lambda_)
  gradient = calculate_reg_logistic_regression_gradient(y, tx, w, lambda_)
  w -= gamma * gradient
  return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  '''
  Function that implements Logistic Regression using Gradient Descent method.
  :param y: labels, numpy column vector
  :param tx: numpy multidimensional array, data in matrix form (with first column = 1 for bias), one data entry per row
  :param initial_w: numpy column vector, weights of the model
  :param max_iters: number of steps for the stochastic gradient descent method. Must be >0 to return meaningful loss
  :param gamma: the learning step
  :return: (w, loss), where the first position represents the vector w defining the trained
   model, and the second position represents the loss of the resulted model.
  '''

  assert max_iters > 0

  y = y.reshape(y.shape[0], 1)  # Making sure y is a 2D column array

  w = initial_w
  for iteration in range(max_iters):
    # performing one step of the algorithm
    loss, w = reg_logistic_regression_learning_by_gradient_descent(y, tx, w, gamma, lambda_)
    if iteration % 100 == 0:
      print("Iteration {it}, loss: {l}".format(it=iteration, l=loss))
  return w, loss

