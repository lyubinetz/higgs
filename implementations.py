import numpy as np

def compute_mse_loss(y, tx, w):
    '''
    Calculates the MSE loss.
    '''
    return np.power(y - tx.dot(w), 2).sum() / (2 * y.shape[0])

def least_squares(y, tx):
  '''
  Calculate the least squares solution using normal equations.
  '''
  w = np.linalg.inv((tx.T.dot(tx))).dot(tx.T).dot(y)
  mse_loss = compute_mse_loss(y, tx, w)
  return w, mse_loss

def ridge_regression(y, tx, lambda_):
  '''
  Calculate ridge regression solution using normal euqations.
  '''
  xtx = tx.T.dot(tx) # Product of tx and its transpose
  w = np.linalg.inv((xtx + lambda_ * np.identity(xtx.shape[0]))).dot(tx.T).dot(y)
  mse_ridge = compute_mse_loss(y, tx, w) + lambda_ * (w * w).sum()
  return w, mse_ridge
