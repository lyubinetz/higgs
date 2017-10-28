import numpy as np

class BaseMethodWrapper():
  def __init__(self, method, method_params):
    self._fit = method
    self._fit_params = method_params
    self._w = None
    self._loss = None

  def fit(self, y, tx):
    self._w, self._loss = self._fit(y, tx, **self._fit_params)

  def predict(self, data, threshold=0.5):
    y_pred = np.dot(data, self._w)
    y_pred = y_pred.reshape((y_pred.shape[0]))

    y_pred[np.where(y_pred <= threshold)] = 0 # Note that this differs from what was given in github - we use 0
    y_pred[np.where(y_pred > threshold)] = 1

    return y_pred
