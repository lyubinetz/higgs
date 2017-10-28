import numpy as np

def majority_combine(predictions):
  '''
  Given pairs of (prediction, weight), combines them into one set of predictions
  using majority voting.

  Yay democracy!
  https://vignette.wikia.nocookie.net/polandball/images/4/41/Democracy-and-freedom-from-usaball-white-text-t-shirts-men-s-t-shirt.jpg/revision/latest?cb=20170304131320
  '''
  num_y = len(predictions[0][0])
  res = np.zeros(num_y)
  total_w = 0.0

  for p in predictions:
    (y, w) = p
    res += w * y
    total_w += w

  # Majority voting
  res[np.where(res <= total_w / 2)] = 0
  res[np.where(res > total_w / 2)] = 1

  return res
