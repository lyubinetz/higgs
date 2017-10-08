import numpy as np
from helpers import *
from neural_network import *

'''
This file contains various utilities for creating the best artificial features.
'''

def featurize_x2(data):
  '''
  Adds x^2 features to the data.
  '''
  return np.c_[data, np.power(data, 2)]

def featurize_x2_and_minus(data):
  '''
  Adds x^2 features, and -data to the data.
  '''
  return np.c_[data, np.power(data, 2), data * -1]

def pairwise_feature_search(X_train, y_train, X_val, y_val, num_iter):
  '''
  Find a combination of 40 pairwise product features that produces the
  best result on a 1000-layer network when combined with x^2 featurization.
  '''
  best_score = -1
  best_fs = []

  for i in range(num_iter):
    print('Search iter: ' + str(i))
    pws = []
    for j in range(40):
      v1 = np.random.randint(30)
      v2 = np.random.randint(30)
      while v2 == v1:
        v2 = np.random.randint(30)
      pws.append((v1, v2))

    dat = featurize_with_pairwise(X_train, pws)
    nn = NeuralNet([1000], reg=0.001, input_dim=100)
    # Train the net
    nn.fit(dat, y_train, verbose=True, num_iters=40, learning_rate=2)

    y_pred_val = nn.predict(featurize_with_pairwise(X_val, pws))
    num_correct = (y_pred_val == y_val).sum()
    print('New pws results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

    if num_correct > best_score:
      best_score = num_correct
      best_fs = pws

  print('Bst score ' + str(best_score))
  print('Bst pws ' + str(best_fs))

def featurize_with_pairwise(data, pairs):
  '''
  Extends data with specified pairwise product features.
  '''
  new_data = data
  for p in pairs:
    new_data = np.c_[new_data, data[:,p[0]] * data[:,p[1]]]
  return new_data

if __name__ == '__main__':
  np.random.seed(777)

  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  replace_missing_values_with_means(X_train, mean_map)
  X_train = featurize_x2(X_train)
  X_train = standardize(X_train)
  
  X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
  print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  pairwise_feature_search(X_train, y_train, X_val, y_val, 50)