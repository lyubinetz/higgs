import numpy as np
import pandas as pd

'''
This function reads training data from CSV file in a given location into a pandas datafrmame.
'''
def read_train_data(fname):
  data = pd.read_csv(fname, sep=',')
  X = data.drop('Prediction', axis=1).drop('Id', axis=1)
  y = data['Prediction']
  replace_classes = lambda x: 0 if x == 'b' else 1
  y = y.map(replace_classes)
  return np.array(X), np.array(y)

'''
This function reads test data from CSV file in a given location into a pandas datafrmame.
'''
def read_test_data(fname):
  data = pd.read_csv(fname, sep=',')
  X = data.drop('Prediction', axis=1).drop('Id', axis=1)
  ids = data['Id']
  return np.array(X), np.array(ids)

'''
Makes each column to be Gaussian(0, 1)
'''
def standardize(x):
  cd = x - np.mean(x, axis=0)
  std_data = cd / np.std(cd, axis=0)  
  return std_data

'''
Computes means for each column, while skipping invalid values.
'''
def compute_means_for_columns(data):
  mean_map = {}
  for i in range(data.shape[1]):
    good_positions = np.where(data[:, i] > -999)    
    col = data[:, i][good_positions]
    mean_map[i] = np.mean(col)

  return mean_map

'''
Replaces -999 in the given data with means computed by compute_means_for_columns()
Mutates data.
'''
def replace_missing_values_with_means(data, mean_map):
  for i in range(data.shape[1]):
    bad_positions = np.where(data[:, i] <= -999)
    data[:, i][bad_positions] = mean_map[i]
