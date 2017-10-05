import numpy as np
import csv

def load_csv_data(data_path, sub_sample=False):
  '''Loads data and returns y (class labels), tX (features) and ids (event ids)'''
  y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
  x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
  ids = x[:, 0].astype(np.int)
  input_data = x[:, 2:]

  # convert class labels from strings to binary (0,1)
  yb = np.ones(len(y)).astype(np.int)
  yb[np.where(y=='b')] = 0 # Note, and this is important - we use 0 instead of -1
  
  # sub-sample
  if sub_sample:
    yb = yb[::50]
    input_data = input_data[::50]
    ids = ids[::50]

  return yb, input_data, ids

def read_train_data(fname, sub_sample=False):
  '''
  This function reads training data from CSV file.
  '''
  y, X, ids = load_csv_data(fname, sub_sample)
  return X, y

def read_test_data(fname, sub_sample=False):
  '''
  This function reads test data from CSV file.
  '''
  y, X, ids = load_csv_data(fname, sub_sample)
  return X, ids

def create_csv_submission(ids, y_pred, name):
  '''
  Creates an output file in csv format for submission to kaggle
  Arguments: ids (event ids associated with each prediction)
             y_pred (predicted class labels)
             name (string name of .csv output file to be created)
  '''
  with open(name, 'w') as csvfile:
    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1, r2 in zip(ids, y_pred):
      writer.writerow({'Id':int(r1),'Prediction':int(r2)})

'''
Makes each column to be Gaussian(0, 1)
'''
def standardize(x, mean=None, var=None):
  if mean is None:
    mean = np.mean(x, axis=0)

  if var is None:
    var = np.std(cd, axis=0)

  cd = x - mean
  std_data = cd / var

  return std_data

'''
Computes means for each column, while skipping invalid values.
'''
def compute_means_and_vars_for_columns(data):
  mean_map = np.zeros(data.shape[1])
  var_map = np.zeros(data.shape[1])

  for i in range(data.shape[1]):
    good_positions = np.where(data[:, i] > -999)    
    col = data[:, i][good_positions]
    mean_map[i] = np.mean(col)
    var_map[i] = np.std(col)

  return mean_map, var_map

'''
Replaces -999 in the given data with means computed by compute_means_for_columns()
Mutates data.
'''
def replace_missing_values_with_means(data, mean_map):
  for i in range(data.shape[1]):
    bad_positions = np.where(data[:, i] <= -999)
    data[:, i][bad_positions] = mean_map[i]

'''
Splits data into two halfs with frac being the fraction of the data to go into the first
half. Frac must be between 0 and 1. Optionally takes labels as well, and splits them
identically to data.
'''
def split_data(frac, data, labels):
  if frac < 0 or frac > 1:
    raise Exception('Illegal frac value in split_data!')

  n = data.shape[0]
  indices = np.random.choice(n, int(n * frac))
  indices_set = set(indices)
  not_in_indices = [x for x in range(n) if x not in indices_set]

  return data[indices], labels[indices], data[not_in_indices], labels[not_in_indices]

def stack_ones(x):
  return np.c_[np.ones(x.shape[0]), x]

def predict_labels(weights, data):
  '''
  Generates class predictions given weights, and a test data matrix
  '''
  y_pred = np.dot(data, weights)
  y_pred[np.where(y_pred <= 0.5)] = 0 # Note that this differs from what was given in github - we use 0
  y_pred[np.where(y_pred > 0.5)] = 1

  return y_pred
