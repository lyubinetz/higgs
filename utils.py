import numpy as np
from helpers import *
from simple_net import *
from neural_network import *
from featurization import *
from implementations import *

def compare_results(y_pred1, y_pred2, y_correct):
  '''
  This function compares two predictions - depending on how
  similar or 
  '''
  sz = len(y_correct)
  both_correct = ((y_pred1 == y_correct) * (y_pred2 == y_correct)).sum()
  none_correct = ((y_pred1 != y_correct) * (y_pred2 != y_correct)).sum()
  first_correct = ((y_pred1 == y_correct) * (y_pred2 != y_correct)).sum()
  second_correct = ((y_pred1 != y_correct) * (y_pred2 == y_correct)).sum()

  print('%% of both correct: ' + str(both_correct * 1.0 / sz))
  print('%% of none correct: ' + str(none_correct * 1.0 / sz))
  print('%% of 1st correct, 2nd wrong: ' + str(first_correct * 1.0 / sz))
  print('%% of 2nd correct, 1st wrong: ' + str(second_correct * 1.0 / sz))

if __name__ == '__main__':
  X_train, y_train = read_train_data('datasets/train.csv')

  mean_map, var_map = compute_means_and_vars_for_columns(X_train)

  replace_missing_values(X_train, mean_map)
  X_train = featurize_x2(X_train)
  X_train = standardize(X_train)

  X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
  print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  #nn1 = SimpleNet([500, 500], reg=0.001, input_size=X_train.shape[1])
  #nn1.fit(X_train, y_train, verbose=True, num_iters=50, learning_rate=2)
  w1, _ = least_squares(y_train, stack_ones(X_train))

  nn2 = SimpleNet([500], reg=0.001, input_size=X_train.shape[1])
  nn2.fit(X_train, y_train, verbose=True, num_iters=50, learning_rate=2)

  y_pred_val_1 = predict_labels(w1, stack_ones(X_val))
  y_pred_val_2 = nn2.predict(X_val)

  compare_results(y_pred_val_1, y_pred_val_2, y_val)
