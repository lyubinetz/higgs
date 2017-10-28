import numpy as np
from helpers import *
from simple_net import *
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

  Xt1 = X_train[:,:14]
  Xt2 = X_train[:,15:]

  Xt1 = featurize_x2(Xt1)
  Xt1 = standardize(Xt1)

  Xt2 = featurize_x2(Xt2)
  Xt2 = standardize(Xt2)

  indices = np.random.choice(len(y_train), int(len(y_train) * 0.8), replace=False)
  indices_set = set(indices)
  not_in_indices = [x for x in range(len(y_train)) if x not in indices_set]

  Xt1, Xv1 = Xt1[indices], Xt1[not_in_indices]
  Xt2, Xv2 = Xt2[indices], Xt2[not_in_indices]

  y_train, y_val = y_train[indices], y_train[not_in_indices]

  nn1 = SimpleNet([600, 600], reg=0.00001, input_size=Xt1.shape[1])
  nn1.fit(Xt1, y_train, verbose=True, num_iters=500, learning_rate=0.01, update_strategy='rmsprop',
    optimization_strategy='sgd', mini_batch_size=600)

  nn2 = SimpleNet([600, 600], reg=0.00001, input_size=Xt2.shape[1])
  nn2.fit(Xt2, y_train, verbose=True, num_iters=500, learning_rate=0.01, update_strategy='rmsprop',
    optimization_strategy='sgd', mini_batch_size=600)

  y_pred_val_1 = nn1.predict(Xv1)
  y_pred_val_2 = nn2.predict(Xv2)

  compare_results(y_pred_val_1, y_pred_val_2, y_val)
