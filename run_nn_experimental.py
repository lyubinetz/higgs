import threading
import numpy as numpy
from helpers import *
from simple_net import *
from neural_network import *
from featurization import *

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''
def run():
  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  X_train_good, y_train_good, X_train_bad, y_train_bad = split_into_full_and_missing(X_train, y_train)
  replace_missing_values(X_train_bad, mean_map)

  # Compute featurzied means
  replace_missing_values(X_combined, mean_map)
  good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize(X_combined))

  X_train_good = featurize(X_train_good)
  X_train_good = standardize(X_train_good, mean=good_featurized_means, var=good_featurized_vars)

  X_train_bad = featurize(X_train_bad)
  X_train_bad = standardize(X_train_bad, mean=good_featurized_means, var=good_featurized_vars)

  nn1 = SimpleNet([300], reg=0.001, input_size=X_train_good.shape[1])
  #nn1.fit(X_train_good, y_train_good, verbose=True, num_iters=50, learning_rate=0.01, update_strategy='rmsprop')

  nn2 = SimpleNet([300], reg=0.001, input_size=X_train_bad.shape[1])
  #nn2.fit(X_train_bad, y_train_bad, verbose=True, num_iters=50, learning_rate=0.01, update_strategy='rmsprop')

  t1 = threading.Thread(target = nn1.fit, args = (X_train_good, y_train_good, 0.01, 50, True, 'rmsprop', 0.9))
  t1.start()

  t2 = threading.Thread(target = nn2.fit, args = (X_train_bad, y_train_bad, 0.01, 50, True, 'rmsprop', 0.9))
  t2.start()

  t1.join()
  t2.join()

  # Compute result for submission
  X_test_good, X_good_ids, X_test_bad, X_bad_ids = split_into_full_and_missing(X_test, X_test_ids)

  replace_missing_values(X_test_bad, mean_map)

  X_test_good = featurize(X_test_good)
  X_test_good = standardize(X_test_good, mean=good_featurized_means, var=good_featurized_vars)

  X_test_bad = featurize(X_test_bad)
  X_test_bad = standardize(X_test_bad, mean=good_featurized_means, var=good_featurized_vars)

  test_predictions_1 = nn1.predict(X_test_good)
  test_predictions_2 = nn2.predict(X_test_bad)

  test_predictions = np.concatenate((test_predictions_1, test_predictions_2), axis=0)
  X_test_ids = np.concatenate((X_good_ids, X_bad_ids), axis=0)

  # HACK: Right now predictions are 0,1 , and we need -1,1
  test_predictions = 2 * test_predictions
  test_predictions = test_predictions - 1

  create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')

if __name__ == '__main__':
  np.random.seed(777)
  run()
