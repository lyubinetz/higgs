import numpy as numpy
from helpers import *
from simple_net import *
from neural_network import *
from featurization import *

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''
def run(validation, classify_test):
  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  replace_missing_values_with_means(X_train, mean_map)
  X_train = featurize_x2(X_train)
  X_train = standardize(X_train)

  if validation:
    X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
    print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  nn = SimpleNet([500, 500], reg=0.001, input_size=X_train.shape[1])
  # Train the net
  nn.fit(X_train, y_train, verbose=True, num_iters=100, learning_rate=2)

  # Compute validation score
  if validation:
    y_pred_val = nn.predict(X_val)
    num_correct = (y_pred_val == y_val).sum()
    print('Validation results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  if classify_test:
    # Compute result for submission
    replace_missing_values_with_means(X_test, mean_map)
    X_test = featurize_x2(X_test)
    X_test = standardize(X_test)
    test_predictions = nn.predict(X_test)

    # HACK: Right now predictions are 0,1 , and we need -1,1
    test_predictions = 2 * test_predictions
    test_predictions = test_predictions - 1

    create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')

if __name__ == '__main__':
  np.random.seed(777)
  run(True, False)
