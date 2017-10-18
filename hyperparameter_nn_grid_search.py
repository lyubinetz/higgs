import numpy as numpy
from helpers import *
from simple_net import *
from featurization import *
import itertools

def grid_search(LearnerClass,X_train, y_train, X_val, y_val, constructor_params_dict, fit_params_dict):
  '''
    Basic implementation for NN hyperparameters grid-search
    :param LearnerClass: Class that implements fit and predict method. The constructor and fit method must accept
    the parameters in the constructor_params_dict and fit_params_dict
    :param X_train: training data
    :param y_train: labels vector for training data
    :param X_val: validation data
    :param y_val: labels vector for validation data
    :param constructor_params_dict: dictionary of "param name":[values] which specifies for each constructor argument
    the values set over which we perform the grid search.
    :param fit_params_dict: dictionary of "param name":[values] which specifies for each fit method argument the
    values set over which we perform the grid search.
    :return: A tuple with (best_accuracy, [best_constructor_params, best_fit_params])
    '''
  best_correct = 0
  best_params = []

  # iterate over all possible combinations of constructor parameters values
  for constructor_values in itertools.product(*constructor_params_dict.values()):
    # construct the constructor arguments as a dictionary of param name and value
    constructor_dict = dict(zip(constructor_params_dict.keys(), constructor_values))
    nn = LearnerClass(**constructor_dict)
    # iterate over all possible combinations of fit parameters values
    for fit_values in itertools.product(*fit_params_dict.values()):
      # construct the fit method arguments as a dictionary of param name and value
      fit_dict = dict(zip(fit_params_dict.keys(), fit_values))
      nn.fit(X_train, y_train, **fit_dict)

      y_pred_val = nn.predict(X_train)
      num_correct = (y_pred_val == y_train).sum()

      print('Train results ' + str(num_correct) + ' out of ' +
            str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

      y_pred_val = nn.predict(X_val)
      num_correct = (y_pred_val == y_val).sum()
      print('Validation results ' + str(num_correct) + ' out of ' +
            str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

      if num_correct > best_correct:
        best_correct = num_correct
        best_params = [constructor_dict, fit_dict]

  print('Best result of {} was obtained with params'.format(best_correct/y.shape[0]))
  print('Constructor params:', best_params[0])
  print('Fit params:', best_params[1])
  return best_correct/y.shape[0], best_params


if __name__ == '__main__':
  print('Started the run!')
  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  print('Finished loading data!')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  # Compute featurzied means
  replace_missing_values(X_combined, mean_map)
  good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize_before_standardize(X_combined))

  replace_missing_values(X_train, mean_map)

  X_train = featurize_and_standardize(X_train, mean=good_featurized_means, var=good_featurized_vars)

  print('New number of features is ' + str(X_train.shape[1]))
  print('Finished data ops!')

  X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
  print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  constructor_params_values = {
    'matrix_dims' : [[300], [300, 300], [300, 300, 300]],
    'reg' : [0],
    'input_size' : [X_train.shape[1]]
  }

  fit_params_values = {
    'verbose' : [True],
    'num_iters' :[800, 1200],
    'learning_rate' : [0.001, 0.01, 0.1],
    'update_strategy' : ['rmsprop', 'fixed'],
    'optimization_strategy' : ['sgd'],
    'mini_batch_size' : [5000, 10000, 15000],
    'mini_batch_class_ratio' : [0.5, None]
  }
  np.random.seed(777)
  grid_search(SimpleNet, X_train, y_train, X_val, y_val, constructor_params_values, fit_params_values)
