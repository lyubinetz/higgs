from base_methods_wrappers import *
from implementations import *
from featurization import *

def create_method_params(method_name, shape=30):
  '''
  Creates a dictionary with parameters for the method defined in implementation.py specified by method_name
  :param method_name: values set: linear_regression, ridge_regression, reg_log_reg (i.e. regularized logistic
  regression)
  :param shape: number of features in the dataset
  :return: dictionary with parameter_name:value
  '''
  if method_name == 'linear_regression':
    method_params = {}
  elif method_name == 'ridge_regression':
    method_params = {
      'lambda_':0.01
    }
  elif method_name == 'reg_log_reg':
    method_params = {
      'lambda_':0.01,
      'initial_w':np.zeros((shape, 1)),
      'max_iters':1000,
      'gamma':0.000001
    }
  else:
    raise Exception('Wrong method name in baseline evaluation')

  return method_params


def get_method(method_name):
  '''
  Simply returns the method defined in implmenetations.py with the correspoding name
  :param method_name:  values set: linear_regression, ridge_regression, reg_log_reg (regularized logistic regression)
  :return: method
  '''
  if method_name == 'linear_regression':
    return least_squares
  if method_name == 'ridge_regression':
    return ridge_regression
  if method_name == 'reg_log_reg':
    return reg_logistic_regression
  raise Exception("Wrong method name in get_method baseline evaluation")


def evaluate(X, y, method_name, log_file_name ="logs.tx", verbose=False):
  '''
  Evaluates the the result of the learning method specified by method_name over a 3-fold cross-validation.
  The evaluation result is appended in the log_file_name file
  :param X: train data
  :param y: train labels
  :param method_name: method to evaluate
  :param log_file_name: file where to append the evaluation result
  :param verbose:
  :return: Nothing
  '''
  if verbose:
    print('Started evaluation run for {}'.format(method_name))

  learn_method_params = create_method_params(method_name, X.shape[1])
  if verbose:
    print('Learn parameters {}'.format(learn_method_params))

  wrapper_constructor_params = {
    'method':get_method(method_name),
    'method_params':learn_method_params
  }
  loss_tr, loss_te = cross_validate(
    BaseMethodWrapper,
    y,
    X,
    3,
    lambda y_pred, y_test: (y_pred == y_test).sum()*100/len(y_pred),
    wrapper_constructor_params,
    {},
    777,
    True
  )

  with open(log_file_name, 'a') as f:
    f.write("{} obtained results:\ntrain: {}\ntest: {}\n with params: {}\n input shape: {}".format(method_name,
                                                                                                   loss_tr, loss_te,
                                                                                                   learn_method_params,
                                                                                                   X.shape[1]))
    f.write('\n\n')
    f.flush()

def compute_featurized_data(X_train, X_combined, featurization_method = None):
  if featurization_method is None:
    good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(X_combined)
    X_train = standardize(X_train, good_featurized_means, good_featurized_vars)
  elif featurization_method == 'x^2':
    good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize_x2(X_combined))
    X_train = standardize(featurize_x2(X_train), good_featurized_means, good_featurized_vars)
  elif featurization_method == 'final':
    good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize_before_standardize(X_combined))
    X_train = featurize_and_standardize(X_train, mean=good_featurized_means, var=good_featurized_vars)

  return X_train


if __name__ == '__main__':
  print('Started the run!')
  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  print('Finished loading data!')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  replace_missing_values(X_combined, mean_map)
  replace_missing_values(X_train, mean_map)

  featurization_methods = [None, 'x^2','final']
  learning_methods = ['linear_regression', 'ridge_regression','reg_log_reg']

  for feat_met in featurization_methods:
    X = compute_featurized_data(X_train, X_combined, feat_met)
    for learn_met in learning_methods:
      evaluate(X, y_train, learn_met, 'result.txt', verbose=True)