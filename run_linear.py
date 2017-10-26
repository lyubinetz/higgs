import numpy as numpy
from helpers import *
from implementations import *
from featurization import *
from base_methods_wrappers import BaseMethodWrapper

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''

def run(validation, classify_test):
  print('Started the run!')
  X_train, y_train = read_train_data('datasets/train.csv', load_pickle=True)
  X_test, X_test_ids = read_test_data('datasets/test.csv', load_pickle=True)

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

  if validation:
    X_train, y_train, X_val, y_val = split_data(0.9, X_train, y_train)

  method_params = {
    'lambda_':0.0001
  }

  learner = BaseMethodWrapper(ridge_regression, method_params)
  learner.fit(y_train, stack_ones(X_train))
  print('Training loss is ' + str(learner._loss))

  # Compute validation score
  if validation:
    y_pred_val = learner.predict(stack_ones(X_val))
    num_correct = (y_pred_val == y_val).sum()
    print('Validation results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  if classify_test:
    # Compute result for submission
    replace_missing_values(X_test, mean_map)
    X_test = standardize(X_test, mean=mean_map, var=var_map)
    test_predictions = predict_labels(w, stack_ones(X_test))

    # HACK: Right now predictions are 0,1 , and we need -1,1
    test_predictions = 2 * test_predictions
    test_predictions = test_predictions - 1

    create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')


def run_cv():
  print('Started the run!')
  X_train, y_train = read_train_data('datasets/train.csv', load_pickle=True)
  X_test, X_test_ids = read_test_data('datasets/test.csv', load_pickle=True)

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

  learn_method_params = {
    'lambda_':0.001
  }

  wrapper_constructor_params = {
    'method':ridge_regression,
    'method_params':learn_method_params
  }

  k_folds = 1
  loss_tr, loss_te = cross_validate(
    BaseMethodWrapper,
    y_train,
    stack_ones(X_train),
    k_folds,
    lambda y_test, y_pred : (y_pred == y_test).sum()*100/len(y_pred),
    wrapper_constructor_params,
    {},
    777,
    verbose=True
  )

  print('Mean train error over {} folds is {}'.format(k_folds, loss_tr))
  print('Mean test error over {} folds is {}'.format(k_folds, loss_te))


if __name__ == '__main__':
  np.random.seed(777)
  #run(True, False)
  run_cv()
