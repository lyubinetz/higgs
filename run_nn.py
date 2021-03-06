import numpy as numpy
from helpers import *
from simple_net import *
from featurization import *

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''
def run(validation, classify_test):
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

  if validation:
    X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
    print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  nn = SimpleNet([600, 600], reg=0.0001, input_size=X_train.shape[1])
  # Train the net
  nn.fit(X_train, y_train, verbose=True, num_iters=4000, learning_rate=0.01, update_strategy='rmsprop',
    optimization_strategy='sgd', mini_batch_size=600, lr_decay=0.9995)

  y_pred_val = nn.predict(X_train)
  num_correct = (y_pred_val == y_train).sum()
  print('Train results ' + str(num_correct) + ' out of ' +
    str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  # Compute validation score
  if validation:
    y_pred_val = nn.predict(X_val)
    num_correct = (y_pred_val == y_val).sum()
    print('Validation results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  if classify_test:
    # Compute result for submission
    replace_missing_values(X_test, mean_map)
    X_test = featurize_and_standardize(X_test, mean=good_featurized_means, var=good_featurized_vars)
    test_predictions = nn.predict(X_test)

    # HACK: Right now predictions are 0,1 , and we need -1,1
    test_predictions = 2 * test_predictions
    test_predictions = test_predictions - 1

    create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')

def run_cv():
  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test_sample.csv')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  # _, _, X_train, y_train = split_into_full_and_missing(X_train, y_train)
  replace_missing_values(X_train, mean_map)

  X_train = featurize_and_standardize(X_train)

  k_folds = 3
  fit_params_dict = {'verbose':True, 'num_iters':4000, 'learning_rate':0.01, 'update_strategy':'rmsprop',
                     'optimization_strategy':'sgd', 'mini_batch_size':1000}
  constructor_params_dict = {'matrix_dims':[200,200,200,200], 'reg':0.0001, 'input_size':X_train.shape[1]}
  loss_tr, loss_te = cross_validate(SimpleNet,
                                    y_train,
                                    X_train,
                                    k_folds,
                                    lambda y_test, y_pred : (y_pred == y_test).sum()*100/len(y_pred),
                                    constructor_params_dict,
                                    fit_params_dict,
                                    777,
                                    verbose=True)

  print('Mean train error over {} folds is {}'.format(k_folds, loss_tr))
  print('Mean test error over {} folds is {}'.format(k_folds, loss_te))

if __name__ == '__main__':
  np.random.seed(777)
  run(True, False)
  #run_cv()
