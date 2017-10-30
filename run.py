import numpy as np
from helpers import *
from simple_net import *
from featurization import *
from majority_combinator import *

# Number of nets to use for bagging
NUM_NETS = 6

def run(verbose_training):
  '''
  Runs the clasification pipeline. In the end this should produce a file
  called prediction.csv with test set classification.
  '''
  print('Started the run!')
  try:
    X_train, y_train = read_train_data('datasets/train.csv')
    X_test, X_test_ids = read_test_data('datasets/test.csv')
  except:
    print('Failed to load datasets! Please make sure that the datasets folder contains train.csv and test.csv !')
    exit(1)
  print('Finished loading data!')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  # Compute featurzied means
  replace_missing_values(X_combined, mean_map)
  good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize_before_standardize(X_combined))

  replace_missing_values(X_train, mean_map)
  replace_missing_values(X_test, mean_map)

  X_train = featurize_and_standardize(X_train, mean=good_featurized_means, var=good_featurized_vars)
  X_test = featurize_and_standardize(X_test, mean=good_featurized_means, var=good_featurized_vars)

  print('New number of features is ' + str(X_train.shape[1]))
  print('Finished featurization pipeline!')

  all_test_preds = [] # Prediction from individual networks with corresponding weight

  for idx in range(NUM_NETS):
    print('Training the net ' + str(idx + 1) + ' out of ' + str(NUM_NETS))

    # Choose a subset of the data
    indices = np.random.choice(X_train.shape[0], int(X_train.shape[0] * 0.8), replace=False)
    Xt = X_train[indices,:]

    # Train the net
    nn = SimpleNet([600, 600], reg=0.00015, input_size=Xt.shape[1])
    nn.fit(Xt, y_train[indices], verbose=verbose_training, num_iters=4000, learning_rate=0.01, update_strategy='rmsprop',
      optimization_strategy='sgd', mini_batch_size=600, lr_decay=0.995)

    y_pred_val = nn.predict(Xt)
    num_correct = (y_pred_val == y_train[indices]).sum()
    print('Train results (index  ' + str(idx) + ') = ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

    # Weight to use in majority voting - correctness on the entire train dataset
    w = num_correct * 1.0 / len(y_pred_val)
    
    test_pred_idx = nn.predict(X_test)
    all_test_preds.append((test_pred_idx, w))

  # Generate final predictions from majority voting
  test_predictions = majority_combine(all_test_preds)

  # Right now predictions are [0,1], and we should convert them to [-1, 1]
  test_predictions = 2 * test_predictions
  test_predictions = test_predictions - 1

  create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')

if __name__ == '__main__':
  np.random.seed(777)
  run(verbose_training=True)
