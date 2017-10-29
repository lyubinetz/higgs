import numpy as np
from helpers import *
from simple_net import *
from featurization import *
from majority_combinator import *
from utils import compare_results

NUM_NETS = 10

def run(validation, classify_test):
  '''
  Runs the clasification pipeline. In the end this should produce a file
  called prediction.csv with test set classification.
  '''
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
    all_val_preds = []

  all_indices = [] # Indices of columns used for each partial NN
  weights = []

  all_nets = []

  for idx in range(NUM_NETS):
    print('Idx is ' + str(idx))
    indices = np.random.choice(X_train.shape[0], 200000)
    Xt = X_train[indices,:]
    nn1 = SimpleNet([600, 600], reg=0.0001, input_size=Xt.shape[1])
    # Train the net
    nn1.fit(Xt, y_train[indices], verbose=True, num_iters=3000, learning_rate=0.01, update_strategy='rmsprop',
      optimization_strategy='sgd', mini_batch_size=600, lr_decay=0.995)

    all_nets.append(nn1)

    y_pred_val = nn1.predict(Xt)
    num_correct = (y_pred_val == y_train[indices]).sum()
    print('Train results (index  ' + str(idx) + ') = ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

    # Weight to use in majority voring
    w = num_correct * 1.0 / len(y_pred_val)
    weights.append(w)

    if validation:
      y_pred_val = nn1.predict(X_val)
      all_val_preds.append((y_pred_val, w))
      num_correct = (y_pred_val == y_val).sum()
      print('Validation results (index  ' + str(idx) + ') = ' + str(num_correct) + ' out of ' +
        str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  # Compute validation score
  if validation:
    final_pred = majority_combine(all_val_preds)
    num_correct = (final_pred == y_val).sum()
    print('Final validation results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

  if classify_test:
    # Compute result for submission
    replace_missing_values(X_test, mean_map)
    X_test = featurize_and_standardize(X_test, mean=good_featurized_means, var=good_featurized_vars)

    all_test_preds = []
    for idx in range(NUM_NETS):
      nn = all_nets[idx]
      test_pred_idx = nn.predict(X_test)
      all_test_preds.append((test_pred_idx, weights[idx]))

    test_predictions = majority_combine(all_test_preds)

    # HACK: Right now predictions are 0,1 , and we need -1,1
    test_predictions = 2 * test_predictions
    test_predictions = test_predictions - 1

    create_csv_submission(X_test_ids, test_predictions, 'prediction.csv')

if __name__ == '__main__':
  np.random.seed(777)
  run(False, True)
