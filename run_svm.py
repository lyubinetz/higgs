import numpy as numpy
import pandas as pd
from helpers import *
from svm import *

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''
def run():
  X_train, y_train = read_train_data('datasets/train.csv')

  mean_map = compute_means_for_columns(X_train)
  replace_missing_values_with_means(X_train, mean_map)
  X_train = standardize(X_train)
  
  svm = SVM()
  # Train the SVM
  svm.fit(X_train, y_train, verbose=True)

  X_test, X_test_ids = read_test_data('datasets/test.csv')
  replace_missing_values_with_means(X_test, mean_map)
  X_test = standardize(X_test)
  test_predictions = svm.predict(X_test)
  # HACK: Right now predictions are 0,1 , and we need -1,1
  test_predictions = 2 * test_predictions
  test_predictions = test_predictions - 1

  test_predictions = pd.DataFrame(np.array([X_test_ids, test_predictions]).T, columns=['Id', 'Prediction'])
  test_predictions.to_csv('prediction.csv', sep=',', columns=['Id', 'Prediction'], index=False)

if __name__ == '__main__':
  run()
