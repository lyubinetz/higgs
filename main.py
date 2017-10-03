import numpy as numpy
from helpers import *

'''
Runs the clasification pipeline. In the end this should produce a file
called prediction.csv with test set classification.
'''
def run():
  X_train, y_train = read_experiment_data('datasets/train.csv')
  # TODO: do something smart :)

if __name__ == '__main__':
  run()
