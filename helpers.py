import numpy as np
import pandas as pd

'''
This function reads a CSV file from a given location into a pandas datafrmame.
This can be converted to numpy
'''
def read_experiment_data(fname):
  data = pd.read_csv(fname, sep=',')
  X = data.drop('Prediction', axis=1)
  y = data['Prediction']
  return X, y
