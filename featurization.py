import numpy as np
from helpers import *

'''
This file contains various utilities for creating the best artificial features.
'''

feature_names = 'DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt'.split(',')

def featurize_before_standardize(data):
  return featurize_angles(featurize_inverse(featurize_x2(data)))

def featurize_and_standardize(data, mean=None, var=None):
  '''
  Ultimate featurization to use
  '''
  rv = drop_original_angle_feat(
    featurize_rbf(
      standardize(
        featurize_angles(featurize_inverse(featurize_x2(data))),
        mean=mean,
        var=var
      )
    )
  )
  return rv

def featurize_inverse(data):
  x = np.abs(data[:,:30]) + 1
  return np.c_[data, 1.0 / x]

def featurize_rbf(data):
  '''
  Adds RBF features - see https://en.wikipedia.org/wiki/Radial_basis_function_kernel
  We only add them accross the same categories.
  '''
  categories = ['mass', 'centrality', 'eta'] # 'pt',
  for cat in categories:
    for i in range(30):
      if cat not in feature_names[i]:
        continue
      for j in range(30):
        if j <= i:
          continue
        if cat not in feature_names[j]:
          continue
        new_col = np.power(data[:, i] - data[:, j], 2)
        new_col = -1.0 * (new_col / 2.0)
        new_col = np.exp(new_col)
        data = np.c_[data, new_col]
  return data

def featurize_angles(data):
  '''
  Adds absolute values of pairwise angle differences to the data.
  '''
  rv = data

  angle_features = [i for i in range(len(feature_names)) if feature_names[i].endswith('phi')]
  for i in angle_features:
    for j in angle_features:
      if j <= i:
        continue
      rv = np.c_[data, np.abs(data[:, i] - data[:, j])]

  return rv

def drop_original_angle_feat(data):
  '''
  Drops angle and angle^2 features - they were checked to be useless
  '''
  angle_features = [i for i in range(len(feature_names)) if feature_names[i].endswith('phi')]
  # Drop squares of original angle features
  for i in reversed(angle_features):
    data = np.delete(data, i + 30, 1)

  # Drop original angle features
  for i in reversed(angle_features):
    data = np.delete(data, i, 1)

  return data

def featurize_x2(data):
  '''
  Adds x^2 features to the data.
  '''
  return np.c_[data, np.power(data, 2)]

def featurize_x2_and_minus(data):
  '''
  Adds x^2 features, and -data to the data.
  '''
  return np.c_[data, np.power(data, 2), data * -1]

def pairwise_feature_search(X_train, y_train, X_val, y_val, num_iter):
  '''
  Find a combination of 40 pairwise product features that produces the
  best result on a 1000-layer network when combined with x^2 featurization.
  '''
  best_score = -1
  best_fs = []

  for i in range(num_iter):
    print('Search iter: ' + str(i))
    pws = []
    for j in range(40):
      v1 = np.random.randint(30)
      v2 = np.random.randint(30)
      while v2 == v1:
        v2 = np.random.randint(30)
      pws.append((v1, v2))

    dat = featurize_with_pairwise(X_train, pws)
    nn = NeuralNet([1000], reg=0.001, input_dim=100)
    # Train the net
    nn.fit(dat, y_train, verbose=True, num_iters=40, learning_rate=2)

    y_pred_val = nn.predict(featurize_with_pairwise(X_val, pws))
    num_correct = (y_pred_val == y_val).sum()
    print('New pws results ' + str(num_correct) + ' out of ' +
      str(len(y_pred_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_pred_val)) + '%).')

    if num_correct > best_score:
      best_score = num_correct
      best_fs = pws

  print('Bst score ' + str(best_score))
  print('Bst pws ' + str(best_fs))

def featurize_with_pairwise(data, pairs):
  '''
  Extends data with specified pairwise product features.
  '''
  new_data = data
  for p in pairs:
    new_data = np.c_[new_data, data[:,p[0]] * data[:,p[1]]]
  return new_data

if __name__ == '__main__':
  np.random.seed(777)

  X_train, y_train = read_train_data('datasets/train.csv')
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  replace_missing_values(X_train, mean_map)
  X_train = featurize_x2(X_train)
  X_train = standardize(X_train)
  
  X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
  print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

  pairwise_feature_search(X_train, y_train, X_val, y_val, 50)
