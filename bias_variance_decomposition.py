from base_methods_wrappers import *
from implementations import *
from featurization import *


def bias_variance_decomposition_visualization(complexity_regulator, rmse_tr, rmse_te, x_label, y_label):
  """visualize the bias variance decomposition."""
  rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
  rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(
    complexity_regulator,
    rmse_tr.T,
    'b',
    linestyle="-",
    color=([0.7, 0.7, 1]),
    label='train',
    linewidth=0.3)
  ax.plot(
    complexity_regulator,
    rmse_te.T,
    'r',
    linestyle="-",
    color=[1, 0.7, 0.7],
    label='test',
    linewidth=0.3)
  ax.plot(
    complexity_regulator,
    rmse_tr_mean.T,
    'b',
    linestyle="-",
    label='train',
    linewidth=3)
  ax.plot(
    complexity_regulator,
    rmse_te_mean.T,
    'r',
    linestyle="-",
    label='test',
    linewidth=3)
  #fig.ylim(np.min(complexity_regulator), np.max(complexity_regulator))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title("Bias-Variance Decomposition")
  fig.savefig("bias_variance_{}".format(x_label))


def bias_var_decomp_logistic_regression_regularizer(X_train, y_train):
  lambdas = [ 0.001, 4][::-1]
  seeds = range(5)

  # define list to store the variable
  rmse_tr = np.empty((len(seeds), len(lambdas)))
  rmse_te = np.empty((len(seeds), len(lambdas)))

  for index_seed, seed in enumerate(seeds):
    fold_indices = build_k_indices(y_train, 200, seed)[0]
    for index_lambda, lambda_ in enumerate(lambdas):
      learn_method_params = {
        'lambda_':lambda_,
        'initial_w':np.zeros((X_train.shape[1]+1,1)),
        'max_iters':200,
        'gamma':0.0000001,
        'sgd':False
      }

      wrapper_constructor_params = {
        'method':reg_logistic_regression,
        'method_params':learn_method_params
      }

      learner = BaseMethodWrapper(**wrapper_constructor_params)

      loss_tr, loss_te = train_and_evaluate(learner,
                         y_train,
                         stack_ones(X_train),
                         fold_indices,
                         lambda y_test, y_pred : (y_pred == y_test).sum()*100/len(y_pred),
                         {}
                         )
      rmse_tr[index_seed, index_lambda] = 1-loss_tr/100
      rmse_te[index_seed, index_lambda] = 1-loss_te/100

  bias_variance_decomposition_visualization(lambdas, rmse_tr, rmse_te, 'lambda', 'accuracy')


def bias_var_logistic_reg_pbf_degree(X, y_train, X_c):
  print("pbf degree")
  degrees = [2,3,4,5,7]
  seeds = range(5)

  # define list to store the variable
  rmse_tr = np.empty((len(seeds), len(degrees)))
  rmse_te = np.empty((len(seeds), len(degrees)))

  for index_seed, seed in enumerate(seeds):
    fold_indices = build_k_indices(y_train, 200, seed)[0]
    for index_degree, degree in enumerate(degrees):
      X_comb_fit = build_poly(X_c, degree)
      means, vars = compute_means_and_vars_for_columns(X_comb_fit)
      X_train = standardize(build_poly(X, degree), means, vars)
      learn_method_params = {
        'lambda_':0.01,
        'initial_w':np.zeros((X_train.shape[1]+1,1)),
        'max_iters':200,
        'gamma':0.0000001,
        'sgd':False
      }

      wrapper_constructor_params = {
        'method':reg_logistic_regression,
        'method_params':learn_method_params
      }

      learner = BaseMethodWrapper(**wrapper_constructor_params)

      loss_tr, loss_te = train_and_evaluate(learner,
                                            y_train,
                                            stack_ones(X_train),
                                            fold_indices,
                                            lambda y_test, y_pred : (y_pred == y_test).sum()*100/len(y_pred),
                                            {}
                                            )
      rmse_tr[index_seed, index_degree] = 1-loss_tr/100
      rmse_te[index_seed, index_degree] = 1-loss_te/100

  bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te, 'degree', 'accuracy')


if __name__ == "__main__":
  print('Started the run!')
  X_train, y_train = read_train_data('datasets/train.csv')
  pbf = False 
  X_test, X_test_ids = read_test_data('datasets/test.csv')

  print('Finished loading data!')

  X_combined = np.vstack((X_train, X_test))
  mean_map, var_map = compute_means_and_vars_for_columns(X_combined)

  # Compute featurzied means
  replace_missing_values(X_combined, mean_map)
  replace_missing_values(X_train, mean_map)

  if not pbf:
    good_featurized_means, good_featurized_vars = compute_means_and_vars_for_columns(featurize_before_standardize(X_combined))

    X_train = featurize_and_standardize(X_train, mean=good_featurized_means, var=good_featurized_vars)

    print('New number of features is ' + str(X_train.shape[1]))
    print('Finished data ops!')
    bias_var_decomp_logistic_regression_regularizer(X_train, y_train)
  else:
    bias_var_logistic_reg_pbf_degree(X_train, y_train, X_combined)
