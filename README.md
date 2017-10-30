# higgs
EPFL ML Project 1 - detecting the Higgs Boson.

### How to install and run (*nix friendly guide)
If you're using Windows, some steps might differ.

#### Prerequisites

1. Python 3

2. PIP

3. Numpy

#### Usage instructions
1. Install numpy

  ```
  pip3 install numpy
  ```
2. Change current directory to the project's folder

  ```
  cd higgs
  ```
3. Make sure **train.csv** and **test.csv** files are located in the **datasets/** folder. If such folder does not exist, create it at the root of project's folder then copy the **train.csv** and **test.csv** files there.
4. Run 

  ```
  python3 run.py
  ```
  
### Code structure

* run.py - this file contains the final pipeline, which is a bag of neural networks. It is a simplified version of neural_bagging.py.
* run_nn.py - a pipeline that trains a single neural network. Depending on the options specified inside the file, it either runs it on a validation set and prints the result, or on test set and produces predition.csv
* neural_bagging.py - a pipeline that trains several neural networks and combines them with majority voting. Depending on the options specified inside the file, it either runs it on a validation set and prints the result, or on test set and produces predition.csv
* simple_net.py - A neural network implementation. See the file for detailed parameters behind it. A typical usage involes 2 calls - fit() to train a network and predict() to get predictions on some set.
* run_linear.py - Used to run pipelines on regressions.
* majority_combinator.py - function that combines several predictions into one using majority voting.
* utils.py - Utility for checking intesresction of two predictions (what they both get right, wrong, etc)
* helpers.py - Various helper functions.
* implementations.py - Implementaions of basic methods.
* featurization.py - Featurization pipeline.
* hyperparapemeter_nn_grid_search.py - Grid search utility to pick the best hyperparameters.
* DecisionTree.py - decision tree.

