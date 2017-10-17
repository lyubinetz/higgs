import pickle
import os

DECISION_TREE_FILE = os.path.join('.', 'tree_serialized.bin')
SERIALIZED_TRAINING_FILE = os.path.join('.', 'datasets', 'serialized_training')
SERIALIZED_TEST_FILE = os.path.join('.', 'datasets', 'serialized_test')

#################### Data structure definition for storing a Decision Tree ##############

class Node:
    def __init__(self, attribute_index=None, split_value=None, subtree_missing=None, subtree_left=None,
                 subtree_right=None,
                 leaf_classification=None):
        self.attribute = attribute_index
        self.split_value = split_value
        self.subtree_missing = subtree_missing
        self.subtree_left = subtree_left
        self.subtree_right = subtree_right
        self.leaf_classification = leaf_classification

    def classify(self, instance_vector, missing_value=-999):
        if self.leaf_classification is not None:  # If it's a leaf, then output the classification
            return self.leaf_classification

        if instance_vector[self.attribute] == missing_value:
            # If current attribute is missing, then follow the missing branch
            return self.subtree_missing.classify(instance_vector, missing_value)

        if instance_vector[self.attribute] < self.split_value:
            # follow the left branch
            return self.subtree_left.classify(instance_vector, missing_value)

        # follow the right branch
        return self.subtree_right.classify(instance_vector, missing_value)


########################### Methods for C4.5 algorithm ############################

def all_values_same(arr):
    '''
    Function that return True iff all the values in the array given as parameter are mutually equal.
    '''
    if len(arr) == 0:
        return True
    return np.all(arr == arr[0])

def compute_target_majority(y):
    '''
    Function that returns the most common target from a target vector.
    :param y: the target vector
    :return: the most common element in y
    '''

    output_counts = np.unique(y, return_counts=True)[1]
    if output_counts[0] > output_counts[1]:
        return 0

    return 1

def split_vector_by_missing_values(pair_vector, missing_value=-999):
    '''
    Method that splits the initial vector into two vectors: one without missing values for the feature
    and one only with missing values for the feature
    
    :param pair_vector: a N x 2 ndarray, which has on the first column the values for the feature and on the second column, the target class of the instances.
    :param missing_value: how is a "missing value" marked in the array
    :return: (no_missing_value_array, missing_value_array)
    '''
    try:
        missing_values_subarray = pair_vector[pair_vector[:,0] == missing_value]
    except IndexError as e:
        print(pair_vector)
    no_missing_values_subarray = pair_vector[pair_vector[:,0] != missing_value]
    return no_missing_values_subarray, missing_values_subarray


def count_target_classes(pair_vec):
    '''
    Method that counts how many instances of each target class have a specific pair-vector.
    
    :param pair_vector: a N x 2 ndarray, which has on the first column the values for the feature and on the second column, the target class of the instances.
    :return: [count_0_targets, count_1_targets]
    '''

    target_0_instances = pair_vec[pair_vec[:,1] == 0]

    return target_0_instances.shape[0], pair_vec.shape[0] - target_0_instances.shape[0]


def compute_entropy(partition):
    '''
    Function that computes the entropy of a specific partition of a set.
    
    :param partition: the partition of the set, represented as a list of integers, representing the number of elements of the set for each subset of the partition.
    :return: the entropy of the partition
    '''

    result = 0
    total_elements = sum(partition)

    if total_elements == 0:
        return 0  # This is for the missing branch, when we don't have any instance in the branch

    for count in partition:
        if count == 0:
            continue  # 0 * log(0) = 0 by convention

        result += ((1.0 * count / total_elements) * np.log2(1.0 * total_elements / count))

    return result


def get_best_split_per_feature(pair_vector, missing_value=-999):
    '''
    Method that computes the best split for a specified feature/attribute.
     
    :param pair_vector: a N x 2 ndarray, which has on the first column the values for the feature and on the second column, the target class of the instances.
    :param missing_value: how is a missing value "marked" for this feature
    :return: (min_entropy, split_value), where min_entropy is the minimum entropy that could be 
    achieved by the best split and split_value is the value where we should change the decision
     so that we can have a minimum entropy.
    '''

    no_missing_value_array, missing_value_array = split_vector_by_missing_values(pair_vector, missing_value)

    no_missing_count_vector = count_target_classes(no_missing_value_array)
    missing_count_vector = count_target_classes(missing_value_array)

    total_missing_count = sum(missing_count_vector)
    total_count = len(pair_vector)

    missing_count_vector = count_target_classes(missing_value_array)
    missing_entropy = compute_entropy(missing_count_vector)

    min_entropy = float("inf")
    split_value = None

    previous_output_value = None
    previous_column_value = None
    left_split_count = [0, 0]  # the counts for targets of 0 and respectively 1 we have in the left split
    right_split_count = list(no_missing_count_vector)

    for column_value, output_value in pair_vector:
        if previous_output_value is None:  # first entry
            previous_output_value = output_value
            previous_column_value = column_value

        elif previous_output_value == output_value:  # didn't change output, this is not a good split
            # A split worths being checked only in a point in which the target class changes.
            previous_column_value = column_value

        else:
            ## The actual code for information gain maximizer.
            left_partition_total = sum(left_split_count)  # total count in left split
            right_partition_total = sum(right_split_count)  # total count in right split

            split_entropy = \
                ((1.0 * left_partition_total / total_count) * compute_entropy(left_split_count) + \
                 (1.0 * right_partition_total / total_count) * compute_entropy(right_split_count))

            if split_entropy < min_entropy:
                min_entropy = split_entropy
                split_value = (previous_column_value + column_value) / 2.0

            previous_output_value = output_value
            previous_column_value = column_value

        if output_value == 0:  # update the counts of left and right branch on the fly
            left_split_count[0] += 1
            right_split_count[0] -= 1
        else:
            left_split_count[1] += 1
            right_split_count[1] -= 1

    min_entropy = (1.0 * total_missing_count / total_count) * missing_entropy + min_entropy
    # The value above is in fact the actual entropy induced by the split

    return min_entropy, split_value


def get_best_information_gain(x, y, missing_value=-999.0):
    '''
    Function that computes the attribute for which we can find a split that maximizes the Information 
    Gain, and also computes that specific split.
    
    :param x: the array of instances and attributes for which we want to obtain the best split
    :param y: the array of target classes, coresponding to each feature in vector x
    :return: (selected_feature, split_value), where selected_feature is the column index of the 
    desired feature and split_value is the number on which  we will base our decisions. 
    '''

    min_entropy = float("inf")
    selected_feature = None
    best_split = None

    for attribute_index in range(x.shape[1]):  # column number
        attribute_column = x[:, attribute_index]  # same size as y

        if all_values_same(attribute_column) and attribute_column[0] == missing_value:
            # we are on a missing column branch, shouldn't use that attribute again.
            continue
        pair_vector = np.array((attribute_column, y)).T
        pair_vector = np.array(sorted(pair_vector, key=lambda x: x[0]))  # sort the key-value pairs.

        min_entropy_feature, best_split_feature = get_best_split_per_feature(pair_vector)

        if min_entropy_feature < min_entropy:
            min_entropy = min_entropy_feature
            selected_feature = attribute_index
            best_split = best_split_feature

    return selected_feature, best_split


# x = np.array([[1,2,3], [4,5,6],[-999.0000, 10, 10], [-999.0000, 10, 10], [-999.000, 100, 100], [1,2,3]])
# y = np.array([0, 1, 1, 0,1, 0])
# get_best_information_gain(x, y)

def construct_split_datasets(X, y, feature_index, split_value, missing_value=-999.0):
    '''
    Method that splits our current dataset in two: one of them with all the values less than 
    the value of split_value, and the other with all the values greater or equal to that value.
    Notice that the weights will be kept the same for those instances.
    
    Furthermore, the instances with missing values will be included in both datasets, but with 
    different weights, based on the relative size of the branch compared to the entire dataset.
    :param x: a N x D array, representing the features (# features = D) for each instance 
    (#instances = N)
    :param feature_index: the index for the feature for which we want to perform the split
    :param split_value: the decision value for deciding if an instance id on the left or right branch
    :param missing_value: how is a "missing value" marked in the array
    :return (missing_branch_x, missing_branch_y, left_branch_x, left_branch_y, right_branch_x, right_branch_y):
     a tuple representing three pairs of arrays, which correspond to the instances on the 
     missing_branch, left branch and for those on the right branch.
    '''

    pair_vector = np.hstack((X, np.array([y]).T))

    no_missing_value_array = pair_vector[pair_vector[:, feature_index] != missing_value]
    left_branch = no_missing_value_array[no_missing_value_array[:,feature_index] < split_value]
    right_branch = no_missing_value_array[no_missing_value_array[:,feature_index] >= split_value]
    missing_value_array = pair_vector[pair_vector[:,feature_index] == missing_value]

    return missing_value_array[:,:-1], missing_value_array[:,-1], left_branch[:,:-1], left_branch[:,-1], \
           right_branch[:,:-1], right_branch[:,-1]


############################ Decision Tree Pruning #################################

def prune_decision_tree(decision_tree):
    return decision_tree
    # TODO: Implement pruning function


def apply_C45(X, y):
    '''
    Function that applies the C4.5 algorithm on the training set defined by X and y. 
    :param x: a N x D array, representing the features (# features = D) for each instance 
    (#instances = N)
    :param y: an array of length N, storing the output classes for each instance. 
    :return: the decision tree resulted from applying C4.5 algorithm on the initial dataset.
    '''

    def apply_C45_recursive(X, y):
        if len(X) == 0:
            return None  # Might enter here only for missing branch

        if all_values_same(y):
            return Node(leaf_classification=y[0])
            # we have a node with only one possible value, so we consider it a leaf.4

        if all_values_same(X):
            # all values the same for X, but not all have same target => leaf node with majority
            return Node(leaf_classification=compute_target_majority(y))

        # Else, it means we did not reach an end of the algorithm, split again and reiterate process
        selected_feature, best_split = get_best_information_gain(X, y)
        missing_X, missing_y, left_X, left_y, right_X, right_y = construct_split_datasets(X, y, selected_feature,
                                                                                          best_split)

        # If one of the branches contains all instances, then return a leaf node, with the most popular
        # target value in the branch
        if len(missing_y) == len(X) or len(left_y) == len(X) or len(right_y) == len(X) : # all values are in only one branch
            return Node(leaf_classification=compute_target_majority(y))

        missing_branch = apply_C45_recursive(missing_X, missing_y)
        left_branch = apply_C45_recursive(left_X, left_y)
        right_branch = apply_C45_recursive(right_X, right_y)

        return Node(attribute_index=selected_feature, split_value=best_split, subtree_missing=missing_branch,
                    subtree_left=left_branch, subtree_right=right_branch)

    root = apply_C45_recursive(X, y)
    return prune_decision_tree(root)


#############################33 run #####################
from helpers import *


def run(validation, classify_test):
    X_train, y_train = pickle.load(open(SERIALIZED_TRAINING_FILE, 'rb'))
    X_test, X_test_ids = pickle.load(open(SERIALIZED_TEST_FILE, 'rb'))

    # X_train, y_train = read_train_data('datasets/train.csv')
    # X_test, X_test_ids = read_test_data('datasets/test.csv')
    #
    # pickle.dump((X_train, y_train), open(SERIALIZED_TRAINING_FILE, 'wb'))
    # pickle.dump((X_test, X_test_ids), open(SERIALIZED_TEST_FILE, 'wb'))

    if validation:
        X_train, y_train, X_val, y_val = split_data(0.8, X_train, y_train)
        print('Train/Val sizes ' + str(len(y_train)) + '/' + str(len(y_val)))

    # decision_tree = apply_C45(X_train, y_train)
    # pickle.dump(decision_tree, open(DECISION_TREE_FILE, 'wb'))
    decision_tree = pickle.load(open(DECISION_TREE_FILE, 'rb'))


    print("Finished constructing decision tree.")

    # Compute validation score
    if validation:
        num_correct = 0
        for X, y in zip(X_val, y_val):
            y_pred_val = decision_tree.classify(X)
            if y_pred_val == y:
                num_correct += 1
        print('Validation results ' + str(num_correct) + ' out of ' +
              str(len(X_val)) + ' are correct (' + str(num_correct * 100.0 / len(y_val)) + '%).')

    if classify_test:
        # Compute result for submission
        test_predictions = []

        for test_instance in X_test:
            test_predictions.append(decision_tree.classify(test_instance))

        test_predictions = np.array(test_predictions)

        # # HACK: Right now predictions are 0,1 , and we need -1,1
        test_predictions = 2 * test_predictions
        test_predictions = test_predictions - 1
        #
        create_csv_submission(X_test_ids, test_predictions, 'decision_tree_prediction.csv')


if __name__ == '__main__':
    np.random.seed(777)
    run(False, True)
