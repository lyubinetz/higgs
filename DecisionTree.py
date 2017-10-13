import numpy as np


##################### Data structure definition for storing a Decision Tree ##############

class Node:
    def __init__(self, attribute_index=None, split_value=None, subtree_left=None, subtree_right=None, leaf_classification=None):
        self.attribute = attribute_index
        self.split_value = split_value
        self.subtree_left = subtree_left
        self.subtree_right = subtree_right
        self.leaf_classification = leaf_classification

    def classify(self, instance_vector):
        pass
        # TODO: Add method for classifying a new instance


########################### Methods for C4.5 algorithm ############################

def all_values_same(arr):
    '''
    Function that return True iff all the values in the array given as parameter are mutually equal.
    '''
    if len(arr) == 0:
        return True
    return np.all(arr == arr[0])


def split_vector_by_missing_values(pair_vector, missing_value=-999):
    '''
    Method that splits the initial vector into two vectors: one without missing values for the feature
    and one only with missing values for the feature
    
    :param pair_vector: a N x 3 ndarray, which has on the first column the values for the feature, on the second column, the target class of the instances and on the third column the weights of each instance
    :param missing_value: how is a "missing value" marked in the array
    :return: (no_missing_value_array, missing_value_array)
    '''

    missing_values_subarray = np.array(list(filter(lambda x: x[0] == missing_value, pair_vector)))
    no_missing_values_subarray = np.array(list(filter(lambda x: x[0] != missing_value, pair_vector)))
    return no_missing_values_subarray, missing_values_subarray


def count_weighted_target_classes(pair_vec):
    '''
    Method that counts how many instances of each target class have a specific pair-vector.
    
    :param pair_vector: a N x 3 ndarray, which has on the first column the values for the feature, on the second column, the target class of the instances and on the third column the weights of each instance
    :return: [weight_0_targets, weight_1_targets]
    '''

    target_0_instances = list(filter(lambda x: x[1] == 0, pair_vec))
    target_1_instances = list(filter(lambda x: x[1] == 1, pair_vec))

    return sum(list(map(lambda x: x[2], target_0_instances))), sum(list(map(lambda x: x[2], target_1_instances)))


def compute_entropy(partition):
    '''
    Function that computes the entropy of a specific partition of a set.
    
    :param partition: the partition of the set, represented as a list of integers, representing the number of elements of the set for each subset of the partition.
    :return: the entropy of the partition
    '''

    result = 0
    total_elements = sum(partition)
    for count in partition:
        if count == 0:
            continue  # 0 * log(0) = 0 by convention

        result += ((1.0 * count / total_elements) * np.log2(1.0 * total_elements / count))

    return result


def get_best_split_per_feature(pair_vector, missing_value=-999):
    '''
    Method that computes the best split for a specified feature/attribute.
     
    :param pair_vector: a N x 3 ndarray, which has on the first column the values for the feature, on the second column, the target class of the instances and on the third column the weights of each instance
    :param missing_value: how is a missing value "marked" for this feature
    :return: (max_information_gain, split_value), where max_information gain is the maximum split
     information gain and split_value is the value where we should change the decision so that we
     can have a maximum Information Gain.
    '''

    no_missing_value_array, missing_value_array = split_vector_by_missing_values(pair_vector, missing_value)

    no_missing_weighted_count = count_weighted_target_classes(no_missing_value_array)
    missing_weighted_count = count_weighted_target_classes(missing_value_array)

    total_missing_weighted_count = sum(missing_weighted_count)
    total_no_missing_weighted_count = sum(no_missing_weighted_count)
    no_missing_entropy = compute_entropy(no_missing_weighted_count)

    min_entropy = float("inf")
    split_value = None

    previous_output_value = None
    previous_column_value = None
    left_split_count = [0, 0]  # the sum of weights for branch 0 and respectively 1 we have in the left split
    right_split_count = no_missing_weighted_count

    for column_value, output_value in pair_vector:
        if previous_output_value is None:  # first entry
            previous_output_value = output_value
            previous_column_value = column_value

        elif previous_output_value == output_value:  # didn't change output, this is not a good split
            # A split worths being checked only in a point in which the target class changes.
            previous_column_value = column_value

        else:
            ## The actual code for information gain maximizer.
            left_partition_total = sum(left_split_count)  # total weight in left split
            right_partition_total = sum(right_split_count)  # total weight in right split

            split_entropy = \
                ((1.0 * left_partition_total / total_no_missing_weighted_count) * compute_entropy(left_split_count) + \
                 (1.0 * right_partition_total / total_no_missing_weighted_count) * compute_entropy(right_split_count))

            if split_entropy < min_entropy:
                min_entropy = split_entropy
                split_value = (previous_column_value + column_value) / 2.0

            previous_output_value = output_value
            previous_column_value = column_value

        if output_value == 0:  # update
            left_split_count[0] += 1
            right_split_count[0] -= 1
        else:
            left_split_count[1] += 1
            right_split_count[1] -= 1

    max_information_gain = no_missing_entropy - min_entropy
    weighted_IG = (1.0 * total_no_missing_weighted_count) / \
                  (total_no_missing_weighted_count + total_missing_weighted_count) * \
                  max_information_gain

    return weighted_IG, split_value


def get_best_information_gain(x, y, weights):
    '''
    Function that computes the attribute for which we can find a split that maximizes the Information 
    Gain, and also computes that specific split.
    
    :param x: the array of instances and attributes for which we want to obtain the best split
    :param y: the array of target classes, coresponding to each feature in vector x
    :param weights: the array of weights for each instance, used in the C4.5 algorithm
    :return: (selected_feature, split_value), where selected_feature is the column index of the 
    desired feature and split_value is the number on which  we will base our decisions. 
    '''

    max_IG = float("-inf")
    selected_feature = None
    best_split = None

    for attribute_index in range(x.shape[1]):  # column number
        attribute_column = x[:, attribute_index]  # same size as y

        pair_vector = np.array((attribute_column, y, weights)).T
        pair_vector = sorted(pair_vector, key=lambda x: x[0])  # sort the key-value pairs.

        max_IG_feature, best_split_feature = get_best_split_per_feature(pair_vector)

        if max_IG_feature < max_IG:
            max_IG = max_IG_feature
            selected_feature = attribute_index
            best_split = best_split_feature

    return selected_feature, best_split


# x = np.array([[1,2,3], [4,5,6],[-999.0000, 10, 10], [-999.0000, 10, 10], [-999.000, 100, 100], [1,2,3]])
# y = np.array([0, 1, 1, 0,1, 0])
# get_best_information_gain(x, y)

def construct_split_datasets(X, y, weights, feature_index, split_value, missing_value=-999.0):
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
    :return (left_branch_x, left_branch_y, left_branch_weights, right_branch_x, right_branch_y, right_branch_weights):
     a tuple representing a pair of three arrays, which correspond to the instances on the 
     left branch and for those on the right branch.
    '''

    pair_vector = np.array((X, y, weights)).T
    no_missing_value_array = np.array(list(filter(lambda x: x[feature_index] != missing_value, pair_vector)))
    missing_value_array = np.array(list(filter(lambda x: x[feature_index] == missing_value, pair_vector)))

    left_branch = np.array(list(filter(lambda x: x[feature_index] < split_value, no_missing_value_array)))
    right_branch = np.array(list(filter(lambda x: x[feature_index] >= split_value, no_missing_value_array)))

    weight_left_branch = 1.0 * left_branch.shape[0] / no_missing_value_array.shape[0]
    reweighted_left_missing_values_array = missing_value_array.copy()
    reweighted_left_missing_values_array[:, 3] *= weight_left_branch
    left_branch_final = np.vstack([left_branch, reweighted_left_missing_values_array])

    weight_right_branch = 1.0 * right_branch.shape[0] / no_missing_value_array.shape[0]
    reweighted_right_missing_values_array = missing_value_array.copy()
    reweighted_right_missing_values_array[:, 3] *= weight_right_branch
    right_branch_final = np.vstack([left_branch, reweighted_left_missing_values_array])

    return left_branch_final[0], left_branch_final[1], left_branch_final[2], \
           right_branch_final[0], right_branch_final[1], right_branch_final[2]

############################ Decision Tree Pruning #################################

def prune_decision_tree(decision_tree):
    pass
    # TODO: Implement pruning function

def apply_C45(X, y):
    '''
    Function that applies the C4.5 algorithm on the training set defined by X and y. 
    :param x: a N x D array, representing the features (# features = D) for each instance 
    (#instances = N)
    :param y: an array of length N, storing the output classes for each instance. 
    :return: the decision tree resulted from applying C4.5 algorithm on the initial dataset.
    '''

    def apply_C45(X, y):
        if all_values_same(y):
            return Node(leaf_classification=y[0])
            # we have a node with only one possible value, so we consider it a leaf.

        # Else, it means we did not reach an end of the agorithm, split again and reiterate process

        selected_feature, best_split = get_best_information_gain(X, y, weights)
        left_X, left_y, left_weights, right_X, right_y, right_weights = construct_split_datasets(X, y, weights, selected_feature, best_split)

        left_branch = apply_C45_weights(left_X, left_y, left_weights)
        right_branch = apply_C45_weights(right_X, right_y, right_weights)

        return Node(attribute_index=selected_feature, split_value=best_split, subtree_left=left_branch, subtree_right=right_branch)

    weights = np.array([1.0] * len(y))  # All the weights are initialized with value 1
    root = apply_C45_weights(X, y, weights)
    return prune_decision_tree(root)




