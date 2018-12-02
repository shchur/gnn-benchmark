# -*- coding: utf-8 -*-
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from gnnbench.data.io import load_dataset
from gnnbench.data.preprocess import to_binary_bag_of_words, remove_underrepresented_classes, \
    eliminate_self_loops, binarize_labels
from gnnbench.util import is_binary_bag_of_words


def get_dataset(name, data_path, standardize, _log, train_examples_per_class=None, val_examples_per_class=None):
    if _log is not None:
        _log.info(f'Loading dataset {name}.')
    dataset_graph = load_dataset(data_path)

    # some standardization preprocessing
    if standardize:
        dataset_graph = dataset_graph.standardize()
    else:
        dataset_graph = dataset_graph.to_undirected()
        dataset_graph = eliminate_self_loops(dataset_graph)

    if train_examples_per_class is not None and val_examples_per_class is not None:
        if name == 'cora_full':
            # cora_full has some classes that have very few instances. We have to remove these in order for
            # split generation not to fail
            dataset_graph = remove_underrepresented_classes(dataset_graph,
                                                            train_examples_per_class, val_examples_per_class)
            dataset_graph = dataset_graph.standardize()
            # To avoid future bugs: the above two lines should be repeated to a fixpoint, otherwise code below might
            # fail. However, for cora_full the fixpoint is reached after one iteration, so leave it like this for now.

    graph_adj, node_features, labels = dataset_graph.unpack()
    labels = binarize_labels(labels)

    # convert to binary bag-of-words feature representation if necessary
    if not is_binary_bag_of_words(node_features):
        if _log is not None:
            _log.debug(f"Converting features of dataset {name} to binary bag-of-words representation.")
        node_features = to_binary_bag_of_words(node_features)

    # some assertions that need to hold for all datasets
    # adj matrix needs to be symmetric
    assert (graph_adj != graph_adj.T).nnz == 0
    # features need to be binary bag-of-word vectors
    assert is_binary_bag_of_words(node_features), f"Non-binary node_features entry!"

    return graph_adj, node_features, labels


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_split_feed_dicts(train_indices, val_indices, test_indices):
    dataset_indices_placeholder = tf.placeholder(tf.int32, shape=[None], name='dataset_indices_placeholder')

    train_feed = {dataset_indices_placeholder: train_indices}
    trainval_feed = {dataset_indices_placeholder: train_indices}
    val_feed = {dataset_indices_placeholder: val_indices}
    test_feed = {dataset_indices_placeholder: test_indices}

    return dataset_indices_placeholder, train_feed, trainval_feed, val_feed, test_feed


def get_dataset_and_split_planetoid(dataset, data_path, _log):
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    if _log is not None:
        _log.info('Loading dataset %s.' % dataset)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        os.path.join(data_path, "ind.{}.test.index".format(dataset))
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # cast!!!
    adj = adj.astype(np.float32)
    features = features.tocsr()
    features = features.astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return adj, features, labels, idx_train, idx_val, idx_test

