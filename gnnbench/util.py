import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import yaml
from pymongo import MongoClient


def to_sparse_tensor(M, value=False):
    """Convert a scipy sparse matrix to a tf SparseTensor or SparseTensorValue.

    Parameters
    ----------
    M : scipy.sparse.sparse
        Matrix in Scipy sparse format.
    value : bool, default False
        Convert to tf.SparseTensorValue if True, else to tf.SparseTensor.

    Returns
    -------
    S : tf.SparseTensor or tf.SparseTensorValue
        Matrix as a sparse tensor.

    Author: Oleksandr Shchur
    """
    M = sp.coo_matrix(M)
    if value:
        return tf.SparseTensorValue(np.vstack((M.row, M.col)).T, M.data, M.shape)
    else:
        return tf.SparseTensor(np.vstack((M.row, M.col)).T, M.data, M.shape)


def dropout_supporting_sparse_tensors(X, keep_prob):
    """Add dropout layer on top of X.

    Parameters
    ----------
    X : tf.Tensor or tf.SparseTensor
        Tensor over which dropout is applied.
    keep_prob : float, tf.placeholder
        Probability of keeping a value (= 1 - probability of dropout).

    Returns
    -------
    X : tf.Tensor or tf.SparseTensor
        Tensor with elementwise dropout applied.

    Author: Oleksandr Shchur & Johannes Klicpera
    """
    if isinstance(X, tf.SparseTensor):
        # nnz = X.values.shape  # number of nonzero entries
        # random_tensor = keep_prob
        # random_tensor += tf.random_uniform(nnz)
        # dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        # pre_out = tf.sparse_retain(X, dropout_mask)
        # return pre_out * (1.0 / keep_prob)
        values_after_dropout = tf.nn.dropout(X.values, keep_prob)
        return tf.SparseTensor(X.indices, values_after_dropout, X.dense_shape)
    else:
        return tf.nn.dropout(X, keep_prob)


def scatter_add_tensor(tensor, indices, out_shape, name=None):
    """
    Code taken from https://github.com/tensorflow/tensorflow/issues/2358 and adapted.

    Adds up elements in tensor that have the same value in indices.

    Must have shape(tensor)[0] == shape(indices)[0].
    :param tensor: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
        int16, int8, complex64, complex128, qint8, quint8, qint32, half.
    :param indices: 1-D tensor of indices.
    :param out_shape: The shape of the output tensor. Must have out_shape[1] == shape(tensor)[1].
    :param name: A name for the operation (optional).
    :return: Tensor with same datatype as tensor and shape out_shape.
    """
    with tf.name_scope(name, 'scatter_add_tensor') as scope:
        indices = tf.expand_dims(indices, -1)
        # the scatter_nd function adds up values for duplicate indices what is exactly what we want
        return tf.scatter_nd(indices, tensor, out_shape, name=scope)


def uniform_float(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit)
    else:
        return random_state.uniform(lower, upper, number)


def uniform_int(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if not isinstance(lower, int):
        raise ValueError("lower must be of type 'int', got {0} instead"
                         .format(type(lower)))
    if not isinstance(upper, int):
        raise ValueError("upper must be of type 'int', got {0} instead"
                         .format(type(upper)))
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit).astype(np.int32)
    else:
        return random_state.randint(int(lower), int(upper), number)


def generate_random_parameter_settings(search_spaces_dict, num_experiments, seed):
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()

    settings = {}
    for param in search_spaces_dict:
        if search_spaces_dict[param]["format"] == "values":
            settings[param] = random_state.choice(search_spaces_dict[param]["values"], size=num_experiments)

        elif search_spaces_dict[param]["format"] == "range":
            if search_spaces_dict[param]["dtype"] == "int":
                gen_func = uniform_int
            else:
                gen_func = uniform_float
            settings[param] = gen_func(random_state,
                                       lower=search_spaces_dict[param]["min"],
                                       upper=search_spaces_dict[param]["max"],
                                       number=num_experiments,
                                       log_scale=search_spaces_dict[param]["log_scale"])

        else:
            raise ValueError(f"Unknown format {search_spaces_dict[param]['format']}.")

    settings = {key: settings[key].tolist() for key in settings}  # convert to python datatypes since MongoDB cannot
    # serialize numpy datatypes
    return settings


def get_mongo_config(config_path):
    with open(config_path, 'r') as conf:
        config = yaml.load(conf)
    return config['db_host'], config['db_port']


def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.load(conf)


def get_pending_collection(db_host, db_port):
    client = MongoClient(f"mongodb://{db_host}:{db_port}/pending")
    return client["pending"].pending


def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def get_num_trainable_weights():
    variables = tf.trainable_variables()
    return sum(np.prod(variable.get_shape()) for variable in variables)
