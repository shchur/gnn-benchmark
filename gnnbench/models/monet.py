import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor, dropout_supporting_sparse_tensors, scatter_add_tensor

GAUSSIAN_WEIGHTS = "gaussian_weights"
FILTER_WEIGHTS = "filter_weights"


# dims:
#   transformed_coordinates: num_edges x r
def weighting_function(transformed_coordinates, graph_adj, adj_with_self_loops_indices, num_kernels, kernel_no):
    with tf.name_scope(f"kernel{kernel_no}"):
        r = int(transformed_coordinates.get_shape()[1])
        num_nodes = int(graph_adj.get_shape()[0])

        mean_init_values = get_mean_inits(num_kernels)

        with tf.variable_scope(f"kernel{kernel_no}", reuse=tf.AUTO_REUSE):
            # the gaussian kernels are the same for each layer, they are just different for each kernel
            # The helper is needed in order to create the variable with tf.get_variable, which is the only way that
            # works with tf.AUTO_REUSE.
            init_helper = tf.Variable([mean_init_values[kernel_no]] * r, dtype=tf.float32, trainable=False)
            mu = tf.get_variable(f"mu", dtype=tf.float32,
                                 initializer=init_helper.initialized_value()
                                 # no regularization here
                                 )
            if mu not in tf.get_collection(GAUSSIAN_WEIGHTS):
                tf.add_to_collection(GAUSSIAN_WEIGHTS, mu)

            sigma = tf.get_variable(f"sigma", [r], dtype=tf.float32,
                                    initializer=tf.ones_initializer(),
                                    # no regularization here
                                    )
            if sigma not in tf.get_collection(GAUSSIAN_WEIGHTS):
                tf.add_to_collection(GAUSSIAN_WEIGHTS, sigma)

        # dims: num_edges x r, r, r -> num_edges x r
        gaussian_weights = tf.exp(-0.5 * tf.square(transformed_coordinates - mu) / (1e-14 + tf.square(sigma)))
        # dims: num_edges x r -> num_edges
        # This is equivalent to summing over the r columns of the coordinate matrix and then applying the tf.exp
        gaussian_weights = tf.reduce_prod(gaussian_weights, axis=1)

        # Normalize weights. This needs to be done in the num_edges-vector form since SparseTensors do not support a
        # reduce sum operation. This is not given in the paper, but is done in the reference implementation.
        # dims: num_nodes x num_nodes -> num_nodes
        self_node_indices, _ = adj_with_self_loops_indices
        # First sum up the weights that belong to the neighbours of each node.
        # dims: num_edges -> num_nodes
        gaussian_weight_means = scatter_add_tensor(gaussian_weights, self_node_indices, out_shape=[num_nodes])
        # Then replicate the summed weights belonging to each node.
        gaussian_weight_means = tf.gather(gaussian_weight_means, self_node_indices)
        # Finally normalize.
        gaussian_weights = gaussian_weights / (1e-14 + gaussian_weight_means)

        # Blow vector of gaussian weights up into a sparse matrix. Use the coordinates from the original
        # adjacency matrix to specify which gaussian weight belongs to which edge.
        # dims: num_nodes x num_nodes, num_edges -> num_nodes x num_nodes
        gaussian_weight_matrix = tf.SparseTensor(
            indices=graph_adj.indices,
            values=gaussian_weights,
            dense_shape=graph_adj.dense_shape
        )

        return gaussian_weight_matrix


# dims:
#    inputs: num_nodes x num_features
#    transformed_coordinates: num_edges x r
def gaussian_kernel(inputs, output_dim, transformed_coordinates, graph_adj, adj_with_self_loops_indices,
                    num_kernels, kernel_no, weight_decay, name):
    with tf.name_scope(name):
        input_dim = int(inputs.get_shape()[1])

        # These are the g_j from the paper.
        linear_transform_weights = tf.get_variable(
            f"{name}-linear_transform_weights",
            [input_dim, output_dim], dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer(),
            regularizer=slim.l2_regularizer(weight_decay)
        )
        tf.add_to_collection(FILTER_WEIGHTS, linear_transform_weights)

        # dims: num_nodes x num_features, num_features x output_dim -> num_nodes x output_dim
        if isinstance(inputs, tf.SparseTensor):
            transformed_features = tf.sparse_tensor_dense_matmul(inputs, linear_transform_weights)
        else:
            transformed_features = tf.matmul(inputs, linear_transform_weights)

        # dims: num_edges x r -> num_nodes x num_nodes
        gaussian_weights = weighting_function(transformed_coordinates, graph_adj, adj_with_self_loops_indices,
                                              num_kernels, kernel_no)

        # dims: num_nodes x num_nodes, num_nodes x output_dim -> num_nodes x output_dim
        output = tf.sparse_tensor_dense_matmul(gaussian_weights, transformed_features)
        return output


def gmm_layer(inputs, output_dim, transformed_coordinates, num_kernels, graph_adj, adj_with_self_loops_indices,
              activation_fn, dropout_prob, weight_decay, name):
    with tf.name_scope(name):
        inputs = tf.cond(
            tf.cast(dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(inputs, 1.0 - dropout_prob)),
            false_fn=(lambda: inputs)
        )

        kernel_results = []
        for kernel_no in range(num_kernels):
            # dims: num_nodes x num_features, output_dim, num_nodes x num_nodes -> num_nodes x output_dim
            kernel_results.append(gaussian_kernel(inputs=inputs, output_dim=output_dim,
                                                  transformed_coordinates=transformed_coordinates,
                                                  graph_adj=graph_adj,
                                                  adj_with_self_loops_indices=adj_with_self_loops_indices,
                                                  num_kernels=num_kernels,
                                                  kernel_no=kernel_no,
                                                  weight_decay=weight_decay,
                                                  name=f"{name}-kernel{kernel_no}"))

        output = tf.add_n(kernel_results)
        return activation_fn(output) if activation_fn else output


class MoNet(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, num_kernels, r, dropout_prob, weight_decay,
                 normalize_features, alt_opt):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_kernels = num_kernels
        self.r = r
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.alt_opt = alt_opt

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            # the coordinate transform weights and bias are the same for every layer
            coordinate_transform_weights = tf.get_variable(f"coordinate_transform_weights",
                                                           [2, self.r], dtype=tf.float32,
                                                           initializer=tf.glorot_uniform_initializer(),
                                                           # no regularization here
                                                           )
            tf.add_to_collection(GAUSSIAN_WEIGHTS, coordinate_transform_weights)
            # dims: num_edges x 2, 2 x r -> num_edges x r
            transformed_coordinates = tf.matmul(self.coordinates, coordinate_transform_weights)
            transformed_coordinates = tf.contrib.layers.bias_add(transformed_coordinates,
                                                                 variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                        tf.GraphKeys.VARIABLES,
                                                                                        GAUSSIAN_WEIGHTS])
            transformed_coordinates = tf.nn.tanh(transformed_coordinates)

            x = self.features
            for i in range(0, self.num_layers - 1):
                x = gmm_layer(inputs=x,
                              output_dim=self.hidden_size,
                              transformed_coordinates=transformed_coordinates,
                              num_kernels=self.num_kernels,
                              graph_adj=self.graph_adj,
                              adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                              activation_fn=tf.nn.relu,
                              dropout_prob=self.dropout_prob,
                              # original implementation uses L2 regularization only on first layer
                              weight_decay=self.weight_decay if i == 0 else 0.0,
                              name=f"gmm_layer{i}")
            output = gmm_layer(inputs=x,
                               output_dim=self.targets.shape[1],
                               transformed_coordinates=transformed_coordinates,
                               num_kernels=self.num_kernels,
                               graph_adj=self.graph_adj,
                               adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                               activation_fn=None,
                               dropout_prob=self.dropout_prob,
                               weight_decay=0.0,
                               name=f"gmm_layer{self.num_layers - 1}")

        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return tf.constant(features.todense())

    def _preprocess_adj(self, graph_adj):
        adj_with_self_loops = add_self_loops(graph_adj)
        self.adj_dense_shape = adj_with_self_loops.shape
        adj_with_self_loops_tensor = to_sparse_tensor(adj_with_self_loops)

        adj_with_self_loops_coo = adj_with_self_loops.tocoo()
        # extract the coordinates of all the edges
        # since both row and column coordinates are ordered, row[0] corresponds to col[0] etc.
        self.adj_with_self_loops_indices = (adj_with_self_loops_coo.row, adj_with_self_loops_coo.col)
        self.coordinates = self._generate_coordinates(adj_with_self_loops, self.adj_with_self_loops_indices)

        return adj_with_self_loops_tensor

    @staticmethod
    def _generate_coordinates(adj_with_self_loops, adj_with_self_loops_indices):
        degrees = adj_with_self_loops.sum(axis=1).astype(np.float32)
        # here, a small additive constant in the denominator would be necessary to avoid division by 0. However,
        # adding the self loops ensures no degree is 0.
        inv_degrees = 1.0 / np.sqrt(degrees)

        start_nodes, end_nodes = adj_with_self_loops_indices
        start_node_degrees = inv_degrees[list(start_nodes)]
        end_node_degrees = inv_degrees[list(end_nodes)]
        # dims: num_edges x 2
        return np.hstack([start_node_degrees, end_node_degrees])

    # override optimize method to employ alternating optimization
    def optimize(self, learning_rate, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if self.alt_opt:
            return [
                optimizer.minimize(self.loss, global_step=global_step, var_list=tf.get_collection(GAUSSIAN_WEIGHTS)),
                optimizer.minimize(self.loss, global_step=global_step, var_list=tf.get_collection(FILTER_WEIGHTS)),
                ]
        return optimizer.minimize(self.loss, global_step=global_step)


def get_mean_inits(num_kernels):
    return np.arange(-1.0 + (1.0 / (2 * num_kernels)), 1.0, 2.0 / num_kernels)


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, num_kernels, r, dropout_prob, alt_opt):
    dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[dropout] = dropout_prob
    trainval_feed[dropout] = False
    val_feed[dropout] = False
    test_feed[dropout] = False

    return MoNet(node_features, graph_adj, labels, dataset_indices_placeholder,
                 num_layers=num_layers, hidden_size=hidden_size,
                 num_kernels=num_kernels,
                 r=r,
                 dropout_prob=dropout,
                 weight_decay=weight_decay,
                 normalize_features=normalize_features,
                 alt_opt=alt_opt)
