"""Code in this file is inspired by Velickovic et al. - Graph Attention Networks
and Master's Thesis of Johannes Klicpera (TUM, KDD)."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor

ATTENTION_WEIGHTS = "attention_weights"
FILTER_WEIGHTS = "filter_weights"


def attention_mechanism(features, graph_adj, adj_with_self_loops_indices, coefficient_dropout_prob, weight_decay, name):
    # apply a feedforward network parametrized with a weight vector to the transformed features.
    input_dim = int(features.get_shape()[1])
    a_i = tf.get_variable(f"{name}-att_i", [input_dim, 1], dtype=tf.float32,
                          initializer=tf.glorot_uniform_initializer(),
                          regularizer=slim.l2_regularizer(weight_decay))
    a_j = tf.get_variable(f"{name}-att_j", [input_dim, 1], dtype=tf.float32,
                          initializer=tf.glorot_uniform_initializer(),
                          regularizer=slim.l2_regularizer(weight_decay))
    tf.add_to_collection(ATTENTION_WEIGHTS, a_i)
    tf.add_to_collection(ATTENTION_WEIGHTS, a_j)

    # dims: num_nodes x input_dim, input_dim, 1 -> num_nodes x 1
    att_i = tf.matmul(features, a_i)
    att_i = tf.contrib.layers.bias_add(att_i)
    # dims: num_nodes x input_dim, input_dim, 1 -> num_nodes x 1
    att_j = tf.matmul(features, a_j)
    att_j = tf.contrib.layers.bias_add(att_j)

    # Extracts the relevant attention coefficients with respect to the 1-hop neighbours of each node
    # Method: first extract all the attention coefficients of the left nodes of each edge, then those
    # of the right nodes and add them up.
    # The result is a list of relevant attention weights ordered in the same way as the edges in the
    # sparse adjacency matrix.
    # dims: num_nodes x 1, num_edges, num_nodes x 1, num_edges -> 1 x num_edges x 1
    attention_weights_of_edges = tf.gather(att_i, adj_with_self_loops_indices[0], axis=0) + \
                                 tf.gather(att_j, adj_with_self_loops_indices[1], axis=0)
    # dims: 1 x num_edges x 1 -> num_edges
    attention_weights_of_edges = tf.squeeze(attention_weights_of_edges)

    # blow list of attention weights up into a sparse matrix. Use the coordinates from the original
    # adjacency matrix to specify which attention weight belongs to which edge.
    # Simultaneously applies the LeakyReLU as given in the paper.
    # dims: num_nodes x num_nodes, num_edges -> num_nodes x num_nodes
    attention_weight_matrix = tf.SparseTensor(
        indices=graph_adj.indices,
        values=tf.nn.leaky_relu(attention_weights_of_edges, alpha=0.2),
        dense_shape=graph_adj.dense_shape
    )

    # finish the attention by normalizing coefficients using softmax
    attention_coefficients = tf.sparse_softmax(attention_weight_matrix)

    # apply dropout to attention coefficients, meaning that in every epoch a single node is only exposed to a
    # sampled subset of its neighbour
    attention_coefficients = tf.cond(
        tf.cast(coefficient_dropout_prob, tf.bool),
        true_fn=(lambda: dropout_supporting_sparse_tensors(attention_coefficients, 1.0 - coefficient_dropout_prob)),
        false_fn=(lambda: attention_coefficients)
    )

    return attention_coefficients


def attention_head(inputs, output_dim, graph_adj, adj_with_self_loops_indices, activation_fn,
                   input_dropout_prob, coefficient_dropout_prob,
                   weight_decay,
                   name):
    with tf.name_scope(name):
        input_dim = int(inputs.get_shape()[1])
        # apply dropout to the inputs
        inputs = tf.cond(
            tf.cast(input_dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(inputs, 1.0 - input_dropout_prob)),
            false_fn=(lambda: inputs)
        )

        linear_transform_weights = tf.get_variable(
            f"{name}-linear_transform_weights",
            [input_dim, output_dim], dtype=tf.float32,
            initializer=tf.glorot_uniform_initializer(),
            regularizer=slim.l2_regularizer(weight_decay)
        )
        tf.add_to_collection(FILTER_WEIGHTS, linear_transform_weights)

        # dims: num_nodes x input_dim, input_dim x output_dim -> num_nodes x output_dim
        if isinstance(inputs, tf.SparseTensor):
            transformed_features = tf.sparse_tensor_dense_matmul(inputs, linear_transform_weights)
        else:
            transformed_features = tf.matmul(inputs, linear_transform_weights)

        # dims: num_nodes x num_output -> num_nodes x num_nodes
        attention_coefficients = attention_mechanism(transformed_features, graph_adj, adj_with_self_loops_indices,
                                                     coefficient_dropout_prob, weight_decay, name)

        # this additional dropout is used in the official implementation, but is not described in the paper.
        transformed_features = tf.cond(
            tf.cast(input_dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(transformed_features, 1.0 - input_dropout_prob)),
            false_fn=(lambda: transformed_features),
        )

        # normal feedforward layer to finish up
        # dims: num_nodes x num_nodes, num_nodes x output_dim -> num_nodes x output_dim
        output = tf.sparse_tensor_dense_matmul(attention_coefficients, transformed_features)
        output = tf.contrib.layers.bias_add(output)

        if activation_fn is not None:
            output = activation_fn(output)
        return output


def attention_layer(inputs, output_dim, num_heads, graph_adj, adj_with_self_loops_indices, activation_fn,
                    use_averaging,
                    input_dropout_prob, coefficient_dropout_prob,
                    weight_decay,
                    name):
    with tf.name_scope(name):
        head_results = []
        for i in range(num_heads):
            # dims: num_nodes x num_features, output_dim, num_nodes x num_nodes -> num_nodes x output_dim
            head_results.append(attention_head(inputs=inputs, output_dim=output_dim,
                                               graph_adj=graph_adj,
                                               adj_with_self_loops_indices=adj_with_self_loops_indices,
                                               activation_fn=activation_fn,
                                               input_dropout_prob=input_dropout_prob,
                                               coefficient_dropout_prob=coefficient_dropout_prob,
                                               weight_decay=weight_decay,
                                               name="%s-head%d" % (name, i)))
        if use_averaging:
            return tf.add_n(head_results) / num_heads
        else:
            # dims: num_nodes x output_dim -> num_nodes x num_heads x output_dim
            return tf.concat(head_results, axis=1)


class GAT(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, num_heads, input_dropout_prob, coefficient_dropout_prob, weight_decay,
                 normalize_features, alt_opt):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_dropout_prob = input_dropout_prob
        self.coefficient_dropout_prob = coefficient_dropout_prob
        self.weight_decay = weight_decay
        self.alt_opt = alt_opt

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = attention_layer(inputs=x,
                                    output_dim=self.hidden_size,
                                    num_heads=self.num_heads[i],
                                    graph_adj=self.graph_adj,
                                    adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                    activation_fn=tf.nn.elu,
                                    use_averaging=False,
                                    input_dropout_prob=self.input_dropout_prob,
                                    coefficient_dropout_prob=self.coefficient_dropout_prob,
                                    # L2 regularization is employed on every layer
                                    weight_decay=self.weight_decay,
                                    name="attention_layer%d" % i)
            output = attention_layer(inputs=x,
                                     output_dim=self.targets.shape[1],
                                     num_heads=self.num_heads[self.num_layers - 1],
                                     graph_adj=self.graph_adj,
                                     adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                     activation_fn=None,
                                     use_averaging=True,
                                     input_dropout_prob=self.input_dropout_prob,
                                     coefficient_dropout_prob=self.coefficient_dropout_prob,
                                     weight_decay=self.weight_decay,
                                     name="attention_layer%d" % (self.num_layers - 1))
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        adj_with_self_loops = add_self_loops(graph_adj)
        self.adj_dense_shape = adj_with_self_loops.shape
        adj_with_self_loops_tensor = to_sparse_tensor(adj_with_self_loops)
        adj_with_self_loops_coo = adj_with_self_loops.tocoo()
        # extract the coordinates of all the edges
        # since both row and column coordinates are ordered, row[0] corresponds to col[0] etc.
        self.adj_with_self_loops_indices = np.mat([adj_with_self_loops_coo.row, adj_with_self_loops_coo.col])
        return adj_with_self_loops_tensor

    # override optimize method to employ alternating optimization
    def optimize(self, learning_rate, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if self.alt_opt:
            return [
                optimizer.minimize(self.loss, global_step=global_step, var_list=tf.get_collection(ATTENTION_WEIGHTS)),
                optimizer.minimize(self.loss, global_step=global_step, var_list=tf.get_collection(FILTER_WEIGHTS)),
                ]
        return optimizer.minimize(self.loss, global_step=global_step)


MODEL_INGREDIENT = Ingredient('model')

# differences of jklicpera's and the original model:
# - dropout on inputs is on different place
# - test set size is 860 instead of 1000


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, num_heads, input_dropout_prob, coefficient_dropout_prob, alt_opt):
    input_dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[input_dropout] = input_dropout_prob
    trainval_feed[input_dropout] = False
    val_feed[input_dropout] = False
    test_feed[input_dropout] = False

    coefficient_dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[coefficient_dropout] = coefficient_dropout_prob
    trainval_feed[coefficient_dropout] = False
    val_feed[coefficient_dropout] = False
    test_feed[coefficient_dropout] = False

    return GAT(node_features, graph_adj, labels, dataset_indices_placeholder,
               num_layers=num_layers, hidden_size=hidden_size,
               num_heads=num_heads,
               input_dropout_prob=input_dropout,
               coefficient_dropout_prob=coefficient_dropout,
               weight_decay=weight_decay,
               normalize_features=normalize_features,
               alt_opt=alt_opt)
