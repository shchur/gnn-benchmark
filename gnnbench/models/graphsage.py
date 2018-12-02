import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor, scatter_add_tensor, dropout_supporting_sparse_tensors


def aggregate_mean(transformed_features, graph_adj, degrees, name):
    with tf.name_scope(name):
        # Note: this is very similar to the graph convolution operation defined in gcn.py
        # dims: num_nodes x num_nodes, num_nodes x num_features -> num_nodes x num_features
        output = tf.sparse_tensor_dense_matmul(graph_adj, transformed_features)
        # dims: num_nodes x num_features, num_nodes -> num_nodes x num_features
        return output / degrees


# dims:
#   features: num_nodes x num_features
#   adj_with_self_loops_indices: num_nodes x max_degree
def aggregate_maxpool(features, agg_transform_size, adj_with_self_loops_indices, num_features, name):
    with tf.name_scope(name):
        fc_weights = tf.get_variable(f"{name}-fc_weights",
                                     shape=[num_features, agg_transform_size],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer(),
                                     )
        # dims: num_nodes x num_features, num_features x agg_transform_size -> num_nodes x agg_transform_size
        if isinstance(features, tf.SparseTensor):
            transformed_features = tf.sparse_tensor_dense_matmul(features, fc_weights)
        else:
            transformed_features = tf.matmul(features, fc_weights)
        transformed_features = tf.nn.relu(transformed_features)

        # Spread out the transformed features to neighbours.
        # dims: num_nodes x agg_transform_size, num_nodes x max_degree -> num_nodes x agg_transform_size x max_degree
        neighbours_features = tf.gather(transformed_features, adj_with_self_loops_indices)

        # employ the max aggregator
        output = tf.reduce_max(neighbours_features, axis=1)
        return output


# dims:
#   features: num_nodes x num_features
def aggregate_meanpool(features, agg_transform_size, adj_with_self_loops_indices, degrees, num_nodes, num_features,
                       name):
    with tf.name_scope(name):
        self_indices, neighbours_indices = adj_with_self_loops_indices

        fc_weights = tf.get_variable(f"{name}-fc_weights",
                                     shape=[num_features, agg_transform_size],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer(),
                                     )
        # dims: num_nodes x num_features, num_features x agg_transform_size -> num_nodes x agg_transform_size
        if isinstance(features, tf.SparseTensor):
            transformed_features = tf.sparse_tensor_dense_matmul(features, fc_weights)
        else:
            transformed_features = tf.matmul(features, fc_weights)
        transformed_features = tf.nn.relu(transformed_features)

        # Spread out the transformed features to num_edges.
        # dims: num_nodes x agg_transform_size, num_edges -> num_edges x agg_transform_size
        edge_features = tf.gather(transformed_features, neighbours_indices)

        # employ the mean aggregator
        # dims: num_edges x agg_transform_size, num_edges -> num_nodes x agg_transform_size
        output = scatter_add_tensor(edge_features, self_indices, out_shape=[num_nodes, agg_transform_size])
        # divide sum by degree to get mean
        # dims: num_nodes x agg_transform_size, num_nodes -> num_nodes x agg_transform_size
        output = output / degrees
        return output


def sage_layer(features, output_dim, graph_adj, adj_with_self_loops_indices, degrees,
               aggregator, agg_transform_size, activation_fn,
               weight_decay, dropout_prob, name):
    with tf.name_scope(name):
        num_nodes = int(features.get_shape()[0])
        num_features = int(features.get_shape()[1])

        # Apply dropout to features if required. Put this here as GCN also uses dropout at this place.
        features = tf.cond(
            tf.cast(dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(features, 1 - dropout_prob)),
            false_fn=(lambda: features),
        )

        # dims: num_nodes x num_features -> num_nodes x aggregated_feature_size
        agg_name = f"{name}-aggregator"
        if aggregator == 'mean' or aggregator == 'gcn':
            # dims: aggregated_feature_size x output_dim
            agg_weights = tf.get_variable(f"{name}-agg-weights",
                                          shape=[num_features, output_dim],
                                          dtype=tf.float32,
                                          initializer=tf.glorot_uniform_initializer(),
                                          regularizer=slim.l2_regularizer(weight_decay)
                                          )
            if isinstance(features, tf.SparseTensor):
                transformed_features = tf.sparse_tensor_dense_matmul(features, agg_weights)
            else:
                transformed_features = tf.matmul(features, agg_weights)
            agg_features = aggregate_mean(transformed_features, graph_adj, degrees, agg_name)

        # meanpool and maxpool receive special treatment since the have the additional transformation beforehand
        elif aggregator == 'meanpool' or aggregator == 'maxpool':
            if aggregator == 'meanpool':
                aggregated = aggregate_meanpool(features, agg_transform_size, adj_with_self_loops_indices, degrees,
                                                num_nodes,
                                                num_features, agg_name)
            elif aggregator == 'maxpool':
                aggregated = aggregate_maxpool(features, agg_transform_size, adj_with_self_loops_indices,
                                               num_features, agg_name)
            else:
                raise ValueError('Undefined aggregator.')

            # dims: aggregated_feature_size x output_dim
            agg_weights = tf.get_variable(f"{name}-agg-weights",
                                          shape=[int(aggregated.get_shape()[1]), output_dim],
                                          dtype=tf.float32,
                                          initializer=tf.glorot_uniform_initializer(),
                                          regularizer=slim.l2_regularizer(weight_decay)
                                          )
            # dims: num_nodes x aggregated_feature_size, aggregated_feature_size x output_dim -> num_nodes x output_dim
            agg_features = tf.matmul(aggregated, agg_weights)
        else:
            raise ValueError('Undefined aggregator.')

        if aggregator == 'gcn':
            # to compare to GCN, do not use skip connections
            output = agg_features
        else:
            # dims: num_features x output_dim
            skip_conn_weights = tf.get_variable(f"{name}-skip_conn-weights",
                                                shape=[num_features, output_dim],
                                                dtype=tf.float32,
                                                initializer=tf.glorot_uniform_initializer(),
                                                regularizer=slim.l2_regularizer(weight_decay)
                                                )
            # dims: num_nodes x num_features, num_features x output_dim -> num_nodes x output_dim
            if isinstance(features, tf.SparseTensor):
                skip_features = tf.sparse_tensor_dense_matmul(features, skip_conn_weights)
            else:
                skip_features = tf.matmul(features, skip_conn_weights)

            output = agg_features + skip_features

        output = tf.contrib.layers.bias_add(output)

        # This normalization strongly improves performance. It is introduced in the original algorithm from the paper.
        # The value clipping of the normalization constant is taken from
        # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/sage_conv.py
        if aggregator != 'gcn':
            normalizer = tf.norm(output, ord=2, axis=1, keepdims=True)
            normalizer = tf.clip_by_value(normalizer, clip_value_min=1.0, clip_value_max=np.inf)
            output = output / normalizer

        if activation_fn is not None:
            output = activation_fn(output)
        return output


class GraphSAGE(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, aggregator, agg_transform_size, dropout_prob, weight_decay,
                 normalize_features):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        self.aggregator = aggregator
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.agg_transform_size = agg_transform_size
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = sage_layer(features=x,
                               output_dim=self.hidden_size,
                               graph_adj=self.graph_adj,
                               adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                               degrees=self.degrees,
                               aggregator=self.aggregator,
                               agg_transform_size=self.agg_transform_size,
                               activation_fn=tf.nn.relu,
                               # use the same training parameters as GCN which only uses weight decay on first layer
                               weight_decay=self.weight_decay if i == 0 else 0.0,
                               dropout_prob=self.dropout_prob,
                               name=f"sage_layer{i}")
            output = sage_layer(features=x,
                                output_dim=self.targets.shape[1],
                                graph_adj=self.graph_adj,
                                adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                degrees=self.degrees,
                                aggregator=self.aggregator,
                                agg_transform_size=self.agg_transform_size,
                                activation_fn=None,
                                weight_decay=0.0,
                                dropout_prob=self.dropout_prob,
                                name=f"sage_layer{(self.num_layers - 1)}")
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
        self.degrees = adj_with_self_loops.sum(axis=1).astype(np.float32)
        if self.aggregator == 'maxpool':
            # maxpool aggregator requires special preprocessing:
            # Fill a matrix with up to max_degree neighbours. max_degree is the maximum degree appearing in the
            # graph. If a node has less than max_degree neighbours, the remaining entries in the matrix are set
            # to 0. This finally refers to node 0. However, for the maxpool aggregator, duplicates do not matter
            # since the max-operation is invariant to duplicates in the input.
            neighbours_matrix = np.zeros((self.num_nodes, int(np.max(self.degrees))), dtype=np.int32)
            insert_index = 0
            self_node_old = 0
            for i, self_node in enumerate(adj_with_self_loops_coo.row):
                if self_node != self_node_old:
                    insert_index = 0
                neighbours_matrix[self_node, insert_index] = adj_with_self_loops_coo.col[i]
                insert_index += 1
                self_node_old = self_node
            self.adj_with_self_loops_indices = neighbours_matrix
        else:
            # extract the coordinates of all the edges
            # since both row and column coordinates are ordered, row[0] corresponds to col[0] etc.
            self.adj_with_self_loops_indices = np.mat([adj_with_self_loops_coo.row, adj_with_self_loops_coo.col])
        return adj_with_self_loops_tensor


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, aggregator, agg_transform_size, dropout_prob):
    dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[dropout] = dropout_prob
    trainval_feed[dropout] = False
    val_feed[dropout] = False
    test_feed[dropout] = False

    return GraphSAGE(node_features, graph_adj, labels, dataset_indices_placeholder,
                     num_layers=num_layers, hidden_size=hidden_size,
                     dropout_prob=dropout,
                     weight_decay=weight_decay,
                     aggregator=aggregator,
                     agg_transform_size=agg_transform_size,
                     normalize_features=normalize_features)
