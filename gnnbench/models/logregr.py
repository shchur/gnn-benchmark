import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor


class LogisticRegression(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider, weight_decay, normalize_features):
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.weight_decay = weight_decay

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            weights = tf.get_variable("weights", [int(self.features.get_shape()[1]), self.targets.shape[1]],
                                      dtype=tf.float32,
                                      initializer=tf.glorot_uniform_initializer(),
                                      regularizer=slim.l2_regularizer(self.weight_decay))
            output = tf.sparse_tensor_dense_matmul(self.features, weights)
            output = tf.contrib.layers.bias_add(output)

        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        return to_sparse_tensor(graph_adj)


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features):
    return LogisticRegression(node_features, graph_adj, labels, dataset_indices_placeholder,
                              weight_decay=weight_decay,
                              normalize_features=normalize_features)
