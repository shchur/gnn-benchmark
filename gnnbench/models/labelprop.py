import numpy as np
import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import normalize_adj
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor


class LabelPropagation(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider, labelled_nodes, prop_type, return_prob):

        if prop_type not in ['vanilla', 'smoothed']:
            raise ValueError('Unsupported propagation type.')
        self.prop_type = prop_type

        # if running on Planetoid data these typecasts are necessary
        if isinstance(labelled_nodes, range):
            labelled_nodes = np.array(list(labelled_nodes), dtype=np.int64)
        if targets.dtype != np.float32:
            targets = targets.astype(np.float32)

        super().__init__(features, graph_adj, tf.gather(targets, nodes_to_consider))
        self.labelled_nodes = tf.constant(labelled_nodes, dtype=tf.int64)
        self.initial_predicted_labels = tf.scatter_nd(tf.expand_dims(self.labelled_nodes, -1),
                                                      targets[labelled_nodes], shape=targets.shape)
        self.predicted_labels = tf.Variable(self.initial_predicted_labels, dtype=tf.float32, name="predicted_labels")

        self.nodes_to_consider = nodes_to_consider
        self.num_nodes = int(self.graph_adj.get_shape()[0])
        self.num_classes = int(self.targets.get_shape()[1])

        self.return_prob = return_prob

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(self.predicted_labels, self.nodes_to_consider)

    def optimize(self, learning_rate, global_step):
        if self.prop_type == 'vanilla':
            # dims: num_nodes x num_nodes, num_nodes x num_labels, num_nodes -> num_nodes x num_labels
            new_predicted_labels = tf.sparse_tensor_dense_matmul(self.graph_adj, self.predicted_labels) / self.degrees
            # set entries where we have a label to zero...
            new_predicted_labels *= self._get_labelled_nodes_mask()
            # ... and add already known labels
            new_predicted_labels += self.initial_predicted_labels
        else:
            new_predicted_labels = (1 - self.return_prob) * tf.sparse_tensor_dense_matmul(self.graph_adj,
                                                                                          self.predicted_labels) \
                                   + self.return_prob * self.initial_predicted_labels

        # update predictions variable
        predicted_labels_update_op = self.predicted_labels.assign(new_predicted_labels)
        return predicted_labels_update_op, global_step.assign_add(1)

    def _get_labelled_nodes_mask(self):
        inv_mask = tf.scatter_nd(
            tf.expand_dims(self.labelled_nodes, -1),
            tf.ones([self.labelled_nodes.get_shape()[0], self.num_classes], dtype=tf.float32),
            shape=[self.num_nodes, self.num_classes]
        )
        return 1 - inv_mask

    def _preprocess_features(self, features):
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        self.degrees = graph_adj.sum(axis=1).astype(np.float32)
        if self.prop_type == 'smoothed':
            graph_adj = normalize_adj(graph_adj)
        return to_sparse_tensor(graph_adj)


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features, prop_type, return_prob):

    return LabelPropagation(node_features, graph_adj, labels, dataset_indices_placeholder,
                            labelled_nodes=train_feed[dataset_indices_placeholder],
                            prop_type=prop_type, return_prob=return_prob)
