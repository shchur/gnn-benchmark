import tensorflow as tf
import tensorflow.contrib.slim as slim
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize
from gnnbench.models.base_model import GNNModel
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor


def fully_connected_layer(inputs, output_dim, activation_fn, dropout_prob, weight_decay, name):
    with tf.name_scope(name):
        input_dim = int(inputs.get_shape()[1])
        weights = tf.get_variable("%s-weights" % name, [input_dim, output_dim], dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer(),
                                  regularizer=slim.l2_regularizer(weight_decay))

        # Apply dropout to inputs if required
        inputs = tf.cond(
            tf.cast(dropout_prob, tf.bool),
            true_fn=(lambda: dropout_supporting_sparse_tensors(inputs, 1 - dropout_prob)),
            false_fn=(lambda: inputs),
        )

        if isinstance(inputs, tf.SparseTensor):
            output = tf.sparse_tensor_dense_matmul(inputs, weights)
        else:
            output = tf.matmul(inputs, weights)
        output = tf.contrib.layers.bias_add(output)
        return activation_fn(output) if activation_fn else output


class MLP(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, dropout_prob, weight_decay, normalize_features):
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay

        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = fully_connected_layer(
                    inputs=x,
                    output_dim=self.hidden_size,
                    activation_fn=tf.nn.relu,
                    dropout_prob=self.dropout_prob,
                    weight_decay=self.weight_decay,
                    name="fc%d" % i,
                )
            output = fully_connected_layer(
                inputs=x,
                output_dim=self.targets.shape[1],
                activation_fn=None,
                dropout_prob=self.dropout_prob,
                weight_decay=self.weight_decay,
                name="fc%d" % (self.num_layers - 1),
            )
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
                weight_decay, normalize_features,
                num_layers, hidden_size, dropout_prob):
    dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[dropout] = dropout_prob
    trainval_feed[dropout] = False
    val_feed[dropout] = False
    test_feed[dropout] = False

    return MLP(node_features, graph_adj, labels, dataset_indices_placeholder,
               num_layers=num_layers, hidden_size=hidden_size,
               dropout_prob=dropout,
               weight_decay=weight_decay,
               normalize_features=normalize_features)
