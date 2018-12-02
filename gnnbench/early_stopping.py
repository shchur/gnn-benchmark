import numpy as np
import tensorflow as tf


class EarlyStoppingCriterion(object):
    def __init__(self, patience, sess, _log):
        self.patience = patience
        self.sess = sess

        self._log = _log

    def should_stop(self, epoch, val_loss, val_accuracy):
        raise NotImplementedError

    def after_stopping_ops(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class NoStoppingCriterion(EarlyStoppingCriterion):

    def should_stop(self, epoch, val_loss, val_accuracy):
        return False

    def after_stopping_ops(self):
        pass

    def reset(self):
        pass


class GCNCriterion(EarlyStoppingCriterion):
    def __init__(self, patience, sess, _log):
        super().__init__(patience, sess, _log)
        self.val_losses = []

    def should_stop(self, epoch, val_loss, val_accuracy):
        self.val_losses.append(val_loss)

        return epoch >= self.patience and self.val_losses[-1] > np.mean(
            self.val_losses[-(self.patience + 1):-1])

    def after_stopping_ops(self):
        pass

    def reset(self):
        self.val_losses = []


class CriterionWithVariablesReset(EarlyStoppingCriterion):
    def __init__(self, patience, sess, _log):
        super().__init__(patience, sess, _log)
        self.extract_variables_state_op, self.variables_state, self.set_trainable_variables_op = \
            get_reset_variable_ops()
        self.best_step = 0
        self.best_variable_state = None

    def should_stop(self, epoch, val_loss, val_accuracy):
        raise NotImplementedError

    def after_stopping_ops(self):
        self._log.debug(f"Resetting to best state of variables which occurred at step {self.best_step + 1}.")
        self.sess.run(self.set_trainable_variables_op,
                      feed_dict={placeholder: self.best_variable_state[i] for i, placeholder in
                                 enumerate(self.variables_state)})

    def reset(self):
        self.best_step = 0
        self.best_variable_state = self.sess.run(self.extract_variables_state_op)


class GATCriterion(CriterionWithVariablesReset):
    def __init__(self, patience, sess, _log):
        super().__init__(patience, sess, _log)
        self.val_accuracy_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_accuracy >= self.val_accuracy_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved
            self.val_accuracy_max = np.max((val_accuracy, self.val_accuracy_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = self.sess.run(self.extract_variables_state_op)
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_accuracy_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0


class KDDCriterion(CriterionWithVariablesReset):
    def __init__(self, patience, sess, _log):
        super().__init__(patience, sess, _log)
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        # only pay attention to validation loss
        if val_loss <= self.val_loss_min:
            # val loss improved
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = self.sess.run(self.extract_variables_state_op)
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_loss_min = np.inf
        self.patience_step = 0


class GATCriterionWithTolerance(GATCriterion):
    def __init__(self, patience, tolerance, sess, _log):
        super().__init__(patience, sess, _log)
        self.tolerance = tolerance

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_accuracy >= self.val_accuracy_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved, so we have a new best state
            self.val_accuracy_max = np.max((val_accuracy, self.val_accuracy_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.best_step = epoch
            self.best_variable_state = self.sess.run(self.extract_variables_state_op)

            # But only reset patience if accuracy or loss improved by a certain degree. This avoids long-running
            # convergence processes like for the LabelPropagation algorithm.
            if val_accuracy >= self.val_accuracy_max + self.tolerance or val_loss <= self.val_loss_min - self.tolerance:
                self.patience_step = 0
            else:
                self.patience_step += 1
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience


def get_reset_variable_ops():
    # Operations for resetting the state of trainable variables to the one at the minimum val loss step.
    with tf.name_scope('reset_variable_state_ops'):
        extract_variables_state_op = extract_variables_state()
        variables_state = [tf.placeholder(var.dtype, var.shape) for var in tf.trainable_variables()]
        set_trainable_variables_op = set_trainable_variables(variables_state)
    return extract_variables_state_op, variables_state, set_trainable_variables_op


def extract_variables_state():
    """Code taken from jklicpera's Master's Thesis."""
    return tf.trainable_variables()


def set_trainable_variables(variables_state):
    """Code taken from jklicpera's Master's Thesis."""
    return [var.assign(variables_state[i]) for i, var in enumerate(tf.trainable_variables())]
