import numpy as np
import tensorflow as tf

import gnnbench.metrics
from gnnbench.early_stopping import GCNCriterion, GATCriterion, KDDCriterion, NoStoppingCriterion, \
    GATCriterionWithTolerance


def build_train_ops(sess, model, early_stopping_tolerance, early_stopping_criterion, improvement_tolerance,
                    _run, _log):
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, shape=[])

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    train_step = model.optimize(learning_rate_placeholder, global_step)

    if early_stopping_criterion == 'gcn':
        early_stopping = GCNCriterion(early_stopping_tolerance, sess, _log)
    elif early_stopping_criterion == 'gat':
        early_stopping = GATCriterion(early_stopping_tolerance, sess, _log)
    elif early_stopping_criterion == 'gnnbench':
        early_stopping = KDDCriterion(early_stopping_tolerance, sess, _log)
    elif early_stopping_criterion == "gat_tol":
        early_stopping = GATCriterionWithTolerance(early_stopping_tolerance, improvement_tolerance, sess, _log)
    else:
        _log.warn("Disabled early stopping.")
        early_stopping = NoStoppingCriterion(early_stopping_tolerance, sess, _log)

    init_op = tf.global_variables_initializer()

    return train_step, early_stopping, learning_rate_placeholder, global_step, init_op


def run_train_ops(sess, train_step, early_stopping, learning_rate_placeholder, global_step, init_op,
                  train_feed, trainval_feed, val_feed, test_feed, metrics_to_use,
                  model, learning_rate, num_epochs, early_stopping_criterion,
                  alternating_optimization_interval, lr_decay_factor, lr_decay_steps,
                  report_interval, _run, run_no, _log, traces):
    # reset all training progress
    sess.run(init_op)
    early_stopping.reset()

    start_iteration = sess.run(global_step)

    _log.info("Started training for %s epochs." % num_epochs)

    for epoch in range(start_iteration, num_epochs):

        # add learning rate to feed dict
        train_feed[learning_rate_placeholder] = learning_rate

        if isinstance(train_step, list):
            # triggers alternating optimization as needed for MoNet
            loss_to_optimize = int(epoch / alternating_optimization_interval) % len(train_step)
            _, train_loss = sess.run([train_step[loss_to_optimize], model.loss], feed_dict=train_feed)
        else:
            _, train_loss = sess.run([train_step, model.loss], feed_dict=train_feed)

        # compute metrics needed for logging and early stopping
        val_loss = sess.run(model.loss, feed_dict=val_feed)
        val_metrics = compute_metrics(sess, model, val_feed, metrics_to_use)

        # report some statistics to the logs
        if (epoch + 1) % report_interval == 0:
            train_metrics = compute_metrics(sess, model, trainval_feed, metrics_to_use)
            metrics_string = build_metrics_string(train_loss, val_loss, train_metrics, val_metrics)
            _log.debug(f"After {sess.run(global_step)} epochs:\n" + metrics_string)

            _run.log_scalar(f"train.loss-{run_no}", train_loss, epoch + 1)
            _run.log_scalar(f"val.loss-{run_no}", val_loss, epoch + 1)
            for name, value in train_metrics.items():
                _run.log_scalar(f"train.{name}-{run_no}", value, epoch + 1)
            for name, value in val_metrics.items():
                _run.log_scalar(f"val.{name}-{run_no}", value, epoch + 1)

        # test early stopping criterion
        if early_stopping.should_stop(epoch, val_loss, val_metrics['accuracy']):
            _log.debug(f"Early stopping by {early_stopping_criterion} criterion"
                       f" after {sess.run(global_step)} epochs.")
            break

        # decay learning rate if enabled
        if lr_decay_factor > 0.0 and lr_decay_steps:
            if lr_decay_factor >= 1.0:
                raise ValueError(f"A learning rate decay factor of {lr_decay_factor} will not "
                                 f"decay the learning rate.")
            if epoch + 1 in lr_decay_steps:
                learning_rate *= lr_decay_factor
                _log.debug(f"Decaying learning rate to {learning_rate}.")

    # run after-stopping operations of early stopping criterion
    early_stopping.after_stopping_ops()

    # after-training operations
    final_train_loss = sess.run(model.loss, feed_dict=trainval_feed)
    final_val_loss = sess.run(model.loss, feed_dict=val_feed)
    final_train_metrics = compute_metrics(sess, model, trainval_feed, metrics_to_use)
    final_val_metrics = compute_metrics(sess, model, val_feed, metrics_to_use)

    final_metrics_string = build_metrics_string(final_train_loss, final_val_loss,
                                                final_train_metrics, final_val_metrics)
    _log.debug(f"---\nTraining finished after {sess.run(global_step)} epochs. Final values:\n" + final_metrics_string)

    _log.info("Evaluating on test set.")
    final_test_metrics = compute_metrics(sess, model, test_feed, metrics_to_use)
    _log.debug("\n".join(f"Test {name} = {value:.4f}" for name, value in final_test_metrics.items()))

    traces['train.loss'].append(final_train_loss)
    traces['val.loss'].append(final_val_loss)
    for name, value in final_train_metrics.items():
        traces[f'train.{name}'].append(value)
    for name, value in final_val_metrics.items():
        traces[f'val.{name}'].append(value)
    for name, value in final_test_metrics.items():
        traces[f'test.{name}'].append(value)

    return final_test_metrics


def compute_metrics(sess, model, feed, metrics_to_use):
    predictions, ground_truth = sess.run([model.predict, model.targets], feed_dict=feed)
    predictions = np.argmax(predictions, axis=1)
    ground_truth = np.argmax(ground_truth, axis=1)
    return {name: getattr(gnnbench.metrics, name)(ground_truth, predictions) for name in metrics_to_use}


def build_metrics_string(train_loss, val_loss, train_metrics, val_metrics):
    train_part = f"    - " + "; ".join(
        [f"train loss = {train_loss:.4f}"] + [f"train {name} = {value:.4f}" for name, value in train_metrics.items()]
    )
    val_part = f"    - " + "; ".join(
        [f"val loss = {val_loss:.4f}"] + [f"val {name} = {value:.4f}" for name, value in val_metrics.items()]
    )
    return train_part + "\n" + val_part + "\n"
