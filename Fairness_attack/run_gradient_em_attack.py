from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import argparse
import time

import numpy as np

import scipy.sparse as sparse
import data_utils as data
import datasets
import iterative_attack
from influence.influence.smooth_hinge import SmoothHinge
from influence.influence.dataset import DataSet

import tensorflow as tf


def run_attack(
        total_grad_iter=300,
        use_slab=True,
        dataset="german",
        percentile=90,
        epsilon=0.5,
        lamb=1,
        weight_decay=0.09,
        step_size=0.1,
        no_LP=False,
        timed=False,
        sensitive_feature_idx=0,
        method="IAF",
        stop_after=2,
        batch_size=1,
        eval_mode=False,
        iter_to_load=0,
        stopping_method='Accuracy',
        log_metrics=False,
        display_iter_time=False,
        seed=1):
    """
    :param total_grad_iter: Number of maximum iterations of the attack
    :param use_slab: Use the intersection between the L2 and the slab defense
    :param dataset: Name of the dataset
    :param percentile: Percentage of data kept in the feasible set
    :param epsilon: Ratio that defines the number of poisoned datapoints as epsilon * len(dataset.train),
    :param lamb=: Ratio of adversarial loss for IAF attack (l_accuracy + lamb * l_fairness)
    :param weight_decay: Amount of weight_decay in SVM training
    :param step_size: Step size for gradient update of poisoned points (IAF, Koh, Solans)
    :param no_LP: Don't use LP rounding
    :param timed: Time the attack iterations
    :param sensitive_feature_idx: Index in dataset for sensitive feature,
    :param method: Attacking method; valid options: 'IAF', 'RAA', 'NRAA', 'Koh', 'Solans'
    :param stop_after: Patience for stopping training
    :param batch_size: Training batch size (currently not used)
    :param eval_mode: Activate eval mode (no training)
    :param iter_to_load: not used (can be edited to load specific checkpoint itertions)
    :param stopping_method: Metric to evaluate best model during training; valid options: 'Accuracy', 'Fairness'
    :param log_metrics: Save logging of accuracy and fariness metrics average during training in json. file
    :param display_iter_time: Print time of each iteration
    :param seed: random seed
    """

    def get_projection_fn_for_dataset(X, Y, use_slab, use_LP, percentile):
        projection_fn = data.get_projection_fn(
            X, Y,
            sphere=True,
            slab=use_slab,
            non_negative=True,
            less_than_one=False,
            use_lp_rounding=use_LP,
            percentile=percentile)

        return projection_fn

    # Make sure the variables have the correc type. If the arguments are taken from the command line,
    # they will be string and will need to be converted.
    seed = int(seed)
    epsilon = float(epsilon)
    step_size = float(step_size)
    percentile = int(np.round(float(percentile)))
    total_grad_iter = int(np.round(float(total_grad_iter)))
    print('ATTACK METHOD', method)
    sensitive_idx = int(sensitive_feature_idx)
    sensitive_file = f"{dataset}_group_label.npz"
    lamb = float(lamb)
    weight_decay = float(weight_decay)
    stop_after = int(stop_after)
    batch_size = int(batch_size)
    eval_mode = bool(eval_mode)
    log_metrics = bool(log_metrics)
    display_iter_time = bool(display_iter_time)
    output_root = os.path.join(
        datasets.OUTPUT_FOLDER, dataset, 'influence_data')

    np.random.seed(seed)
    datasets.safe_makedirs(output_root)

    print('EVAL MODE IS ', eval_mode)
    if(method == "IAF" or method == "Solans"):
        loss_type = 'adversarial_loss'
    elif(method == "Koh"):
        loss_type = 'adversarial_loss'
        lamb = 0
    else:
        loss_type = 'normal_loss'

    print('epsilon: %s' % epsilon)
    print('use_slab: %s' % use_slab)

    # Not used leftover for use in possible extensions (different optimization)
    initial_learning_rate = 0.001
    temp = 0.001  # Delta for smooth hinge loss
    use_copy = True  # Copy the poisened points to get the specified amount given by epsilon
    use_slab = True  # Use intersection between slab and L2 defense (always on)
    use_LP = False if no_LP else True  # Use LP rounding
    num_classes = 2  # Only binary classification possible

    model_name = str(dataset) + '_' + str(method) + '_' + \
        str(epsilon) + '_' + str(lamb) + '_' + str(stopping_method)

    if no_LP:
        model_name = model_name + '_no-LP'
    if timed:
        model_name = model_name + '_timed'
    if epsilon == 0.0:
        model_name = str(dataset) + '_no_attack'

    print(model_name)

    X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset)

    general_train_idx = X_train.shape[0]

    if sparse.issparse(X_train):
        X_train = X_train.toarray()
    if sparse.issparse(X_test):
        X_test = X_test.toarray()

    advantaged = 1

    # Some things only needed for Solans
    p_over_m = 1
    advantaged_group_selector = np.zeros(1)
    disadvantaged_group_selector = np.zeros(1)

    if epsilon > 0:
        print(epsilon)
        class_map, centroids, centroid_vec, sphere_radii, slab_radii = data.get_data_params(
            X_train, Y_train, percentile=percentile)

        feasible_flipped_mask = iterative_attack.get_feasible_flipped_mask(
            X_train, Y_train,
            centroids,
            centroid_vec,
            sphere_radii,
            slab_radii,
            class_map,
            use_slab=use_slab)

        X_modified, Y_modified, indices_to_poison, copy_array, advantaged, test_gender_labels = iterative_attack.init_gradient_attack_from_mask(
            X_train, Y_train,
            epsilon,
            feasible_flipped_mask,
            general_train_idx,
            sensitive_file,
            method,
            use_copy=use_copy)

        if method == "Solans":
            disadvantaged = -1 * advantaged

            # compute indices for advantaged and disadvantaged group
            advantaged_group_selector = np.where(
                test_gender_labels == advantaged)[0]
            disadvantaged_group_selector = np.where(
                test_gender_labels == disadvantaged)[0]

            # compute ratio of advantaged and disadvantaged group count
            p_over_m = len(disadvantaged_group_selector) / \
                len(advantaged_group_selector)

    tf.compat.v1.reset_default_graph()

    input_dim = X_train.shape[1]
    # Datasets include initial poisoned points if an attack is used
    train = DataSet(X_modified, Y_modified) if epsilon != 0.0 else DataSet(
        X_train, Y_train)
    validation = None
    test = DataSet(X_test, Y_test)

    model = SmoothHinge(
        sensitive_feature_idx=sensitive_idx,
        input_dim=input_dim,
        temp=temp,
        weight_decay=weight_decay,
        use_bias=True,
        advantaged=advantaged,
        num_classes=num_classes,
        batch_size=batch_size,
        train_dataset=train,
        validation_dataset=validation,
        test_dataset=test,
        initial_learning_rate=initial_learning_rate,
        decay_epochs=None,
        mini_batch=False,
        train_dir=output_root,
        log_dir='log',
        model_name=model_name,
        method=method,
        general_train_idx=general_train_idx,
        sensitive_file=sensitive_file,
        lamb=lamb,
        p_over_m=p_over_m,
        advantaged_group_selector=advantaged_group_selector,
        disadvantaged_group_selector=disadvantaged_group_selector,
        seed=seed,
        eval_mode=eval_mode,
        stopping_method=stopping_method,
        log_metrics=log_metrics,
        display_iter_time=display_iter_time)

    # If the evaluation of the model takes place, then we skip train and just do eval
    if(eval_mode == True):
        model.checkpoint_file = os.path.join(
            model.train_dir, "%s-checkpoint" % model_name)
        print('MODEL CHECKPOINT NAME \n', model.checkpoint_file)
        results = model.load_checkpoint(int(iter_to_load), do_checks=True)
        print(results)
        return results

    model.train()

    # only attack for valid epsilon
    if epsilon > 0:

        if timed:
            start_time = time.time()
        else:
            start_time = None

        X_modified = model.train_dataset.x
        Y_modified = model.train_dataset.labels

        projection_fn = get_projection_fn_for_dataset(
            X_modified,
            Y_modified,
            use_slab,
            use_LP,
            percentile)

        iterative_attack.iterative_attack(
            model,
            general_train_idx,
            sensitive_file,
            method,
            advantaged,
            indices_to_poison=indices_to_poison,
            test_idx=None,
            test_description=None,
            step_size=step_size,
            num_iter=total_grad_iter,
            loss_type=loss_type,
            projection_fn=projection_fn,
            output_root=output_root,
            num_copies=copy_array,
            stop_after=stop_after,
            start_time=start_time,
            display_iter_time=display_iter_time,
            stopping_method=stopping_method)
        print("The end")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--total_grad_iter', default=300,
                        help="Maximum number of attack iterations")
    parser.add_argument('--dataset', default='german',
                        help="Specify dataset file name")
    parser.add_argument('--percentile', default=90,
                        help="Percentage of data to keep in feasible set")
    parser.add_argument('--epsilon', default=0.5,
                        help="Partial of number of datapoints in training set as number of poisoned points to create (i.e. n_poisoned = epsilon*len(training_data)")
    parser.add_argument('--lamb', default=1.,
                        help="Adversarial loss lambda (IAF) controlling trade off between accuracy and fairness")
    parser.add_argument('--weight_decay', default=0.09,
                        help="Specify weight decay for regularization")
    parser.add_argument('--step_size', default=0.1,
                        help="Step size for poisoned point update using adversarial loss")
    parser.add_argument('--no_LP', action="store_true",
                        help="Don't use LP rounding")
    parser.add_argument('--timed', action="store_true",
                        help="Activate timed")
    parser.add_argument('--sensitive_feature_idx', default=0,
                        help="Sensitive group feature index in data")
    parser.add_argument('--method', default="IAF",
                        help="specify attack method out of 'IAF', 'RAA', 'NRAA', 'Koh', 'Solans'")
    parser.add_argument('--stop_after', default='2',
                        help='Specify after how many iterations without improving the attack should stop')
    parser.add_argument('--batch_size', default=1,
                        help="Specify batch size (Note: in the current implementation no mini-batch training is used. This is leftover for use in possible extensions")
    parser.add_argument('--eval_mode', default=False,
                        help="Evaluation or training mode")
    parser.add_argument('--iter_to_load', default=0,
                        help="Number of the interation of the checkpoint to load")
    parser.add_argument('--stopping_method', default='Accuracy',
                        help="The metric on which the early stopping is based. Choose between 'Accuracy' and 'Fairness'.")
    parser.add_argument('--log_metrics', default=False,
                        help="Log metrics for training one model, and export them as .json")
    parser.add_argument('--display_iter_time', default=False,
                        help="Print time required to run training iteration")
    parser.add_argument('--seed', default=1, help='Specify random seed')

    args = parser.parse_args().__dict__

    run_attack(**args)
