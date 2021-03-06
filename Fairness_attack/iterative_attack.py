from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import numpy as np

import os
import defenses
import data_utils as data
import cvxpy as cvx
import tensorflow as tf
import random


def poison_with_influence_proj_gradient_step(model, general_train_idx,
                                             sensitive_file, attack_method, advantaged, test_idx, indices_to_poison,
                                             projection_fn,
                                             step_size=0.01,
                                             shrink_towards='cluster_center',
                                             loss_type='normal_loss',
                                             force_refresh=True,
                                             test_description=None,
                                             output_root=None):
    """
    Returns poisoned_X_train, a subset of model.train_dataset (marked by indices_to_poison)
    that has been modified by a attack iteration step.
    """
    train_dataset = model.train_dataset
    validation_dataset = model.validation_dataset
    test_dataset = model.test_dataset

    if test_description is None:
        test_description = test_idx
    grad_filename = os.path.join(output_root, 'grad_influence_wrt_input_val_%s_testidx_%s.npy' % (
        model.model_name, test_description))

    if (force_refresh == False) and (os.path.exists(grad_filename)):
        grad_influence_wrt_input_val = np.load(grad_filename)
    else:
        grad_influence_wrt_input_val = model.get_grad_of_influence_wrt_input(
            indices_to_poison,
            test_idx,
            verbose=False,
            force_refresh=force_refresh,
            test_description=test_description,
            loss_type=loss_type)

    poisoned_X_train = train_dataset.x[indices_to_poison, :]

    poisoned_X_train -= step_size * grad_influence_wrt_input_val

    poisoned_labels = train_dataset.labels[indices_to_poison]

    weights = model.sess.run(model.weights)

    if(attack_method == "RAA"):
        DATA_FOLDER = './data'
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, sensitive_file))
        group_label = f['group_label']

        male_train_index = np.where(group_label[0:general_train_idx] == 0)[
            0].astype(np.int32)
        female_train_index = np.where(group_label[0:general_train_idx] == 1)[
            0].astype(np.int32)
        male_test_index = np.where(group_label[general_train_idx:] == 0)[
            0].astype(np.int32)
        female_test_index = np.where(group_label[general_train_idx:] == 1)[
            0].astype(np.int32)

        gender_labels = np.zeros(train_dataset.labels.shape[0])
        for k in range(general_train_idx):
            if(k in male_train_index):
                gender_labels[k] = 1
            elif(k in female_train_index):
                gender_labels[k] = -1

        if(advantaged == -1):
            op_indx = np.where((train_dataset.labels == -1)
                               & (gender_labels == -1))[0]
        else:
            op_indx = np.where((train_dataset.labels == -1)
                               & (gender_labels == 1))[0]

        rand1 = random.randint(0, op_indx.shape[0] - 1)

        print("Hello\n" * 3)
        print(rand1)

        poisoned_X_train[0] = train_dataset.x[op_indx[rand1], :]

        if(advantaged == -1):
            op_indx = np.where((train_dataset.labels == 1)
                               & (gender_labels == 1))[0]
        else:
            op_indx = np.where((train_dataset.labels == 1)
                               & (gender_labels == -1))[0]

        rand2 = random.randint(0, op_indx.shape[0] - 1)
        poisoned_X_train[1] = train_dataset.x[op_indx[rand2], :]

    elif(attack_method == "NRAA"):
        DATA_FOLDER = './data'
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, sensitive_file))
        group_label = f['group_label']

        male_train_index = np.where(group_label[0:general_train_idx] == 0)[
            0].astype(np.int32)
        female_train_index = np.where(group_label[0:general_train_idx] == 1)[
            0].astype(np.int32)
        male_test_index = np.where(group_label[general_train_idx:] == 0)[
            0].astype(np.int32)
        female_test_index = np.where(group_label[general_train_idx:] == 1)[
            0].astype(np.int32)

        gender_labels = np.zeros(train_dataset.labels.shape[0])
        for k in range(general_train_idx):
            if(k in male_train_index):
                gender_labels[k] = 1
            elif(k in female_train_index):
                gender_labels[k] = -1

        if(advantaged == -1):
            op_indx = np.where((train_dataset.labels == -1)
                               & (gender_labels == -1))[0]
        else:
            op_indx = np.where((train_dataset.labels == -1)
                               & (gender_labels == 1))[0]

        maxdist = 0
        maxpoint = 0
        for points in range(op_indx.shape[0]):
            temp = 0
            for p in range(op_indx.shape[0]):
                if(np.allclose(train_dataset.x[op_indx[points], :], train_dataset.x[op_indx[p], :], rtol=0, atol=1)):
                    temp = temp + 1
            if(temp > maxdist):
                maxdist = temp
                maxpoint = points

        poisoned_X_train[0] = train_dataset.x[op_indx[maxpoint], :]

        if(advantaged == -1):
            op_indx = np.where((train_dataset.labels == 1)
                               & (gender_labels == 1))[0]
        else:
            op_indx = np.where((train_dataset.labels == 1)
                               & (gender_labels == -1))[0]

        maxdist = 0
        maxpoint = 0
        for points in range(op_indx.shape[0]):
            temp = 0
            for p in range(op_indx.shape[0]):
                if(np.allclose(train_dataset.x[op_indx[points], :], train_dataset.x[op_indx[p], :], rtol=0, atol=3)):
                    temp = temp + 1
            if(temp > maxdist):
                maxdist = temp
                maxpoint = points

        poisoned_X_train[1] = train_dataset.x[op_indx[maxpoint], :]

    print('weights shape is ', weights.shape)
    poisoned_X_train = projection_fn(
        poisoned_X_train,
        poisoned_labels,
        theta=weights[:-1],
        bias=weights[-1])

    return poisoned_X_train


def iterative_attack(
        model,
        general_train_idx,
        sensitive_file,
        attack_method,
        advantaged,
        indices_to_poison,
        test_idx,
        test_description=None,
        step_size=0.01,
        num_iter=10,
        loss_type='normal_loss',
        projection_fn=None,
        output_root=None,
        num_copies=None,
        stop_after=3,
        start_time=None,
        display_iter_time=False,
        stopping_method='Accuracy'):
    """
    Performs the main specified adversarial attack

    :param model: Model to attack
    :general_train_idx: Index of last element in the training data
    :sensitive_file: File specifiying the sensitive group labels (dataset_group_label)
    :attack_method: Used attack method
    :advantaged: Index of advantaged group (1 or -1)
    :indices_to_poison: Indeces to be poisoned
    :test_idx: Depricated
    :test_description: Depricated
    :step_size: Step size for attacks with adversarial loss
    :num_iter: Maximum number of attack iterations
    :loss_type: Loss type for different attacks. (adversarial_loss when one is used else normal_loss)
    :projection_fn: Projection function to project updated poisoned points to feasible set
    :output_root: Output root directory
    :num_copies: Number of copies to make for poisoned points
    :stop_after: Patience for stopping training
    :start_time: Start time of training
    :display_iter_time: Print time of every iteration if true
    :stopping_method: Method to evaluate best model

    """
    if num_copies is not None:
        assert len(num_copies) == 2
        assert np.min(num_copies) >= 1
        assert len(indices_to_poison) == 2
        assert indices_to_poison[1] == (indices_to_poison[0] + 1)
        assert indices_to_poison[1] + num_copies[0] + \
            num_copies[1] == (model.train_dataset.x.shape[0] - 1)
        assert model.train_dataset.labels[indices_to_poison[0]] == 1
        assert model.train_dataset.labels[indices_to_poison[1]] == -1
        copy_start = indices_to_poison[1] + 1
        assert np.all(
            model.train_dataset.labels[copy_start:copy_start + num_copies[0]] == 1)
        assert np.all(model.train_dataset.labels[copy_start + num_copies[0]:copy_start + num_copies[0] + num_copies[1]] == -1)

    largest_test_loss = 0
    largest_parity = 0
    stop_counter = 0

    print('Test idx: %s' % test_idx)

    if start_time is not None:
        assert num_copies is not None
        times_taken = np.zeros(num_iter)
        Xs_poison = np.zeros(
            (num_iter, len(indices_to_poison), model.train_dataset.x.shape[1]))
        Ys_poison = np.zeros((num_iter, len(indices_to_poison)))
        nums_copies = np.zeros((num_iter, len(indices_to_poison)))

    for attack_iter in range(num_iter):
        since = time.time()
        print(num_iter)
        print('*** Iter: %s' % attack_iter)
        model.attack_iter = attack_iter
        # Create modified training dataset
        old_poisoned_X_train = np.copy(
            model.train_dataset.x[indices_to_poison, :])

        poisoned_X_train_subset = poison_with_influence_proj_gradient_step(
            model,
            general_train_idx,
            sensitive_file,
            attack_method,
            advantaged,
            test_idx,
            indices_to_poison,
            projection_fn,
            step_size=step_size,
            loss_type=loss_type,
            force_refresh=True,
            test_description=test_description,
            output_root=output_root)

        if num_copies is not None:
            poisoned_X_train = model.train_dataset.x
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset
            copy_start = indices_to_poison[1] + 1
            poisoned_X_train[copy_start:copy_start +
                             num_copies[0], :] = poisoned_X_train_subset[0, :]
            poisoned_X_train[copy_start + num_copies[0]:copy_start +
                             num_copies[0] + num_copies[1], :] = poisoned_X_train_subset[1, :]
        else:
            poisoned_X_train = np.copy(model.train_dataset.x)
            poisoned_X_train[indices_to_poison, :] = poisoned_X_train_subset

        # Measure some metrics on what the gradient step did
        labels = model.train_dataset.labels
        dists_sum = 0.0
        poisoned_dists_sum = 0.0
        poisoned_mask = np.array([False] * len(labels), dtype=bool)
        poisoned_mask[indices_to_poison] = True
        if(attack_method != "RAA" and attack_method != "NRAA"):
            for y in set(labels):
                cluster_center = np.mean(
                    poisoned_X_train[labels == y, :], axis=0)
                dists = np.linalg.norm(
                    poisoned_X_train[labels == y, :] - cluster_center, axis=1)
                dists_sum += np.sum(dists)

                poisoned_dists = np.linalg.norm(
                    poisoned_X_train[(labels == y) & (poisoned_mask), :] - cluster_center, axis=1)
                poisoned_dists_sum += np.sum(poisoned_dists)

            dists_mean = dists_sum / len(labels)
            poisoned_dists_mean = poisoned_dists_sum / len(indices_to_poison)

            dists_moved = np.linalg.norm(
                old_poisoned_X_train - poisoned_X_train[indices_to_poison, :], axis=1)
            print('Average distance to cluster center (overall): %s' % dists_mean)
            print('Average distance to cluster center (poisoned): %s' %
                  poisoned_dists_mean)
            print('Average diff in X_train among poisoned indices = %s' %
                  np.mean(dists_moved))
            print('Fraction of 0 gradient points: %s' %
                  np.mean(dists_moved == 0))
            print('Average distance moved by points that moved: %s' %
                  np.mean(dists_moved[dists_moved > 0]))

        # Update training dataset
        model.update_train_x(poisoned_X_train)

        # Retrain model
        results = model.train()

        if start_time is not None:
            end_time = time.time()
            times_taken[attack_iter] = end_time - start_time
            Xs_poison[attack_iter, :, :] = np.copy(poisoned_X_train_subset)
            Ys_poison[attack_iter,
                      :] = model.train_dataset.labels[indices_to_poison]
            nums_copies[attack_iter, :] = num_copies

        print('attack_iter', attack_iter)
        print('num_iter - 1', num_iter - 1)

        if ((attack_iter + 1) % 10 == 0) or (attack_iter == num_iter - 1):
            print('in')
            if(stopping_method == 'Accuracy'):
                # Calculate test loss
                test_loss = results['test_loss']
                if largest_test_loss < test_loss:
                    print('test loss match')
                    largest_test_loss = test_loss
                    np.savez(os.path.join(output_root, '%s_attack' % (model.model_name)),
                             poisoned_X_train=poisoned_X_train,
                             Y_train=model.train_dataset.labels,
                             attack_iter=attack_iter + 1)

                    stop_counter = 0
                else:
                    stop_counter += 1
                if start_time is not None:
                    np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
                             times_taken=times_taken,
                             nums_copies=nums_copies)

            if(stopping_method == 'Fairness'):
                # Calculate test loss
                E0, Parity = results['E0'], results['Parity']
                if largest_parity < E0 + Parity:
                    print('parity match')
                    largest_parity = E0 + Parity
                    np.savez(os.path.join(output_root, '%s_attack' % (model.model_name)),
                             poisoned_X_train=poisoned_X_train,
                             Y_train=model.train_dataset.labels,
                             attack_iter=attack_iter + 1)

                    stop_counter = 0
                else:
                    stop_counter += 1
                if start_time is not None:
                    np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
                             times_taken=times_taken,
                             nums_copies=nums_copies)

            # Printing time for every iter, if display_iter_time is set to True
            now = time.time()
            if (display_iter_time == True):
                total_time = now - since
                print('TOTAL ELAPSED TIME FOR ONE ITERATION \n', total_time)

            if stop_counter >= stop_after:
                print('STOPPING METHOD USED IS: ',
                      stopping_method, ' STOPPING NOW')
                break
    if start_time is not None:
        np.savez(os.path.join(output_root, '%s_timing' % (model.model_name)),
                 times_taken=times_taken,
                 Xs_poison=Xs_poison,
                 Ys_poison=Ys_poison,
                 nums_copies=nums_copies)


def get_feasible_flipped_mask(
        X_train, Y_train,
        centroids,
        centroid_vec,
        sphere_radii,
        slab_radii,
        class_map,
        use_slab=False):

    sphere_dists_flip = defenses.compute_dists_under_Q(
        X_train, -Y_train,
        Q=None,
        subtract_from_l2=False,
        centroids=centroids,
        class_map=class_map,
        norm=2)

    if use_slab:
        slab_dists_flip = defenses.compute_dists_under_Q(
            X_train, -Y_train,
            Q=centroid_vec,
            subtract_from_l2=False,
            centroids=centroids,
            class_map=class_map,
            norm=2)

    feasible_flipped_mask = np.zeros(X_train.shape[0], dtype=bool)

    for y in set(Y_train):
        class_idx_flip = class_map[-y]
        sphere_radius_flip = sphere_radii[class_idx_flip]

        feasible_flipped_mask[Y_train == y] = (
            sphere_dists_flip[Y_train == y] <= sphere_radius_flip)

        if use_slab:
            slab_radius_flip = slab_radii[class_idx_flip]
            feasible_flipped_mask[Y_train == y] = (
                feasible_flipped_mask[Y_train == y] &
                (slab_dists_flip[Y_train == y] <= slab_radius_flip))

    return feasible_flipped_mask


def init_gradient_attack_from_mask(
        X_train, Y_train,
        epsilon,
        feasible_flipped_mask,
        general_train_idx,
        sensitive_file,
        attack_method,
        use_copy=True):
    """
    Calculates the advantaged group and computes initial poisoned data points and adds them to the training data.

    :param X_train: training set features
    :param Y_train: training set labels
    :param epsilon: controlling parameter specifiying number of poisoned points to be copied such that n_poisoned = eps len(X_train)
    :param feasible_flipped_mask: Mask of feasible set
    :param general_train_idx: Index of last element in X_train
    :param sensitive_file: File specifying labels of the sensitive feature
    :param attack_method: Method of attack
    :param use_copy: Make copies of poisoned points if true, otherwise only one point per label gets sampled

    :return:
            - X_modified: X_train with added poisoned points
            - Y_modified: Y_train with added poisoned points
            - indices_to_poison: Indices of poisonoed datapoints
            - copy_array: Array specifiying number of copies of poisoned datapoints [num_pos_copies, num_neg_copies]
            - advantaged: Label of advantaged group
            - test_gender_labels: Sensitive feature labels (1, -1) of test_set (needed for Solans)
    """

    DATA_FOLDER = './data'
    dataset_path = os.path.join(DATA_FOLDER)
    f = np.load(os.path.join(dataset_path, sensitive_file))
    group_label = f['group_label']

    advantaged = 1

    male_train_index = np.where(group_label[0:general_train_idx] == 0)[
        0].astype(np.int32)
    female_train_index = np.where(group_label[0:general_train_idx] == 1)[
        0].astype(np.int32)
    male_test_index = np.where(group_label[general_train_idx:] == 0)[
        0].astype(np.int32)
    female_test_index = np.where(group_label[general_train_idx:] == 1)[
        0].astype(np.int32)

    index_male_true_train = np.where(np.logical_and(
        group_label[0:general_train_idx] == 0, Y_train == 1))[0].astype(np.int32)
    index_female_true_train = np.where(np.logical_and(
        group_label[0:general_train_idx] == 1, Y_train == 1))[0].astype(np.int32)

    train_data_one_female_prob = group_label[0:general_train_idx][
        index_female_true_train].shape[0] / female_train_index.shape[0]
    train_data_one_male_prob = group_label[0:general_train_idx][
        index_male_true_train].shape[0] / male_train_index.shape[0]

    gender_labels = np.zeros(general_train_idx)
    for k in range(general_train_idx):
        if(k in male_train_index):
            gender_labels[k] = 1
        elif(k in female_train_index):
            gender_labels[k] = -1

    test_size = len(male_test_index) + len(female_test_index)
    test_gender_labels = np.zeros(test_size)

    for k in range(test_size):
        if(k in male_test_index):
            test_gender_labels[k] = 1
        elif(k in female_test_index):
            test_gender_labels[k] = -1

    if not use_copy:
        num_copies = int(np.round(epsilon * X_train.shape[0]))

        idx_to_copy = np.random.choice(
            np.where(feasible_flipped_mask)[0],
            size=num_copies,
            replace=True)

        X_modified = data.vstack(X_train, X_train[idx_to_copy, :])
        Y_modified = np.append(Y_train, -Y_train[idx_to_copy])
        copy_array = None
        indices_to_poison = np.arange(X_train.shape[0], X_modified.shape[0])

    else:
        num_copies = int(np.round(epsilon * X_train.shape[0]))
        # Choose this in inverse class balance
        num_pos_copies = int(np.round(np.mean(Y_train == -1) * num_copies))
        num_neg_copies = num_copies - num_pos_copies

        np.random.seed(0)

        if(train_data_one_female_prob > train_data_one_male_prob):
            advantaged = -1
            pos_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == 1) & (gender_labels == -1))[0])
            neg_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == -1) & (gender_labels == 1))[0])
        else:
            advantaged = 1
            pos_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == 1) & (gender_labels == 1))[0])
            neg_idx_to_copy = np.random.choice(
                np.where(feasible_flipped_mask & (Y_train == -1) & (gender_labels == -1))[0])

        if(neg_idx_to_copy in female_train_index):
            print("female")
        else:
            print("male")
        if(pos_idx_to_copy in female_train_index):
            print("female")
        else:
            print("male")
        print(neg_idx_to_copy)
        print(pos_idx_to_copy)
        # exit()
        num_pos_copies -= 1
        num_neg_copies -= 1

        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_train,
            Y_train,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=1)
        X_modified, Y_modified = data.add_points(
            X_train[pos_idx_to_copy, :],
            1,
            X_modified,
            Y_modified,
            num_copies=num_pos_copies)
        X_modified, Y_modified = data.add_points(
            X_train[neg_idx_to_copy, :],
            -1,
            X_modified,
            Y_modified,
            num_copies=num_neg_copies)
        copy_array = [num_pos_copies, num_neg_copies]
        indices_to_poison = np.arange(X_train.shape[0], X_train.shape[0] + 2)

    return X_modified, Y_modified, indices_to_poison, copy_array, advantaged, test_gender_labels
