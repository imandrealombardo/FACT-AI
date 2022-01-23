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


np.random.seed(1)

initial_learning_rate = 0.001


parser = argparse.ArgumentParser()
parser.add_argument('--total_grad_iter', default=300,
                    help="Maximum number of attack gradient iterations for the attack")
parser.add_argument('--use_slab', action='store_true',
                    help="Utilize slab defense --> Anomaly detector=interesection with L2 defense")
parser.add_argument('--dataset', default='german',
                    help="Specify dataset file name")
parser.add_argument('--percentile', default=95,
                    help="percentage of data to keep in feasible set")
parser.add_argument('--epsilon', default=0.03,
                    help="partial of number of datapoints as number of poisened points to create")
parser.add_argument('--lamb', default=1.,
                    help="adversarial loss lambda")
parser.add_argument('--weight_decay', default=0.09,
                    help="Specify weight decay for regularization")
parser.add_argument('--step_size', default=0.1)
parser.add_argument('--no_LP', action="store_true",
                    help="Don't use LP rounding")
parser.add_argument('--timed', action="store_true",
                    help="Activated timed")
parser.add_argument('--sensitive_feature_idx', default=8,
                    help="Sensitive group feature index in data")
parser.add_argument('--method', default="IAF",
                    help="specify attack method out of 'IAF', 'RAA', 'NRAA', 'Koh' ")
parser.add_argument('--sensitive_attr_filename', help="Specify filename of group label file",
                    default='german_group_label.npz')
parser.add_argument('--stop_after', default='2',
                    help='Specify after how many iterations without improving the attack should stop')
parser.add_argument('--batch_size', default=100,
                    help="Specify batch size")
args = parser.parse_args()

dataset_name = args.dataset
use_slab = args.use_slab
epsilon = float(args.epsilon)
step_size = float(args.step_size)
percentile = int(np.round(float(args.percentile)))
total_grad_iter = int(np.round(float(args.total_grad_iter)))
no_LP = args.no_LP
timed = args.timed
attack_method = args.method
print('ATTACK METHOD', attack_method)
sensitive_idx = int(args.sensitive_feature_idx)
sensitive_file = args.sensitive_attr_filename
lamb = float(args.lamb)
weight_decay = float(args.weight_decay)
stop_after = int(args.stop_after)
batch_size = int(args.batch_size)

output_root = os.path.join(datasets.OUTPUT_FOLDER,
                           dataset_name, 'influence_data')
datasets.safe_makedirs(output_root)

if(attack_method == "IAF"):
    loss_type = 'adversarial_loss'
elif(attack_method == "Koh"):
    loss_type = 'adversarial_loss'
    lamb = 1
else:
    loss_type = 'normal_loss'

print('epsilon: %s' % epsilon)
print('use_slab: %s' % use_slab)


temp = 0.001  # Delta for smooth hinge loss
use_copy = True  # Copy the poisened points to get the specified amount given by epsilon
use_LP = True  # Use LP rounding
num_classes = 2  # Only binary classification possible


if no_LP:
    assert dataset_name == 'german'
    use_LP = False
    percentile = 80

model_name = 'smooth_hinge_%s_sphere-True_slab-%s_start-copy_lflip-True_step-%s_t-%s_eps-%s_wd-%s_rs-1' % (
    dataset_name, use_slab,
    step_size, temp, epsilon, weight_decay)
if percentile != 95:
    model_name = model_name + '_percentile-%s' % percentile

if no_LP:
    model_name = model_name + '_no-LP'
if timed:
    model_name = model_name + '_timed'


X_train, Y_train, X_test, Y_test = datasets.load_dataset(dataset_name)

general_train_idx = X_train.shape[0]
unique_sensitives = np.sort(np.unique(X_train[:, sensitive_idx]))
positive_sensitive_el = np.float32(unique_sensitives[1])
negative_sensitive_el = np.float32(unique_sensitives[0])

if sparse.issparse(X_train):
    X_train = X_train.toarray()
if sparse.issparse(X_test):
    X_test = X_test.toarray()


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

X_modified, Y_modified, indices_to_poison, copy_array, advantaged = iterative_attack.init_gradient_attack_from_mask(
    X_train, Y_train,
    epsilon,
    feasible_flipped_mask,
    general_train_idx,
    sensitive_file,
    attack_method,
    use_copy=use_copy)

tf.compat.v1.reset_default_graph()

input_dim = X_train.shape[1]
train = DataSet(X_train, Y_train)
validation = None
test = DataSet(X_test, Y_test)

model = SmoothHinge(
    positive_sensitive_el=positive_sensitive_el,
    negative_sensitive_el=negative_sensitive_el,
    sensitive_feature_idx=sensitive_idx,
    input_dim=input_dim,
    temp=temp,
    weight_decay=weight_decay,
    use_bias=True,
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
    method=attack_method,
    general_train_idx=general_train_idx,
    sensitive_file=sensitive_file,
    lamb=lamb)


model.update_train_x_y(X_modified, Y_modified)
model.train()

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
    attack_method,
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
    start_time=start_time)
print("The end")
