from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import abc
import sys

import IPython

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster

import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse
from scipy.optimize import fmin_ncg

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras import backend as K

from .hessians import hessian_vector_product
from .dataset import DataSet

def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.compat.v1.get_variable(
        name, 
        shape, 
        initializer=initializer, 
        dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name, 
        shape, 
        initializer=tf.compat.v1.truncated_normal_initializer(
            stddev=stddev, 
            dtype=dtype))
 
    if wd is not None:
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      tf.compat.v1.add_to_collection('losses', weight_decay)

    return var
    

class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.compat.v1.set_random_seed(0)
        
        self.batch_size = kwargs.pop('batch_size')
        self.train_dataset = kwargs.pop('train_dataset')
        self.validation_dataset = kwargs.pop('validation_dataset')
        self.test_dataset = kwargs.pop('test_dataset')
        self.train_dir = kwargs.pop('train_dir', 'output')
        log_dir = kwargs.pop('log_dir', 'log')
        self.model_name = kwargs.pop('model_name')
        self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')        
        self.decay_epochs = kwargs.pop('decay_epochs')
        self.attack_method = kwargs.pop('method')
        self.general_train_idx=kwargs.pop('general_train_idx')
        self.sensitive_file=kwargs.pop('sensitive_file')
        
        if 'keep_probs' in kwargs: self.keep_probs = kwargs.pop('keep_probs')
        else: self.keep_probs = None
        
        if 'mini_batch' in kwargs: self.mini_batch = kwargs.pop('mini_batch')        
        else: self.mini_batch = True
        
        if 'damping' in kwargs: self.damping = kwargs.pop('damping')
        else: self.damping = 0.0
        
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.compat.v1.ConfigProto()        
        self.sess = tf.compat.v1.Session(config=config)
        K.set_session(self.sess)
                
        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.train_dataset.labels.shape[0]
        self.num_test_examples = self.test_dataset.labels.shape[0]
    
        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(2))
            self.logits = self.inference(self.input_placeholder, self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):            
            self.logits = self.inference(self.input_placeholder, self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)
        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
        self.logits, 
        self.labels_placeholder,self.input_placeholder)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32)
        self.update_learning_rate_op = tf.compat.v1.assign(self.learning_rate, self.learning_rate_placeholder)
        
        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)        
        self.preds = self.predictions(self.logits)

        # Setup misc
        self.saver = tf.compat.v1.train.Saver()

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(ys=self.total_loss, xs=self.params)
        self.grad_loss_no_reg_op = tf.gradients(ys=self.loss_no_reg, xs=self.params)
        self.v_placeholder = [tf.compat.v1.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.u_placeholder = [tf.compat.v1.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(ys=self.total_loss, xs=self.input_placeholder)        

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)        
        self.influence_op = tf.add_n(
            [tf.reduce_sum(input_tensor=tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(ys=self.influence_op, xs=self.input_placeholder)
    
        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.train_dataset)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.test_dataset)

        init = tf.compat.v1.global_variables_initializer()        
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()

        if(self.attack_method == "IAF"):
            self.adversarial_loss, self.indiv_adversarial_loss = self.hard_adv_loss(self.logits, self.labels_placeholder,self.input_placeholder)
        else:
            self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder,self.input_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(ys=self.adversarial_loss, xs=self.params)

    def get_fairness_measures(self,art_poisoned_predicts_test,art_poisoned_predicts_train):
        DATA_FOLDER = './data'
        dataset_path = os.path.join(DATA_FOLDER)
        f = np.load(os.path.join(dataset_path, self.sensitive_file))
        group_label = f['group_label']
        X_test = self.test_dataset.x
        Y_test = self.test_dataset.labels
        X_train  = self.train_dataset.x
        Y_train = self.train_dataset.labels

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]


        index_male_test = np.where(group_label[self.general_train_idx:] == 0)[0].astype(np.int32)
        index_female_test = np.where(group_label[self.general_train_idx:] == 1)[0].astype(np.int32)
        index_male_true_test = np.where(np.logical_and(group_label[self.general_train_idx:] == 0, Y_test==1))[0].astype(np.int32)
        index_male_false_test = np.where(np.logical_and(group_label[self.general_train_idx:] == 0, Y_test==-1))[0].astype(np.int32)
        index_female_true_test = np.where(np.logical_and(group_label[self.general_train_idx:] == 1, Y_test==1))[0].astype(np.int32)
        index_female_false_test = np.where(np.logical_and(group_label[self.general_train_idx:] == 1, Y_test==-1))[0].astype(np.int32)


        poi_test_hat_one = np.where(art_poisoned_predicts_test[:,0] == 1)[0].astype(np.int32)
        poi_test_hat_zero = np.where(art_poisoned_predicts_test[:,0] == -1)[0].astype(np.int32)

        poi_test_y_one_hat_one =  (np.where(np.logical_and(art_poisoned_predicts_test[:,0] == 1, Y_test==1))[0].astype(np.int32).shape[0]) / Y_test.shape[0]
        poi_test_y_one_hat_zero = (np.where(np.logical_and(art_poisoned_predicts_test[:,0] == -1, Y_test==1))[0].astype(np.int32).shape[0])/Y_test.shape[0]
        poi_test_y_zero_hat_one = (np.where(np.logical_and(art_poisoned_predicts_test[:,0] == 1, Y_test==-1))[0].astype(np.int32).shape[0])/Y_test.shape[0]
        poi_test_y_zero_hat_zero = (np.where(np.logical_and(art_poisoned_predicts_test[:,0] == -1, Y_test==-1))[0].astype(np.int32).shape[0])/Y_test.shape[0]
 
        test_female_one_prediction = np.where(art_poisoned_predicts_test[index_female_test][:,0] == 1)[0].astype(np.int32)
        test_female_zero_prediction = np.where(art_poisoned_predicts_test[index_female_test][:,0] == -1)[0].astype(np.int32)
        test_male_one_prediction = np.where(art_poisoned_predicts_test[index_male_test][:,0] == 1)[0].astype(np.int32)
        test_male_zero_prediction = np.where(art_poisoned_predicts_test[index_male_test][:,0] == -1)[0].astype(np.int32)
        print("******************Poison model parity bias on Test" + str (   abs( (test_female_one_prediction.shape[0]/index_female_test.shape[0]) - (test_male_one_prediction.shape[0]/index_male_test.shape[0])  )     ))

        a_female_test = (test_female_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_one_hat_one
        a_male_test = (test_male_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_one_hat_one

        b_female_test = (test_female_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_one_hat_zero
        b_male_test = (test_male_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_one_hat_zero

        c_female_test = (test_female_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_zero_hat_one
        c_male_test = (test_male_one_prediction.shape[0]/poi_test_hat_one.shape[0])*poi_test_y_zero_hat_one

        d_female_test = (test_female_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_zero_hat_zero
        d_male_test = (test_male_zero_prediction.shape[0]/poi_test_hat_zero.shape[0])*poi_test_y_zero_hat_zero


        print("******************Poison model EO bias on Test" + str ( abs( (a_female_test/(a_female_test+b_female_test)) - (a_male_test/(a_male_test+b_male_test)) )) )
    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))        
        print('Total number of parameters: %s' % self.num_params)


        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos : cur_pos+len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list


    def reset_datasets(self):
        if self.train_dataset is not None:
            self.train_dataset.reset_batch()
        if self.validation_dataset is not None:
            self.validation_dataset.reset_batch()
        if self.test_dataset is not None:
            self.test_dataset.reset_batch()


    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict


    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size
    
        input_feed, labels_feed = data_set.next_batch(batch_size)                              
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,            
        }
        return feed_dict


    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict


    def minibatch_mean_eval(self, ops, data_set):

        if self.mini_batch:
            num_examples = data_set.num_examples
            assert num_examples % self.batch_size == 0
            num_iter = int(num_examples / self.batch_size)

            self.reset_datasets()

            ret = []
            for i in xrange(num_iter):
                feed_dict = self.fill_feed_dict_with_batch(data_set)
                ret_temp = self.sess.run(ops, feed_dict=feed_dict)
                
                if len(ret)==0:
                    for b in ret_temp:
                        if isinstance(b, list):
                            ret.append([c / float(num_iter) for c in b])
                        else:
                            ret.append([b / float(num_iter)])
                else:
                    for counter, b in enumerate(ret_temp):
                        if isinstance(b, list):
                            ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                        else:
                            ret[counter] += (b / float(num_iter))
                
            return ret

        else:
            feed_dict = self.fill_feed_dict_with_all_ex(data_set)
            return self.sess.run(
                ops, 
                feed_dict=feed_dict)

    def print_model_eval(self):
        params_val = self.sess.run(self.params)

        if self.mini_batch == True:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val,train_predictions = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op,self.preds],
                self.train_dataset)
            
            test_loss_val, test_acc_val, test_predictions = self.minibatch_mean_eval(
                [self.loss_no_reg, self.accuracy_op, self.preds],
                self.test_dataset)

        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val,train_predictions = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op,self.preds], 
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val, test_predictions = self.sess.run(
                [self.loss_no_reg, self.accuracy_op, self.preds], 
                feed_dict=self.all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

        print('Test loss (w/o reg) on all data: %s' % test_loss_val)
        print('Train acc on all data:  %s' % train_acc_val)
        print('Test acc on all data:   %s' % test_acc_val)

        self.get_fairness_measures(test_predictions,train_predictions)

        grad_norm = np.linalg.norm(np.concatenate(grad_loss_val))
        params_norm = np.linalg.norm(np.concatenate(params_val))
        print('Norm of the mean of gradients: %s' % grad_norm)
        print('Norm of the params: %s' % params_norm)

        results = {
            'loss': loss_val,
            'loss_no_reg': loss_no_reg_val,
            'test_loss': test_loss_val,
            'train_acc': train_acc_val,
            'test_acc': test_acc_val,
            'grad_norm': grad_norm,
            'params_norm': params_norm
        }

        return results

    # Not used but might be useful later
    # def load_checkpoint(self, iter_to_load, do_checks=True):
    #     checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load) 
    #     self.saver.restore(self.sess, checkpoint_to_load)

    #     if do_checks:
    #         print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
    #         self.print_model_eval()


    def get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op
        """
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_train_sgd_op(self, total_loss, global_step, learning_rate=0.001):
        """
        Return train_sgd_op
        """
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op


    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """     
        correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
        return tf.reduce_sum(input_tensor=tf.cast(correct, tf.int32)) / tf.shape(input=labels)[0]


    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block        
        return feed_dict


    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    
    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch == True:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in xrange(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.train_dataset, batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
            
        hessian_vector_val = [a + self.damping * b for (a,b) in zip(hessian_vector_val, v)]

        return hessian_vector_val


    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
        return get_fmin_loss


    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))
            
            return np.concatenate(hessian_vector_val) - np.concatenate(v)
        return get_fmin_grad


    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)


    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        
        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.train_dataset, idx_to_remove)      
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v), np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback


    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=1e-8,
            maxiter=100,
            disp=0) 

        return self.vec_to_list(fmin_results)


    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        # else:
        #     raise ValueError, 'Loss must be specified'

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i+1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.test_dataset, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end-start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end-start) for (a, b) in zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a/len(test_indices) for a in test_grad_loss_no_reg_val]

        else:            
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.test_dataset)[0]
            
        return test_grad_loss_no_reg_val

    def get_grad_of_influence_wrt_input(self, train_indices, test_indices, 
        approx_type='cg', approx_params=None, force_refresh=True, verbose=True, test_description=None,
        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive, 
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)            

        if verbose: print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
        
        start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (self.model_name, approx_type, loss_type, test_description))
        
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose: print('Loaded inverse HVP from %s' % approx_filename)
        else:            
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose: print('Saved inverse HVP to %s' % approx_filename)            
        
        duration = time.time() - start_time
        if verbose: print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.train_dataset,  
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]            
            
            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val


    def update_train_x(self, new_train_x):
        assert np.all(new_train_x.shape == self.train_dataset.x.shape)
        new_train = DataSet(new_train_x, np.copy(self.train_dataset.labels))
        self.train_dataset = new_train
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.train_dataset)                
        self.reset_datasets()


    def update_train_x_y(self, new_train_x, new_train_y):
        new_train = DataSet(new_train_x, new_train_y)
        self.train_dataset = new_train
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.train_dataset)                
        self.num_train_examples = len(new_train_y)
        self.reset_datasets()        
