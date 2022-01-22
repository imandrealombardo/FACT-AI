from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
from sklearn import svm
from scipy.optimize import fmin_ncg

import tensorflow as tf
import math

from .genericNeuralNet import GenericNeuralNet, variable_with_weight_decay

# Disable eager execution. TensorFlow 2.x released the eager execution mode, for which each node is immediately executed after definition.
tf.compat.v1.disable_eager_execution()

def log_loss(x, t):
    exponents = -(x-1)/t
    # exponents = -(x)/t
    max_elems = tf.maximum(exponents, tf.zeros_like(exponents))

    return t * (max_elems + tf.math.log(
        tf.exp(exponents - max_elems) + 
        tf.exp(tf.zeros_like(exponents) - max_elems)))
    # return t * tf.log(tf.exp(-(x)/t) + 1)        

def smooth_hinge_loss(x, t):    

    # return tf.cond(
    #     tf.equal(t, 0),
    #     lambda: hinge(x),
    #     lambda: log_loss(x,t)
    #     )
    if t == 0:
        return hinge(x)
    else:
        return log_loss(x,t)


class SmoothHinge(GenericNeuralNet):

    # Expects labels to be +1 or -1

    def __init__(self, positive_sensitive_el,negative_sensitive_el,sensitive_feature_idx,input_dim, temp, weight_decay, use_bias, **kwargs):
        self.sensitive_feature_idx = sensitive_feature_idx
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.temp = temp
        self.use_bias = use_bias
        self.positive_sensitive_el =positive_sensitive_el
        self.negative_sensitive_el =negative_sensitive_el

        super(SmoothHinge, self).__init__(**kwargs)

        C = 1.0 / (self.num_train_examples * self.weight_decay)        
        self.svm_model = svm.LinearSVC(
            C=C,
            loss='hinge',
            tol=1e-6,
            fit_intercept=self.use_bias,
            random_state=24,
            max_iter=5000)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.svm_model_minus_one = svm.LinearSVC(
            C=C_minus_one,
            loss='hinge',
            tol=1e-6,
            fit_intercept=self.use_bias,
            random_state=24,
            max_iter=5000)     

        self.set_params_op = self.set_params()

        assert self.num_classes == 2

    def get_all_params(self):
        all_params = []
        temp_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name("%s/%s:0" % ('softmax_linear', 'weights'))
        all_params.append(temp_tensor)
        return all_params
        

    def placeholder_inputs(self):
        input_placeholder = tf.compat.v1.placeholder(
            tf.float32, 
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.compat.v1.placeholder(
            tf.int32,             
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder


    def inference(self, input):        
        # Softmax_linear
        with tf.compat.v1.variable_scope('softmax_linear'):

            # We regularize the bias to keep it in line with sklearn's 
            # liblinear implementation
            weights = variable_with_weight_decay(
                'weights', 
                [self.input_dim + 1],
                stddev=5.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)            
            # biases = variable(
            #     'biases',
            #     [1],
            #     tf.constant_initializer(0.0))

            
            logits = tf.matmul(
            tf.concat([input, tf.ones([tf.shape(input=input)[0], 1])], axis=1),
            tf.reshape(weights, [-1, 1]))# + biases

        self.weights = weights
        return logits


    def get_train_fmin_loss_fn(self, train_feed_dict):
        def fmin_loss(W):
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
            loss_val = self.sess.run(self.total_loss, feed_dict=train_feed_dict)        
            return loss_val
        return fmin_loss


    def get_train_fmin_grad_fn(self, train_feed_dict):        
        def fmin_grad(W):
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
            grad_val = self.sess.run(self.grad_total_loss_op, feed_dict=train_feed_dict)[0]
            return grad_val
        return fmin_grad


    def get_train_fmin_hvp_fn(self, train_feed_dict):
        def fmin_hvp(W, v):            
            params_feed_dict = {}
            params_feed_dict[self.W_placeholder] = W        
            self.sess.run(self.set_params_op, feed_dict=params_feed_dict)

            feed_dict = self.update_feed_dict_with_v_placeholder(train_feed_dict, self.vec_to_list(v))
            hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)[0]            
            return hessian_vector_val
        return fmin_hvp


    def train(self):
        if self.temp == 0:
            results = self.train_with_svm(self.all_train_feed_dict)
        else:
            results = self.train_with_fmin(self.all_train_feed_dict)
        return results
            
    def train_with_fmin(self, train_feed_dict, save_checkpoints=True, verbose=True):
        fmin_loss_fn = self.get_train_fmin_loss_fn(train_feed_dict)
        fmin_grad_fn = self.get_train_fmin_grad_fn(train_feed_dict)
        fmin_hvp_fn = self.get_train_fmin_hvp_fn(train_feed_dict)

        x0 = np.array(self.sess.run(self.params)[0])
        
        # fmin_results = fmin_l_bfgs_b(
        # # fmin_results = fmin_cg(
        #     fmin_loss_fn,
        #     x0,
        #     fmin_grad_fn
        #     # gtol=1e-8
        #     )

        print(f'****PRINT fmin_loss_fn shape**** {fmin_loss_fn}')

        print(f'W_placeholder and shape: {self.W_placeholder}, {self.W_placeholder.shape}')

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=x0,
            fprime=fmin_grad_fn,
            fhess_p=fmin_hvp_fn,            
            avextol=1e-8,
            maxiter=100,disp=0)

        print(f'****PRINT fmin_results**** {fmin_results.shape}')

        W = np.reshape(fmin_results, -1)

        print(f'****PRINT W**** {W.shape}')
                
        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W        
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            # print('CG training took %s iter.' % model.n_iter_)
            print('After training with CG: ')
            results = self.print_model_eval()
        else:
            results = None

        return results


    def set_params(self):
        if self.use_bias:
            self.W_placeholder = tf.compat.v1.placeholder(
                tf.float32,
                shape=[self.input_dim + 1],
                name='W_placeholder')
        else:
            self.W_placeholder = tf.compat.v1.placeholder(
                tf.float32,
                shape=[self.input_dim],
                name='W_placeholder')
        set_weights = tf.compat.v1.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]
    

    def predictions(self, logits):
        preds = tf.sign(logits, name='preds')
        return preds
       
    def loss(self, logits, labels,X_train): 
        self.margin = tf.multiply(
            tf.cast(labels, tf.float32), 
            tf.reshape(logits, [-1]))        

        indiv_loss_no_reg = smooth_hinge_loss(self.margin, self.temp)
        loss_no_reg = tf.reduce_mean(input_tensor=indiv_loss_no_reg) 

        tf.compat.v1.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg
        
 
    def hard_adv_loss(self, logits, labels,X_train):
        x_sensitive = X_train[:,self.sensitive_feature_idx]
        z_i_z_bar = x_sensitive - tf.reduce_mean(input_tensor=x_sensitive)
        cov_thresh = np.abs(0.)
        sens_logits = tf.matmul(
                    X_train,
                    tf.reshape(self.weights[0:self.input_dim], [-1, 1]))
        prod = tf.reduce_mean( input_tensor=tf.multiply(tf.cast(z_i_z_bar, tf.float32), tf.reshape(sens_logits, [-1])),axis=0)
        self.margin = tf.multiply(
            tf.cast(labels, tf.float32), 
            tf.reshape(logits, [-1]))        

        indiv_adversarial_loss1 = smooth_hinge_loss(self.margin, self.temp)
        indiv_adversarial_loss = indiv_adversarial_loss1+prod
        adversarial_loss = tf.reduce_mean(input_tensor=indiv_adversarial_loss1) + tf.abs(tf.reduce_mean(input_tensor=prod))
        return adversarial_loss, indiv_adversarial_loss 

    def adversarial_loss(self, logits, labels,X_train):
        wrong_labels = (labels - 1) * -1 # Flips 0s and 1s
        wrong_margins = tf.multiply(
            tf.cast(wrong_labels, tf.float32), 
            tf.reshape(logits, [-1]))  

        indiv_adversarial_loss = -smooth_hinge_loss(wrong_margins, self.temp)
        adversarial_loss = tf.reduce_mean(input_tensor=indiv_adversarial_loss)
        return adversarial_loss, indiv_adversarial_loss 
     

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
        preds = tf.sign(tf.reshape(logits, [-1]))
        correct = tf.reduce_sum(
            input_tensor=tf.cast(
                tf.equal(
                    preds, 
                    tf.cast(labels, tf.float32)),
                tf.int32))
        return correct / tf.shape(input=labels)[0]

