from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import time
import numpy as np
import scipy.sparse as sparse
import cvxpy as cvx

### Minimizer / CVX

def cvx_dot(a,b):
    return cvx.sum(cvx.multiply(a, b))

class Minimizer(object):

    def __init__(
        self,
        d,
        use_sphere=True,
        use_slab=True,
        non_negative=False,
        less_than_one=False,
        constrain_max_loss=False,
        goal=None,
        X=None):

        assert goal in ['find_nearest_point', 'maximize_test_loss']

        self.use_slab = use_slab
        self.constrain_max_loss = constrain_max_loss
        self.goal = goal

        self.use_projection = not (non_negative or less_than_one or X)
        if self.use_projection:
            eff_d = 3 + (constrain_max_loss == True)
        else:
            eff_d = d


        self.cvx_x = cvx.Variable(eff_d)
        self.cvx_w = cvx.Parameter(eff_d)
        self.cvx_centroid = cvx.Parameter(eff_d)
        self.cvx_sphere_radius = cvx.Parameter(1)

        self.cvx_x_c = self.cvx_x - self.cvx_centroid

        if goal == 'find_nearest_point':
            self.objective = cvx.Minimize(cvx.pnorm(self.cvx_x - self.cvx_w, 2) ** 2)
        elif goal == 'maximize_test_loss':
            self.cvx_y = cvx.Parameter(1)
            # No need for bias term
            self.objective = cvx.Maximize(1 - self.cvx_y * cvx_dot(self.cvx_w, self.cvx_x))

        self.constraints = []
        if use_sphere:
            if X is not None:
                if sparse.issparse(X):
                    X_max = X.max(axis=0).toarray().reshape(-1)
                else:
                    X_max = np.max(X, axis=0).reshape(-1)
                X_max[X_max < 1] = 1
                X_max[X_max > 50] = 50
                self.constraints.append(self.cvx_x <= X_max)
                kmax = int(np.ceil(np.max(X_max)))

                self.cvx_env = cvx.Variable(eff_d)
                for k in range(1, kmax+1):
                    active = k <= (X_max)
                    if(k<11):
                        self.constraints.append(self.cvx_env[active] >= self.cvx_x[active] * (2*k-1) - k*(k-1))
                self.constraints.append(
                    (
                        cvx.sum(self.cvx_env)
                        - 2 * cvx_dot(self.cvx_centroid, self.cvx_x)
                        + cvx.sum_squares(self.cvx_centroid)
                    )
                    <=(self.cvx_sphere_radius ** 2)
                )
            else:
                self.constraints.append(cvx.pnorm(self.cvx_x_c, 2) ** 2 <= self.cvx_sphere_radius ** 2)

        if use_slab:
            self.cvx_centroid_vec = cvx.Parameter(eff_d)
            self.cvx_slab_radius = cvx.Parameter(1)
            self.constraints.append(cvx_dot(self.cvx_centroid_vec, self.cvx_x_c) <= self.cvx_slab_radius)
            self.constraints.append(-cvx_dot(self.cvx_centroid_vec, self.cvx_x_c) <= self.cvx_slab_radius)

        if non_negative:
            self.constraints.append(self.cvx_x >= 0)

        if less_than_one:
            self.constraints.append(self.cvx_x <= 1)

        if constrain_max_loss:
            self.cvx_max_loss = cvx.Parameter(1)
            self.cvx_constraint_w = cvx.Parameter(eff_d)
            self.cvx_constraint_b = cvx.Parameter(1)
            self.constraints.append(
                1 - self.cvx_y * (
                    cvx_dot(self.cvx_constraint_w, self.cvx_x) + self.cvx_constraint_b
                ) <= 1e-3 + self.cvx_max_loss)


        self.prob = cvx.Problem(self.objective, self.constraints)

    def minimize_over_feasible_set(self, y, w,
                                   centroid, centroid_vec, sphere_radius, slab_radius,
                                   max_loss=None,
                                   constraint_w=None,
                                   constraint_b=None,
                                   verbose=False):
        """
        Includes both sphere and slab.
        Returns optimal x.
        """
        start_time = time.time()
        if self.use_projection:
            A = np.concatenate(
                (
                    w.reshape(-1, 1),
                    centroid.reshape(-1, 1),
                    centroid_vec.reshape(-1, 1)
                ),
                axis=1)

            if constraint_w is not None:
                A = np.concatenate(
                    (A, constraint_w.reshape(-1, 1)),
                    axis=1)

            P = get_projection_matrix(A)

            self.cvx_w.value = P.dot(w.reshape(-1))
            self.cvx_centroid.value = P.dot(centroid.reshape(-1))
            self.cvx_centroid_vec.value = P.dot(centroid_vec.reshape(-1))
        else:
            self.cvx_w.value = w.reshape(-1)
            self.cvx_centroid.value = centroid.reshape(-1)
            self.cvx_centroid_vec.value = centroid_vec.reshape(-1)

        if self.goal == 'maximize_test_loss':
            self.cvx_y.value = y

        self.cvx_sphere_radius.value = np.array([sphere_radius])
        self.cvx_slab_radius.value = np.array([slab_radius])

        if self.constrain_max_loss:
            self.cvx_max_loss.value = max_loss
            self.cvx_constraint_b.value = constraint_b
            if self.use_projection:
                self.cvx_constraint_w.value = P.dot(constraint_w.reshape(-1))
            else:
                self.cvx_constraint_w.value = constraint_w.reshape(-1)

        try:
            self.prob.solve(verbose=verbose)
        except:
            raise
            print('centroid', self.cvx_centroid.value)
            print('centroid_vec', self.cvx_centroid_vec.value)
            print('w', self.cvx_w.value)
            print('sphere_radius', sphere_radius)
            print('slab_radius', slab_radius)
            if self.constrain_max_loss:
                print('constraint_w', self.cvx_constraint_w.value)
                print('constraint_b', self.cvx_constraint_b.value)

            print('Resolving verbosely')
            self.prob.solve(verbose=True)
            raise

        x_opt = np.array(self.cvx_x.value).reshape(-1)

        if self.use_projection:
            return x_opt.dot(P)
        else:
            return x_opt