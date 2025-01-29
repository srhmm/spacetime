import math
import warnings

from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from scipy.linalg import cholesky
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

from scipy.linalg import cho_solve, cholesky, solve_triangular

GPR_CHOLESKY_LOWER = True

"""Gaussian processes regression with regularization and MDL description length"""


# Inherits from GaussianProcessRegressor (sklearn)
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

class GaussianProcessRegularized(GaussianProcessRegressor):

    def predict(self, X, y_test, return_std=False, return_cov=False, return_mdl=True):
        """Predict using the Gaussian process regression model.

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.

        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "At most one of return_std or return_cov can be requested."
            )

        if self.kernel is None or self.kernel.requires_vector_input:
            dtype, ensure_2d = "numeric", True
        else:
            dtype, ensure_2d = None, False

        X = self._validate_data(X, ensure_2d=ensure_2d, dtype=dtype, reset=False)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = C(1.0, constant_value_bounds="fixed") * RBF(
                    1.0, length_scale_bounds="fixed"
                )
            else:
                kernel = self.kernel

            n_targets = self.n_targets if self.n_targets is not None else 1
            y_mean = np.zeros(shape=(X.shape[0], n_targets)).squeeze()

            if return_cov:
                y_cov = kernel(X)
                if n_targets > 1:
                    y_cov = np.repeat(
                        np.expand_dims(y_cov, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                if n_targets > 1:
                    y_var = np.repeat(
                        np.expand_dims(y_var, -1), repeats=n_targets, axis=-1
                    )
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            # Alg 2.1, page 19, line 4 -> f*_bar = K(X_test, X_train) . alpha
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans @ self.alpha_

            # undo normalisation
            y_mean = self._y_train_std * y_mean + self._y_train_mean

            # if y_mean has shape (n_samples, 1), reshape to (n_samples,)
            if y_mean.ndim > 1 and y_mean.shape[1] == 1:
                y_mean = np.squeeze(y_mean, axis=1)

            # Alg 2.1, page 19, line 5 -> v = L \ K(X_test, X_train)^T
            V = solve_triangular(
                self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
            )

            if return_cov:
                # Alg 2.1, page 19, line 6 -> K(X_test, X_test) - v^T. v
                y_cov = self.kernel_(X) - V.T @ V

                # undo normalisation
                y_cov = np.outer(y_cov, self._y_train_std ** 2).reshape(
                    *y_cov.shape, -1
                )
                # if y_cov has shape (n_samples, n_samples, 1), reshape to
                # (n_samples, n_samples)
                if y_cov.shape[2] == 1:
                    y_cov = np.squeeze(y_cov, axis=2)

                return y_mean, y_cov
            elif return_std:
                # Compute variance of predictive distribution
                # Use einsum to avoid explicitly forming the large matrix
                # V^T @ V just to extract its diagonal afterward.
                y_var = self.kernel_.diag(X).copy()
                y_var -= np.einsum("ij,ji->i", V.T, V)

                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn(
                        "Predicted variances smaller than 0. "
                        "Setting those variances to 0."
                    )
                    y_var[y_var_negative] = 0.0

                # undo normalisation
                y_var = np.outer(y_var, self._y_train_std ** 2).reshape(
                    *y_var.shape, -1
                )

                # if y_var has shape (n_samples, 1), reshape to (n_samples,)
                if y_var.shape[1] == 1:
                    y_var = np.squeeze(y_var, axis=1)

                return y_mean, np.sqrt(y_var)
            elif return_mdl:
                # if y_test.ndim == 1:
                #    y_test = y_test[:, np.newaxis]

                # log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_test,  # [:self.alpha_.shape[0]],
                #                                       self.alpha_.reshape(1, -1))
                # log_likelihood_dims -= np.log(np.diag(self.L_)).sum()
                # log_likelihood_dims -= self.K.shape[0] / 2 * np.log(2 * np.pi)

                # Gaussian (negative) likelihood
                sigma = 1
                log_lik = ll = np.sum(math.log(2 * math.pi * (sigma ** 2)) / 2 + ((y_test - y_mean) ** 2) / (
                            2 * (sigma ** 2)))  # -log_likelihood_dims.sum(axis=-1)

                X_penalty = 1 / 2 * np.log(
                    np.linalg.det(np.identity(self.K.shape[0]) + 1 ** 2 * self.K))
                if X_penalty == np.inf:
                    X_penalty = 0
                mdl_score = np.abs(log_lik) + self.mdl_model_train + X_penalty
                return mdl_score, log_lik, self.mdl_model_train, X_penalty

            else:
                return y_mean

    def _mdl_score(self, y_test):

        """ Gaussian Process description length.

        :param y_test: test data.
        """
        # MDL data score/log likelihood
        if y_test.ndim == 1:
            y_test = y_test[:, np.newaxis]

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_test,  # [:self.alpha_.shape[0]],
                                               self.alpha_.reshape(1, -1))
        log_likelihood_dims -= np.log(np.diag(self.L_)).sum()
        log_likelihood_dims -= self.K.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = -log_likelihood_dims.sum(axis=-1)

        X_penalty = 1 / 2 * np.log(
            np.linalg.det(np.identity(self.K.shape[0]) + 1 ** 2 * self.K))
        if X_penalty == np.inf:
            X_penalty = 0
        mdl_score = np.abs(log_lik) + self.mdl_model_train + X_penalty

        return mdl_score, np.abs(log_lik), self.mdl_model_train, X_penalty

    def fit(self, X, y):
        """ Gaussian Process model X->y and description length thereof.

        :param X: predictors.
        :param y: target.
        """
        super(GaussianProcessRegularized, self).fit(X, y)

        # Kernel for X
        kernel = self.kernel_
        K = kernel(self.X_train_, eval_gradient=False)
        self.K = K

        sigma = 1

        # Model Complexity parameter alpha and MDL model score
        K[np.diag_indices_from(K)] += (1e-1)
        mat = np.eye(X.shape[0]) + K * 1 ** -2
        alpha = np.linalg.solve(mat, y)
        self.alpha_ = alpha
        mdl_model_train = alpha.T @ mat @ alpha

        # MDL data score/log likelihood
        mdl_lik_train = -self.log_marginal_likelihood_value_
        # precompute for self.mdl_score():
        L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        self.L_ = L

        # MDL remaining term, only depends on X
        mdl_pen_train = 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + sigma ** 2 * K))
        if mdl_pen_train == np.inf:
            mdl_pen_train = 0

        mdl_train = mdl_lik_train + mdl_model_train + mdl_pen_train

        self.mdl_lik_train = mdl_lik_train
        self.mdl_model_train = mdl_model_train
        self.mdl_pen_train = mdl_pen_train
        self.mdl_train = mdl_train

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        mdl, log_lik, m_penalty, X_penalty = self.predict(X_test, y_test, return_mdl=True)  # self._mdl_score(y_test)
        return mdl, log_lik, m_penalty, X_penalty
