import math
import warnings

import torch
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C

from scipy.linalg import cholesky, solve_triangular
from sklearn.kernel_approximation import RBFSampler

from st.scoring.models.embedding import HermiteEmbedding

GPR_CHOLESKY_LOWER = True

"""Fourier feature approximations for Gaussian processes regression with MDL description length"""


# Inherits from GaussianProcessRegressor (sklearn)
# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Modified by: Pete Green <p.l.green@liverpool.ac.uk>
# License: BSD 3 clause

class GaussianProcessFourierRegularized(GaussianProcessRegressor):

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
        super(GaussianProcessFourierRegularized, self).fit(X, y)

        if True:
            self.n_basis_functions = 100
            X = torch.from_numpy(self.X_train_)
            emb = HermiteEmbedding(gamma=0.5, m=self.n_basis_functions, d=self.X_train_.shape[1], groups=None,
                                   approx="hermite")  # Squared exponential with lenghtscale 0.5 with 100 basis functions
            Phi = emb.embed(X.double())
            K = torch.t(Phi) @ Phi
            # todo use torch throughout instead
            K = K.detach().cpu().numpy()
            Phi = Phi.detach().cpu().numpy()
        else:
            self.n_components = 100
            rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)
            Phi = rbf_feature.fit_transform(self.X_train_)
            K = Phi.T @ Phi

        K[np.diag_indices_from(K)] += (1e-1)
        mat = np.eye(K.shape[0]) + K * 1 ** -2

        # Phi2 = emb.embed(torch.from_numpy(y).double())

        noise = 1
        alpha = np.linalg.solve(mat, Phi.T @ y)[:, 0]
        mdl_model_train = alpha @ mat @ alpha.T

        mean_pred = noise ** -2 * \
                    Phi @ np.linalg.solve(mat, Phi.T @ y)[:, 0]

        # Support multi-dimensional output of self.y_train_
        y_train = y
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
        var_pred = np.einsum(
            'fn, fn -> n',
            y_train.T,
            np.linalg.solve(mat, Phi.T @ y_train),
        )
        mdl_pen_train = 1 / 2 * np.log(np.linalg.det(np.identity(K.shape[0]) + 1 ** 2 * K))

        # plt.scatter(self.X_train_, self.y_train_)
        # plt.scatter(self.X_train_, mean_pred)

        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf

        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        # alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

        # Alg 2.1, page 19, line 7
        # -0.5 . y^T . alpha - sum(log(diag(L))) - n_samples / 2 log(2*pi)
        # y is thought to be a (1, n_samples) row vector
        log_likelihood_dims = (-0.5 * (Phi.T @ y).T
                               @ alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        # the log likehood is sum-up across the outputs
        log_likelihood = log_likelihood_dims.sum(axis=-1)

        mdl_lik_train = -log_likelihood

        mdl_score = mdl_lik_train + mdl_model_train + mdl_pen_train

        self.mdl_lik_train = mdl_lik_train
        self.mdl_model_train = mdl_model_train
        self.mdl_pen_train = mdl_pen_train
        self.mdl_train = mdl_score

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        raise NotImplementedError
        mdl, log_lik, m_penalty, X_penalty = self.predict(X_test, y_test, return_mdl=True)
        return mdl, log_lik, m_penalty, X_penalty
