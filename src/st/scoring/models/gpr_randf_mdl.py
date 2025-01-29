
import numpy as np
from scipy.linalg import cholesky, cho_solve
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import mean_squared_error

GPR_CHOLESKY_LOWER = True

"""Approximate Gaussian processes with Bayesian linear regression and random fourier features """


class GaussianProcessRFF:
    def __init__(self, n_components=200):
        self.n_components = n_components

        self.alpha_ = None
        self.kernel_approx_ = None
        self.X_train_ = None
        self.y_train_ = None

        self.mdl_lik_train = None
        self.mdl_model_train = None
        self.mdl_pen_train = None
        self.mdl_train = None

        self.small_alpha = (1e-10)

    def set_alpha(self, alpha):
        self.small_alpha = alpha

    def fit(self, X_train, y_train):

        noise = 1
        self.X_train_ = np.copy(X_train)
        self.y_train_ = np.copy(y_train)
        rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)
        X_feat = rbf_feature.fit_transform(X_train)
        X_feat = X_feat.T

        K = X_feat.T @ X_feat
        K_dual = X_feat @ X_feat.T

        assert K.shape[0] == len(X_train) & K.shape[1] == len(X_train)
        assert K_dual.shape[0] == self.n_components & K_dual.shape[1] == self.n_components
        self.kernel_approx_ = np.copy(K)
        self.K_dual = K_dual
        K_dual[np.diag_indices_from(K_dual)] += (1e-1)  # + (1e-1) # st. MDL scores positive and K pos. definite
        mat = np.eye(X_feat.shape[0]) + K_dual * noise ** -2

        alpha_dual1 = np.linalg.solve(mat, X_feat @ y_train.reshape(-1, 1))[:, 0]
        self.alpha_ = alpha_dual1.reshape(-1, 1)

        L = cholesky(K_dual, lower=True, check_finite=False)
        alpha_dual = cho_solve((L, True),
                               X_feat @ y_train.reshape(-1, 1),
                               check_finite=False)
        mdl_penalty = alpha_dual.T @ mat @ alpha_dual
        mdl_penalty = mdl_penalty[0][0]

        self.L_ = L

        log_lik = -0.5 * np.einsum("ik,ik->k", X_feat @ y_train.reshape(1, -1).T, alpha_dual)
        log_lik -= np.log(np.diag(L)).sum()
        log_lik -= K_dual.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = -log_lik.sum(axis=-1)

        # Alternative:
        mean_pred = noise ** -2 * X_feat.T @ np.linalg.solve(mat, X_feat @ self.y_train_.reshape(-1, 1))[:, 0]
        log_lik = -np.log(mean_squared_error(self.y_train_, mean_pred))

        sigma = 1
        X_penalty = 1 / 2 * np.log(np.linalg.det(
            np.identity(self.kernel_approx_.shape[0]) + sigma ** 2 * self.kernel_approx_))

        mdl_score = log_lik + mdl_penalty + X_penalty

        self.mdl_lik_train = log_lik
        self.mdl_model_train = mdl_penalty
        self.mdl_pen_train = X_penalty
        self.mdl_train = mdl_score

    def mdl_score_ytrain(self):
        return self.mdl_train, self.mdl_lik_train, self.mdl_model_train, self.mdl_pen_train

    def mdl_score_ytest(self, X_test, y_test):
        mdl, mdl_lik, mdl_model, mdl_pen = self._mdl_score(X_test, y_test)
        return mdl, mdl_lik, mdl_model, mdl_pen

    def _mdl_score(self, X_test, y_test):
        rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)
        D_n = len(self.X_train_)
        X = np.concatenate([X_test, self.X_train_])
        X_feat = rbf_feature.fit_transform(X)
        X_feat_test = X_feat.T[:, :-D_n]
        X_feat_train = X_feat.T[:, -D_n:]
        noise = 1
        X_feat = X_feat_train

        K_dual = self.K_dual
        alpha_dual = self.alpha_

        log_lik = -0.5 * np.einsum("ik,ik->k", X_feat_test @ y_test.reshape(-1, 1), alpha_dual)
        log_lik -= np.log(np.diag(self.L_)).sum()
        log_lik -= K_dual.shape[0] / 2 * np.log(2 * np.pi)
        log_lik = -log_lik.sum(axis=-1)
        mdl_penalty = self.mdl_model_train

        ##Alt. log lik:
        mat = np.eye(X_feat_train.shape[0]) + X_feat_train @ X_feat_train.T * noise ** -2
        mean_pred = noise ** -2 * X_feat_test.T @ np.linalg.solve(mat, X_feat_train @ self.y_train_.reshape(-1, 1))[:,
                                                  0]
        log_lik = -np.log(mean_squared_error(y_test, mean_pred))

        sigma = 1
        X_penalty = 1 / 2 * np.log(
            np.linalg.det(np.identity(self.kernel_approx_.shape[0]) + sigma ** 2 * self.kernel_approx_))

        mdl_score = log_lik + mdl_penalty + X_penalty
        return mdl_score, log_lik, mdl_penalty, X_penalty

    def predict(self, X_test, var=False):
        noise = 1
        D_n = self.X_train_.shape[0]

        if X_test.shape[0] < self.n_components:
            print("RFF: # features > # samples")

        rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.n_components)

        X = np.concatenate([X_test, self.X_train_])
        X_feat = rbf_feature.fit_transform(X)
        X_feat_test = X_feat.T[:, :-D_n]
        X_feat_train = X_feat.T[:, -D_n:]

        mat = np.eye(X_feat_train.shape[0]) + X_feat_train @ X_feat_train.T * noise ** -2

        mean_pred = noise ** -2 * X_feat_test.T @ np.linalg.solve(mat, X_feat_train @ self.y_train_.reshape(-1, 1))[:,
                                                  0]
        var_pred = np.einsum('fn, fn -> n', X_feat_test, np.linalg.solve(mat, X_feat_test))

        if var:
            return mean_pred, var_pred
        return mean_pred
