import numpy as np
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from st.utils.util import data_scale
from st.sttypes import TimeseriesScoringFunction, SubsamplingApproach


def fit_gp_time_space(**kwargs):
    """ fit we use in CPS search (needs gp residuals) """
    return fit_functional_model_time_space(
        TimeseriesScoringFunction.GP, SubsamplingApproach.HORIZONTAL, return_models=True, **kwargs)


def fit_functional_model_time_space(
        scoring_function: TimeseriesScoringFunction,
        subsampling_approach: SubsamplingApproach,
        data_C, windows_T, target_links, target, contexts=None, regimes=None, return_models=False):
    """ Fit a functional model over time (regimes) and space (context)

    :param scoring_function: MDL score
    :param subsampling_approach: subsampling approach
    :param data_C: data
    :param windows_T: time windows segmented by cutpoints
    :param target_links: causal links towards target
    :param target: target node
    :param contexts: if known, contexts (dataset groups)
    :param regimes: if known, regimes (time window groups)
    :param return_models: return regression models
    :return:
    """

    gp_time_space = dict()
    contexts = list(range(len(data_C))) if contexts is None else contexts
    regimes = list(range(len(windows_T))) if regimes is None else regimes

    for context in set(contexts):
        gp_time_space[context] = dict()
        for regime in set(regimes):

            pa_i = [(var, lag) for (var, lag), _, _ in target_links]
            M = len(pa_i)
            data_pa_i, data_all, data_node_i = collect_subsample(subsampling_approach,
                data_C=data_C, target_links=target_links, context=context, regime=regime, contexts=contexts,
                regimes=regimes, target=target, windows_T=windows_T)
            if M == 0:
                data_pa_i = data_node_i.reshape(-1, 1)
            score, model = fit_functional_model(scoring_function, data_pa_i, data_node_i, pa_i, data_C)
            gp_time_space[context][regime] = (score, model, data_pa_i, data_node_i, data_all) if return_models else score

    return gp_time_space


def fit_functional_model(scoring_function: TimeseriesScoringFunction, data_pa_i, data_node_i, pa_i, data_C):
    """ Fits a functional model with corresponding MDL score. """
    if scoring_function.value == TimeseriesScoringFunction.GLOBE.value:
        score, spline = fit_MARS_regression_spline(data_C, pa_i, data_pa_i, data_node_i)
        return score, spline
    else:
        assert scoring_function.value in [
            TimeseriesScoringFunction.GP.value,
            TimeseriesScoringFunction.GP_QFF.value
        ]
        gp = fit_gaussian_process(
            data_pa_i, data_node_i,
            scoring_function=scoring_function,
            check_fit=False)
        score, lik, model, pen = gp.mdl_score_ytrain()
        return score, gp


def collect_subsample(subsampling_approach: SubsamplingApproach, **kwargs):
    if subsampling_approach == SubsamplingApproach.HORIZONTAL:
        return subsample_autoregressive(**kwargs)
    elif subsampling_approach == SubsamplingApproach.JUMPING_WINDOW:
        return subsample_jumping_window(**kwargs)
    elif subsampling_approach == SubsamplingApproach.VERTICAL:
        return subsample_vertical(**kwargs)
    else:
        raise ValueError(SubsamplingApproach)


def subsample_autoregressive(data_C, target_links, context, regime, contexts, regimes, target, windows_T):
    """ Subsample the time series. """
    pa_i = [(var, -np.abs(lag)) for (var, lag), _, _ in target_links]  # convention neg lags
    M = len(pa_i)
    for k in data_C:
        N = data_C[k].shape[1]
    data_pa_i = np.zeros((1, M), dtype='float32')
    data_all = np.zeros((1, N + 1), dtype='float32')
    data_node_i = np.zeros((1), dtype='float32')

    for dataset in [d for d in range(len(data_C)) if contexts[d] == context]:
        data = data_C[dataset]

        for window, (t0, tn) in enumerate(windows_T):
            if regimes[window] != regime: continue
            T = tn - t0
            data_pa_i_w = np.zeros((T, M), dtype='float32')
            data_all_w = np.zeros((T, N + 1), dtype='float32')

            for j, (var, lag) in enumerate(pa_i):
                if var == target:
                    data_all_w[:, N] = data[t0 + lag:tn + lag, var]
                data_pa_i_w[:, j] = data[t0 + lag:tn + lag, var]
                data_all_w[:, var] = data[t0 + lag:tn + lag, var]

            data_pa_i = np.concatenate([data_pa_i, data_pa_i_w])
            data_all = np.concatenate([data_all, data_all_w])
            data_node_i = np.concatenate([data_node_i, data[t0:tn, target]])

    data_pa_i = data_pa_i[1:]
    data_all = data_all[1:]
    data_node_i = data_node_i[1:]

    return data_pa_i, data_all, data_node_i


def subsample_jumping_window(data_C, target_links, context, regime, contexts, regimes, target, windows_T):
    raise NotImplementedError("Experiment with this.")


def subsample_vertical(data_C, target_links, context, regime, contexts, regimes, target, windows_T):
    raise NotImplementedError("Experiment with this.")


def fit_MARS_regression_spline(data_C, pa_i, data_pa_i, data_node_i):
    """ Spline Regression. Mini GLOBE implementation (Mian et al. 2021)
    :param data_C:
    :param pa_i:
    :param data_pa_i:
    :param data_node_i:
    :return:
    """
    from st.scoring.models.spline_mdl import Slope

    def _min_diff(tgt):
        sorted_v = np.copy(tgt)
        sorted_v.sort(axis=0)
        diff = np.abs(sorted_v[1] - sorted_v[0])
        if diff == 0: diff = np.array([10.01])
        for i in range(1, len(sorted_v) - 1):
            curr_diff = np.abs(sorted_v[i + 1] - sorted_v[i])
            if curr_diff != 0 and curr_diff < diff:
                diff = curr_diff
        return diff

    def _combinator(M, k):
        from scipy.special import comb
        sum = comb(M + k - 1, M)
        if sum == 0:
            return 0
        return np.log2(sum)

    def _aggregate_hinges(interactions, k, slope_, F):
        cost = 0
        for M in hinges:
            cost += slope_.logN(M) + _combinator(M, k) + M * np.log2(F)
        return cost

    source_g = data_pa_i
    target_g = data_node_i
    slope_ = Slope()
    globe_F = 9
    k, dim, M, rows, mindiff = np.array([len(pa_i)]), data_C[0].shape[1], 3, data_C[0].shape[0], _min_diff(target_g)
    base_cost = slope_.model_score(k) + k * np.log2(dim)
    sse, model, coeffs, hinges, interactions = slope_.FitSpline(source_g, target_g, M, False)
    base_cost = base_cost + slope_.model_score(hinges) + _aggregate_hinges(interactions, k, slope_, globe_F)
    cost = slope_.gaussian_score_emp_sse(sse, rows, mindiff) + model + base_cost
    return cost, slope_


def fit_gaussian_process(
        X, y, scoring_function=TimeseriesScoringFunction.GP,
        alpha=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2),
        scale=True, grid_search=False,
        show_plt=False, check_fit=False):
    """GP regression.

    :param X: parents
    :param y: target
    :param scoring_function: gp or qff
    :param alpha: rbf kernel param
    :param length_scale: rbf kernel param
    :param length_scale_bounds: rbf kernel param
    :param scale: scale data
    :param grid_search: kernel parameter tuning
    :param show_plt: plot
    :param check_fit:
    :return: GP_k, gaussian process per context/group
    """
    kernel = 1 * RBF(length_scale=length_scale,
                     length_scale_bounds=length_scale_bounds)
    size_tr_local = len(X)
    tr_indices = np.sort(np.random.RandomState().choice(len(X), size=size_tr_local, replace=False))
    Xtr = X[tr_indices]
    ytr = y[tr_indices]

    if scale:
        Xtr = data_scale(Xtr)
        ytr = data_scale(ytr.reshape(-1, 1))

    # Optional: Grid search for kernel parameter tuning
    param_grid = [{
        "alpha": [1e-2, 1e-3],
        "kernel": [RBF(l) for l in np.logspace(-1, 1, 2)]
    }, {
        "alpha": [1e-2, 1e-3],
        "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]}]
    score = "r2"

    if grid_search:
        gaussianProcess = scoring_function.get_model(kernel, alpha, n_restarts_optimizer=9)
        gaussianProcessGrid = GridSearchCV(estimator=gaussianProcess, param_grid=param_grid, cv=4,
                                           scoring='%s' % score)
        gaussianProcessGrid.fit(Xtr, ytr)
        gp = gaussianProcessGrid.best_estimator_
    else:
        gaussianProcess = scoring_function.get_model(kernel, alpha, n_restarts_optimizer=9)
        gaussianProcess.fit(Xtr, ytr)

        gp = gaussianProcess
        if check_fit:
            y_pred = gaussianProcess.predict(Xtr, ytr, return_mdl=False)
            score = r2_score(ytr, y_pred)
            assert score >= .5

    if show_plt:
        predictions = gp.predict(Xtr, ytr, return_mdl=False)
        plt.scatter(Xtr, ytr, label=" Values", linewidth=.2, marker=".")
        plt.scatter(Xtr, predictions, label=" Predict", linewidth=.4, marker="+")
    if show_plt: plt.legend()
    return gp
