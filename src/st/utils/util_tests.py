from sklearn.metrics import r2_score
import numpy as np
from st.utils.kcd import KCD
from causallearn.utils.cit import CIT



# Adapted from sparseshift package
def test_mechanism(Xs, m, parents, test='kci', test_kwargs={}, m_data=None, parents_data=None):
    """Tests a mechanism"""

    E = len(Xs)
    parents = np.asarray(parents).astype(bool)
    if test == 'linear_params':
        pvalues = np.ones((E, E, 2))
    else:
        pvalues = np.ones((E, E))

    for e1 in range(E):
        for e2 in range(e1 + 1, E):
            if sum(parents) == 0:
                from hyppo.ksample import MMD
                #MMD = None
                #raise NotImplementedError("incl hyppo.")
                if m_data is not None and parents_data is not None:
                    stat, pvalue = MMD().test(
                        Xs[e1][:, m].reshape(-1, 1),
                        Xs[e2][:, m].reshape(-1, 1),
                    )
                else:
                    stat, pvalue = MMD().test( # Maximum Mean Discrepency (MMD) test statistic and p-value
                        Xs[e1][:, m].reshape(-1, 1),
                        Xs[e2][:, m].reshape(-1, 1),
                    ) # stat is a distance, the lowest the p-value, the more confident we are
            else:
                if test == 'kcd':
                    assert len(Xs) == 2
                    _, pvalue = KCD(n_jobs=test_kwargs['n_jobs']).test(
                        np.vstack((Xs[e1][:, parents], Xs[e2][:, parents])),
                        np.concatenate((Xs[e1][:, m], Xs[e2][:, m])),
                        np.asarray([0] * Xs[e1].shape[0] + [1] * Xs[e2].shape[0]),
                        reps=test_kwargs['n_reps'],
                    )
                elif test == 'invariant_residuals':
                    assert len(Xs) == 2
                    pvalue, *_ = invariant_residual_test(
                        np.vstack((Xs[e1][:, parents], Xs[e2][:, parents])),
                        np.concatenate((Xs[e1][:, m], Xs[e2][:, m])),
                        np.asarray([0] * Xs[e1].shape[0] + [1] * Xs[e2].shape[0]),
                        **test_kwargs
                    )
                elif test == 'fisherz':
                    assert len(Xs) > 1
                    # Test X \indep E | PA_X
                    data = np.block([
                        [np.reshape([0] * Xs[e1].shape[0], (-1, 1)), Xs[e1]],
                        [np.reshape([1] * Xs[e2].shape[0], (-1, 1)), Xs[e2]]
                    ])
                    condition_set = tuple(np.where(parents > 0)[0] + 1)
                    #pvalue = fisherz(data, 0, m+1, condition_set)
                    fisherz_obj = CIT(data, "fisherz")  # construct a CIT instance with data and method name
                    pvalue = fisherz_obj(0, m + 1, condition_set)

                elif test == 'kci':
                    # Kernel-based conditional independence test
                    assert len(Xs) > 1
                    # Test X \indep E | PA_X
                    data = np.block([
                        [np.reshape([0] * Xs[e1].shape[0], (-1, 1)), Xs[e1]],
                        [np.reshape([1] * Xs[e2].shape[0], (-1, 1)), Xs[e2]]
                    ])
                    condition_set = tuple(np.where(parents > 0)[0] + 1)

                    # update of causallearn:
                    kci_obj = CIT(data, "kci")
                    pvalue = kci_obj(0, m+1, condition_set)

                    #pvalue = kci(data, 0, m + 1, condition_set)

                elif test == 'linear_params':
                    assert len(Xs) == 2
                    pvalue, *_ = invariant_residual_test(
                        np.vstack((Xs[e1][:, parents], Xs[e2][:, parents])),
                        np.concatenate((Xs[e1][:, m], Xs[e2][:, m])),
                        np.asarray([0] * Xs[e1].shape[0] + [1] * Xs[e2].shape[0]),
                        combine_pvalues=False,
                        **test_kwargs
                    )
                else:
                    raise ValueError(f'Test {test} not implemented.')
            pvalues[e1, e2] = pvalue
            pvalues[e2, e1] = pvalue

    return pvalues


def invariant_residual_test(
    X,
    Y,
    z,
    method="gam",
    test="ks",
    method_kwargs={},
    return_model=False,
    combine_pvalues=True,
):
    r"""
    Calulates the 2-sample test statistic.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Features to condition on
    Y : ndarray, shape (n,)
        Target or outcome features
    z : list or ndarray, shape (n,)
        List of zeros and ones indicating which samples belong to
        which groups.
    method : {"forest", "gam", "linear"}, default="gam"
        Method to predict the target given the covariates
    test : {"whitney_levene", "ks"}, default="ks"
        Test of the residuals between the groups
    method_kwargs : dict
        Named arguments to pass to the prediction method.
    return_model : boolean, default=False
        If true, returns the fitted model
    combine_pvalues: bool, default=True
        If True, returns hte minimum of the corrected pvalues.

    Returns
    -------
    pvalue : float
        The computed *k*-sample p-value.
    r2 : float
        r2 score of the regression fit
    model : object
        Fitted regresion model, if return_model is True
    """

    if method == "forest":
        from sklearn.ensemble import RandomForestRegressor

        predictor = RandomForestRegressor(max_features="sqrt", **method_kwargs)
    elif method == "gam":
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import SplineTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV

        pipe = Pipeline(
            steps=[
                ("spline", SplineTransformer(n_knots=4, degree=3)),
                ("linear", LinearRegression(**method_kwargs)),
            ]
        )
        param_grid = {
            "spline__n_knots": [3, 5, 7, 9],
        }
        predictor = GridSearchCV(
            pipe, param_grid, n_jobs=-2, refit=True,
            scoring="neg_mean_squared_error"
        )
    elif method == "linear":
        from sklearn.linear_model import LinearRegression

        predictor = LinearRegression(**method_kwargs)
    else:
        raise ValueError(f"Method {method} not a valid option.")

    predictor = predictor.fit(X, Y)
    Y_pred = predictor.predict(X)
    residuals = Y - Y_pred
    r2 = r2_score(Y, Y_pred)

    if test == "whitney_levene":
        from scipy.stats import mannwhitneyu
        from scipy.stats import levene

        _, mean_pval = mannwhitneyu(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
        _, var_pval = levene(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
        # Correct for multiple tests
        if combine_pvalues:
            pval = min(mean_pval * 2, var_pval * 2, 1)
        else:
            pval = (min(mean_pval * 2, 1), min(var_pval * 2, 1))
    elif test == "ks":
        from scipy.stats import kstest

        _, pval = kstest(
            residuals[np.asarray(z, dtype=bool)],
            residuals[np.asarray(1 - z, dtype=bool)],
        )
    else:
        raise ValueError(f"Test {test} not a valid option.")

    if return_model:
        return pval, r2, predictor
    else:
        return pval, r2
