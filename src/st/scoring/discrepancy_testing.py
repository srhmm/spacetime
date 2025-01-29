import itertools

import numpy as np
from enum import Enum

from causallearn.utils.cit import CIT
from st.utils.util import pval_to_map, is_insignificant
from st.utils.util_tests import test_mechanism

from itertools import product


class DiscrepancyTestType(Enum):
    """ Conditional distribution discrepancy testing.
        :Methods:
    * *SKIP* -- placeholder if None
    * *KCI* -- kernelized conditional discrepancy testing
    * *MDL* -- GP regression and  MDL no-hypercompression
    """
    SKIP = 0
    KCI = 1
    MDL = 2

    def __str__(self):
        return 'SKIP' if self.value == 0 else 'KCD' if self.value == 1 else 'GP'

    def __eq__(self, other):
        return self.value == other.value


def discrepancy_test_all(
        discrepancy_test_type: DiscrepancyTestType,
        gp_time_space, cs, ws, pa_i: list, alpha=0.5):
    """ Test regime conditional distribution, disregarding contexts

    :param discrepancy_test_type: test type
    :param gp_time_space: models per regime and context, returned by ``fit_gp_time_space()``
    :param cs: list of contexts
    :param ws: list of windows
    :param pa_i: causal parents of target
    :param alpha: threshold
    :return:
    """
    Dc_list = list()
    Ds = dict.fromkeys(product(cs, ws))
    for (c, w) in Ds.keys():
        _, _, _, data, D = gp_time_space[c][w]
        Dc_list.append(np.hstack((D, data.reshape(-1, 1))))

    n_c = len(Dc_list)

    pval_mat = np.ones((n_c, n_c))
    mp = [0 for _ in range(n_c)]

    if discrepancy_test_type == DiscrepancyTestType.SKIP:
        pass

    elif discrepancy_test_type == DiscrepancyTestType.KCI:
        parents = [1 if node_i in pa_i else 0 for node_i in range(Dc_list[0].shape[1])]
        pval_mat = _test_mechanism_regimes_KCI(
            Dc_list, Dc_list[0].shape[1] - 1, parents, ws, 'kci')  # H0: c ⊥ i | pa_i i.e. same contexts
        mp = pval_to_map(pval_mat, alpha=alpha)  # high p-value = same context (proba of the obs under H0)

    elif discrepancy_test_type == DiscrepancyTestType.MDL:
        pval_mat = _test_mechanism_regimes_GP(gp_time_space, ws)
        mp = pval_to_map(pval_mat, alpha=alpha)

    else:
        raise ValueError(discrepancy_test_type)

    return mp, pval_mat


def _test_mechanism_regimes_KCI(Xs, m, parents, ws, test='kci'):
    """Tests a mechanism (regimes only, all contexts taken together)"""
    R = len(ws)
    parents = np.asarray(parents).astype(bool)
    if test == 'linear_params':
        pvalues = np.ones((R, R, 2))
    else:
        pvalues = np.ones((R, R))

    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            # Kernel-based conditional independence test
            assert len(Xs) > 1
            # Test X \indep E | PA_X
            data = np.vstack([
                np.vstack([np.block([np.reshape([0] * Xs[i].shape[0], (-1, 1)), Xs[i]]) for i in range(len(Xs)) if
                           i % R == r1]),
                np.vstack([np.block([np.reshape([1] * Xs[i].shape[0], (-1, 1)), Xs[i]]) for i in range(len(Xs)) if
                           i % R == r2])
            ])
            condition_set = tuple(np.where(parents > 0)[0] + 1)

            kci_obj = CIT(data, "kci")
            pvalue = kci_obj(0, m + 1, condition_set)

            pvalues[r1, r2] = pvalue
            pvalues[r2, r1] = pvalue

    return pvalues


def _test_mechanism_regimes_GP(gp_time_space, ws):
    """ experimental: test for discrepancy using mdl"""
    R = len(ws)
    pval_mat = np.ones((R, R))
    for w_i in range(R):
        for w_j in range(w_i + 1, R):
            if w_i == w_j: continue
            _, gp_i, data_pa_i, data_i, D1 = gp_time_space[0][w_i]  # c=0 because only one context
            _, gp_j, data_pa_j, data_j, D2 = gp_time_space[0][w_j]

            mdl_ij, loglik_ij, _, _ = gp_i.mdl_score_ytest(data_pa_j, data_j)
            mdl_ji, loglik_ji, _, _ = gp_j.mdl_score_ytest(data_pa_i, data_i)
            mdl_ii, loglik_ii, _, _ = gp_i.mdl_score_ytest(data_pa_i, data_i)
            mdl_jj, loglik_jj, _, _ = gp_j.mdl_score_ytest(data_pa_j, data_j)

            assert (loglik_ij > 0 and loglik_ji > 0
                    and loglik_ii > 0 and loglik_jj > 0)
            eq_i = is_insignificant(abs(loglik_ii - loglik_ji))
            eq_j = is_insignificant(abs(loglik_jj - loglik_ij))
            pval_mat[w_i, w_j] = int(eq_i or eq_j)  # or or and?
            pval_mat[w_j, w_i] = int(eq_i or eq_j)
    return pval_mat


def discrepancy_test_pair(
        gp_time_space, c_i, w_i, c_j, w_j,
        node_i: int, pa_i: list, discrepancy_test_type: DiscrepancyTestType, alpha=0.5):
    """ Tests conditional distributions for equality for a pair of contexts

    :param gp_time_space: models per regime and context, returned by ``fit_gp_time_space()``
    :param w_j: regime 1
    :param c_j: context 1
    :param w_i: regime 2
    :param c_i: context 2
    :param node_i: target node
    :param pa_i: causal parents of target
    :param discrepancy_test_type: test type
    :param alpha:
    :return:
    """
    max_lag = 100

    _, gp_i, data_pa_i, data_i, D1 = gp_time_space[c_i][w_i]
    _, gp_j, data_pa_j, data_j, D2 = gp_time_space[c_j][w_j]

    Dc = np.array([
        np.hstack((D1, data_i.reshape(-1, 1))),
        np.hstack((D2, data_j.reshape(-1, 1)))])  # target node values at time t in last position
    n_c = Dc.shape[0]
    n_vars = D1.shape[1]
    assert (n_vars == D2.shape[1])
    n_pairs = len([i for i in itertools.combinations(range(n_c), 2)])
    pval_mat = np.ones((n_pairs, n_pairs))
    mp = [0 for _ in range(n_c)]

    if discrepancy_test_type == DiscrepancyTestType.SKIP:
        pass
    elif discrepancy_test_type == DiscrepancyTestType.KCI:
        mp, pval_mat = _test_mechanism_KCI(Dc, pa_i, max_lag, n_vars, alpha)
    elif discrepancy_test_type == DiscrepancyTestType.MDL:
        mp, pval_mat = _test_mechanism_GP(data_pa_i, data_pa_j, data_j, data_i, gp_i, gp_j)
    else:
        raise ValueError()

    return mp, pval_mat


def _test_mechanism_KCI(Dc, pa_i, max_lag, n_vars, alpha):
    """ test for discrepancy using KCI"""
    if len(pa_i) == 0:
        parents = [0 for _ in range(n_vars)]
    else:
        parents = [1 if node_i in pa_i and not isinstance(pa_i[0], tuple) \
                   or (isinstance(pa_i[0], tuple) and True in [(node_i, t) in pa_i for t in range(max_lag)])
                   else 0 for node_i in range(n_vars)]
    try:
        pval_mat = test_mechanism(Dc, n_vars, parents, 'kci', {})  # H0: c ⊥ i | pa_i i.e. same contexts
        mp = pval_to_map(pval_mat, alpha=alpha)  # high p-value = same context (proba of the obs under H0)
    except ValueError:
        mp = [0, 0]
        pval_mat = None
        print("Discrepancy test: ValueError")  # zero variance error
    return mp, pval_mat


def _test_mechanism_GP(data_pa_i, data_pa_j, data_j, data_i, gp_i, gp_j):
    """ experimental: test for discrepancy using mdl"""
    data_pa_ij = np.concatenate([data_pa_i, data_pa_j])
    data_ij = np.concatenate([data_i, data_j])

    mdl_ij, loglik_ij, _, _ = gp_i.mdl_score_ytest(data_pa_j, data_j)
    mdl_ji, loglik_ji, _, _ = gp_j.mdl_score_ytest(data_pa_i, data_i)
    mdl_ii, loglik_ii, _, _ = gp_i.mdl_score_ytest(data_pa_i, data_i)
    mdl_jj, loglik_jj, _, _ = gp_j.mdl_score_ytest(data_pa_j, data_j)

    assert (loglik_ij > 0 and loglik_ji > 0
            and loglik_ii > 0 and loglik_jj > 0)

    eq_i = is_insignificant(np.abs(loglik_ii - loglik_ji))
    eq_j = is_insignificant(np.abs(loglik_jj - loglik_ij))

    mp = [0, 0] if (eq_i or eq_j) else [0, 1]
    pval_mat = mp
    return mp, pval_mat
