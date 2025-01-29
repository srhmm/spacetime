import math
import numpy as np
from scipy import stats
from itertools import combinations

from st.scoring.discrepancy_testing import discrepancy_test_all, DiscrepancyTestType, discrepancy_test_pair
from st.scoring.time_space_fitting import fit_gp_time_space, subsample_autoregressive
from st.utils.util import pval_to_map
from st.utils.util_regimes import cuts_to_windows_T, windows_T_to_r_partition, regimes_map_from_constraints, pelt


""" Regime Changepoint Search """


def mdl_error_cp_search(datasets: dict, links: dict, skip: int, min_dur=10, single_run=True, return_residuals=False):
    """ CPS: Our main search algorithm to discover the regime changepoints and partitions.

    :param datasets: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
    :param links: causal links, ``links[i][j]`` contains ``(k, lag, coef)`` when k is parent of j in context i.
    :param skip: time points to skip at start of time window, set to max lag+1
    :param min_dur: min distance between any two cutpoints
    :param single_run:
    :param return_residuals:
    :return: ``r_partition`` e.g. [(0, 145, 0.0), (145, 55, 2.0)]
            ``c_partition`` e.g. [0, 1]
            ``contexts_per_node``, ``regimes_per_node`` resp partitions for each node in the graph
    """

    cuts = [0]
    error_data = list()
    params = {'T': datasets[0].shape[0], 'N': datasets[0].shape[1]}

    while cuts[-1] != params['T']:
        # Fit models on data from last cut to min_dur
        tmp_cuts = [cuts[-1], min(params['T'] - 1, cuts[-1] + min_dur)]
        windows_T = cuts_to_windows_T(tmp_cuts, skip)
        current_mdl, current_r_partition, current_contexts, current_regimes, current_gps, hist, _ = (
            eval_fit_partition(datasets, links, windows_T, skip))

        errors = list()
        for node in range(params['N']):
            gps, contexts = current_gps, current_contexts
            error = get_error(datasets, links, gps, contexts, skip, params)
            for e in error[node].values():
                errors.append(e)
        data = np.column_stack([c for c in errors])
        bkpts = pelt(data, min_dur)
        if single_run:
            cuts = [0] + [b + skip for b in bkpts]
            error_data.append(data)
        else:
            cuts.append([b + skip for b in bkpts if b > cuts[-1]][0])

    windows_T = cuts_to_windows_T(cuts, skip)
    regimes_per_node = partition_regimes(datasets, links, windows_T, params)
    regimes = regimes_map_from_constraints(regimes_per_node)
    useless_cpts = [i for i in range(1, len(regimes)) if regimes[i] == regimes[i - 1]]
    for c in reversed(useless_cpts):
        windows_T[c] = (windows_T[c - 1][0], windows_T[c][1])
        windows_T[c - 1] = (windows_T[c - 1][0], windows_T[c][1])
    for c in reversed(useless_cpts):
        windows_T = windows_T[:c] + windows_T[c + 1:]
        regimes = np.concatenate([regimes[:c], regimes[c + 1:]])
        for node in range(params['N']):
            regimes_per_node[node] = np.concatenate([regimes_per_node[node][:c], regimes_per_node[node][c + 1:]])
    r_partition = windows_T_to_r_partition(windows_T, skip, regimes)

    contexts_per_node = partition_contexts_over_regimes(datasets, links, regimes, windows_T, params)
    c_partition = regimes_map_from_constraints(contexts_per_node)

    if return_residuals:
        return r_partition, c_partition, contexts_per_node, regimes_per_node, error_data
    else:
        return r_partition, c_partition, contexts_per_node, regimes_per_node


def _arbitrary_cps(datasets, links, params, options, w_size, skip):
    windows_T = [(b + skip, b + w_size) for b in range(0, datasets.shape[1], w_size)]
    res = partition_regimes(datasets, links, windows_T, params)
    regimes = regimes_map_from_constraints(res)

    return windows_T_to_r_partition(windows_T, skip, regimes)


def iterative_cp_search(datasets, links, skip, grain=None):
    """
    CPS search variant. Starting from one unique regime, iteratively add the regime cutpoint decreasing the most the MDL until no more improvement
    :param datasets:
    :param links:
    :param skip:
    :param grain: The interval between two interval cutpoints
    :return:
    """
    grain = grain or int(len(datasets[0]) / 10)
    # Initially, only one regime for the whole data
    windows_T = [(skip, datasets[0].shape[0])]
    current_mdl, current_r_partition, current_contexts, current_regimes, gps, hist, _ = (
        eval_fit_partition(datasets, links, windows_T, skip))
    # Candidate regime cutpoints initialization
    candidate_cps = list(range(grain, datasets[0].shape[0], grain))
    selected_cps = [0, datasets[0].shape[0]]
    improvement = True
    # While adding a cutpoint improve MDL
    while improvement and len(candidate_cps) > 0:
        # compute global mdl
        partitions = dict.fromkeys(candidate_cps)
        mdl_candidate = dict()
        for candidate in candidate_cps:
            cuts = sorted(selected_cps + [candidate])
            candidate_windows_T = cuts_to_windows_T(cuts, skip)
            mdl_candidate[candidate], r_partition, contexts_per_node, regimes_per_node, gps, hist, _ = (
                eval_fit_partition(datasets, links, candidate_windows_T.copy(), skip, hist))
            partitions[candidate] = {'r': r_partition, 'c': contexts_per_node}
        # select candidate decreasing the most the mdl or stop if no improvement
        best = min(mdl_candidate, key=mdl_candidate.get)
        if current_mdl > mdl_candidate[best]:
            selected_cps.append(best)
            selected_cps.sort()
            candidate_cps.remove(best)
            windows_T = cuts_to_windows_T(selected_cps, skip)
            current_mdl = mdl_candidate[best]
            current_r_partition = partitions[best]['r']
            print('Current r_partition: ', end='')
            print(current_r_partition)
            current_contexts = partitions[best]['c']
        else:
            improvement = False
    return current_r_partition, current_contexts


def eval_fit_partition(datasets, links, windows_T, skip, hist=None):
    """
    Find a regime and context partition over all nodes given time windows and return the model MDL score
    :param datasets: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
    :param links: causal links, ``links[i][j]`` contains ``(k, lag, coef)`` when k is parent of j in context i.
    :param windows_T: regime changepoints
    :param skip:
    :return: regime map
    """
    params = {'T': datasets[0].shape[0], 'N': datasets[0].shape[1]}

    hist = hist or list()
    # Regime partition per node
    regimes_per_node = partition_regimes(
        datasets, links, windows_T, params)  # get regime partitions per node (using KCI)
    # Get global regime partition
    regimes = regimes_map_from_constraints(regimes_per_node)
    # Remove cutpoints between consecutive chunks of same regime
    useless_cpts = [i for i in range(1, len(regimes)) if
                    regimes[i] == regimes[i - 1]]
    for c in reversed(useless_cpts):
        windows_T[c] = (windows_T[c - 1][0], windows_T[c][1])
        windows_T[c - 1] = (windows_T[c - 1][0], windows_T[c][1])
    for c in reversed(useless_cpts):
        windows_T = windows_T[:c] + windows_T[c + 1:]
        regimes = np.concatenate([regimes[:c], regimes[c + 1:]])
        for node in range(params['N']):
            regimes_per_node[node] = np.concatenate([regimes_per_node[node][:c], regimes_per_node[node][c + 1:]])
    r_partition = windows_T_to_r_partition(windows_T, skip, regimes)
    # Return if partition already evaluated
    if r_partition in hist: return np.inf, None, None, None, hist
    hist.append(r_partition)

    # Context partition per node
    contexts_per_node = partition_contexts_over_regimes(datasets, links, regimes, windows_T, params)
    # c_partition = np.array([d for d in range(len(datasets))]) # or one per dataset to begin with
    # c_partition = np.array([0 for d in range(len(datasets))])  # or one context for all and then how to identify intervention?

    # Fit GPs
    gps = dict()
    for node in range(params['N']):
        target_links = [(l, 1, None) for (l, w, f) in links[0][node]]
        # vsn = options.get_score_vsn()
        c_partition = contexts_per_node[node]
        gps[node] = fit_gp_time_space(data_C=datasets, windows_T=windows_T, target_links=target_links, target=node,
                                      contexts=c_partition, regimes=regimes_per_node[node])
    # discrepancy_test(gps[node], c_i, w_i, c_j, w_j,
    #                  node_i: int, pa_i: list, discrepancy_test_type: DiscrepancyTestType, alpha = 0.5, verbosity = 0)
    # Done: merge contexts by comparing the GPs, they must be constant over regimes but can differ for different nodes
    # fit GPs (same model for a same context mandatory + same model for most contexts)

    # Evaluate model
    mdl = 0
    for node in range(params['N']):
        for ic, c in enumerate(set(contexts_per_node[node])):
            for ir, r in enumerate(set(regimes_per_node[node])):
                gp = gps[node][c][r][1]
                mdl += gp.mdl_train

    return mdl, r_partition, contexts_per_node, regimes_per_node, gps, hist, None  # mdls


def partition_contexts_over_regimes(
        datasets, links, regimes, windows_T, params, discrepancy_test_type=DiscrepancyTestType.KCI):
    """
    Find a context partition for each node constant over the regimes using KCI
    :param discrepancy_test_type: discrepancy test
    :param datasets:
    :param links:
    :param regimes: Regime map (e.g. [1, 0, 1])
    :param windows_T:
    :param params:
    :return:
    """
    contexts_per_node = dict()
    for node in range(params['N']):
        target_links = [(l, 1, None) for (l, w, f) in links[0][node]]
        pa_i = [var for (var, lag), _, _ in target_links]  # [(var, lag) for (var, lag), _, _ in target_links]
        contexts = list(range(len(datasets)))
        gp_time_space = {d: dict.fromkeys(range(len(windows_T))) for d in contexts}
        pval_mat = np.ones((len(contexts), len(contexts)))

        if discrepancy_test_type == DiscrepancyTestType.KCI:
            for (c1, c2) in list(combinations(contexts, 2)):
                same = True
                for regime in set(regimes):
                    if not same: continue
                    data_pa_i, data_all, data_node_i = subsample_autoregressive(datasets, target_links, c1, regime, contexts, regimes,
                                                                                node, windows_T)
                    gp_time_space[c1][regime] = (None, None, data_pa_i, data_node_i, data_all)
                    data_pa_j, data_all, data_node_j = subsample_autoregressive(datasets, target_links, c2, regime, contexts, regimes,
                                                                                node, windows_T)
                    gp_time_space[c2][regime] = (None, None, data_pa_j, data_node_j, data_all)
                    map, _ = discrepancy_test_pair(gp_time_space, c1, regime, c2, regime, node, pa_i, discrepancy_test_type)
                    if len(set(map)) != 1:
                        same = False
                if not same:
                    pval_mat[c1, c2] = 0
            contexts_per_node[node] = pval_to_map(pval_mat, alpha=0.05)

        elif discrepancy_test_type == DiscrepancyTestType.MDL:
            gp_time_space = fit_gp_time_space(data_C=datasets, windows_T=windows_T, target_links=target_links, target=node,
                                              contexts=contexts, regimes=regimes)
            for (c1, c2) in list(combinations(contexts, 2)):
                same = True
                for regime in set(regimes):
                    if not same: continue
                    map, _ = discrepancy_test_pair(gp_time_space, c1, regime, c2, regime, node, pa_i, discrepancy_test_type)
                    if len(set(map)) == 1:
                        same = False
                if not same:
                    pval_mat[c1, c2] = 0
            contexts_per_node[node] = pval_to_map(pval_mat, alpha=0.05)

    return contexts_per_node


def known_cps(datasets, links, params, options, skip, windows_T):
    res = partition_regimes(datasets, links, windows_T, params)
    regimes = regimes_map_from_constraints(res)

    return windows_T_to_r_partition(windows_T, skip, regimes)


def partition_regimes(datasets, links, windows_T, params, discrepancy_test_type=DiscrepancyTestType.KCI):
    """
    Partition data into regimes given the time windows for each node (using KCI)
    :param discrepancy_test_type: discrepancy test
    :param datasets:
    :param links:
    :param windows_T:
    :param params:
    :return: A dict containing the regime partition found for each node
    """
    res = dict()
    for node in range(params['N']):
        # print('Node ' + str(node))
        target_links = [(l, 1, None) for (l, w, f) in links[0][node]]  # if l[0] != node]
        # if len(target_links) > 0:
        cs = [0 for _ in range(len(datasets))]
        ws = list(range(len(windows_T)))
        if discrepancy_test_type == DiscrepancyTestType.MDL:
            gp_time_space = fit_gp_time_space(
                data_C=datasets, windows_T=windows_T, target_links=target_links, target=node, contexts=cs)
        else:
            gp_time_space = dict()
            gp_time_space[0] = dict.fromkeys(ws)
            for w in ws:
                data_pa_i, data_all, data_node_i = subsample_autoregressive(datasets, target_links, 0, w, cs, ws, node, windows_T)
                gp_time_space[0][w] = (None, None, data_pa_i, data_node_i, data_all)

        map, pval_mat = discrepancy_test_all(
            discrepancy_test_type, gp_time_space, cs=cs, ws=ws,
            pa_i=[n[0] for (n, _, _) in target_links], alpha=0.05)
        res[node] = map
    return res


def _partition_regimes_pvals(datasets, links, windows_T, params, discrepancy_test_type):
    """
    Partition data into regimes given the time windows for each node (using KCI), returning pvals
    :param datasets:
    :param links:
    :param windows_T:
    :param params:
    :param discrepancy_test_type:
    :return: A dict containing the regime partition and pvals found for each node
    """
    assert not (discrepancy_test_type == DiscrepancyTestType.MDL)
    res = dict()
    pv = dict()
    for node in range(params['N']):
        target_links = [(l, 1, None) for (l, w, f) in links[0][node]]
        cs = [0 for _ in range(len(datasets))]
        ws = list(range(len(windows_T)))

        gp_time_space = dict()
        gp_time_space[0] = dict.fromkeys(ws)
        for w in ws:
            data_pa_i, data_all, data_node_i = subsample_autoregressive(datasets, target_links, 0, w, cs, ws, node, windows_T)
            gp_time_space[0][w] = (None, None, data_pa_i, data_node_i, data_all)

        map, pval_mat = discrepancy_test_all(
            discrepancy_test_type, gp_time_space, cs=cs, ws=ws,
            pa_i=[n[0] for (n, _, _) in target_links], alpha=0.05)
        res[node] = map
        pv[node] = pval_mat
    return res, pv


def _mdl_error_cp_search_seq(datasets, links, params, options, skip, min_dur=10, k=5, alpha=0.005):
    cuts = [0]
    while cuts[-1] + min_dur < params['T']:

        # Fit model on data from last cut to min_dur
        tmp_cuts = [cuts[-1], cuts[-1] + min_dur]
        windows_T = cuts_to_windows_T(tmp_cuts, skip)
        current_mdl, current_r_partition, current_contexts, current_regimes, gps, hist, _ = (
            eval_fit_partition(datasets, links, windows_T, skip))

        # Compute error on all data
        error = get_error(datasets, links, gps, current_contexts, skip, params)

        # node, c = 0, 0
        # data = error[node][c]
        # plt.plot(data)
        # plt.ylabel("- log-likelihood")
        # plt.xlabel("t")
        # plt.title("Node " + str(node) + ", context " + str(c))

        cut = params['T']
        for node in gps.keys():
            for c in set(current_contexts[node]):
                # Fit KDE on data used for the training
                tmp = error[node][c][:error[node][c].shape[0] // k * k].reshape((-1, k))
                error[node][c] = np.repeat(np.sum(tmp, 1), k)
                data = error[node][c]
                x_data = data[tmp_cuts[0]:tmp_cuts[1]]
                gkde_obj = stats.gaussian_kde(x_data)

                # x_pts = np.linspace(0, 100, 200)
                # estimated_pdf = gkde_obj.evaluate(x_pts)
                # y_normal = stats.norm.pdf(x_pts)
                # plt.figure()
                # plt.hist(data[min_dur:], bins=7, density=1.0)
                # plt.plot(x_pts, estimated_pdf, label="kde estimated PDF", color="r")
                # plt.legend()
                # plt.show()

                # Compute probability of remaining data until under threshold
                for t in range(tmp_cuts[0], len(data), k):  # range(params['T']):
                    probability = gkde_obj.integrate_box(data[t], np.inf)
                    if probability < alpha:
                        cut = min(cut, t + k - 1)
                        break

        cuts.append(cut)
        cuts.sort()

    if params['T'] not in cuts: cuts.append(params['T'])
    windows_T = cuts_to_windows_T(cuts, skip)
    current_mdl, r_partition, contexts_per_node, regimes_per_node, gps, hist, _ = (
        eval_fit_partition(datasets, links, windows_T, skip))

    return r_partition, contexts_per_node


def _get_regimes_via_permutations(error, cutpoints, min_dur):
    # cutpoints = [0, 100, 200, 300]
    chunks = list(range(len(cutpoints) - 1))
    res = dict()
    for node in range(error.shape[1]): res[node] = np.ones(shape=(len(chunks), len(chunks)))
    for c1, c2 in list(combinations(chunks, 2)):
        if c1 > c2: c1, c2 = c2, c1
        data1 = error[0:max(0, cutpoints[c1] - 1)]
        chunck1 = error[cutpoints[c1]:cutpoints[c1 + 1] - 1]
        data2 = error[cutpoints[c1 + 1]:cutpoints[c2] - 1]
        chunck2 = error[cutpoints[c2]:cutpoints[c2 + 1] - 1]
        data3 = error[cutpoints[c2 + 1]:]
        for node in range(error.shape[1]):
            bkpts = pelt(
                np.hstack([chunck1[:, node], chunck2[:, node], data1[:, node], data2[:, node], data3[:, node]]),
                min_dur, plot=False)  # vstack if all nodes at once
            n = len(chunck1) + len(chunck2)
            if n > bkpts[0] + 20:
                res[node][c1, c2] = 0
                res[node][c2, c1] = 0
    regimes = regimes_map_from_constraints(res)

    return regimes


def get_error(datasets, links, gps, contexts, skip, params):
    """
    Compute negative log-likelihood of data given a model
    :param datasets:
    :param links:
    :param gps:
    :param contexts:
    :param skip:
    :param params:
    :return:
    """
    # Compute error on all data
    error = dict.fromkeys(gps.keys())
    for node in gps.keys():
        node = int(node)
        error[node] = dict()
        for c in set(contexts[node]):
            r = 0
            target_links = [(l, 1, None) for (l, w, f) in links[0][node]]

            if len(target_links) == 0:  # case of empty parent set - use a self link
                target_links = [((node, -1), 1, None)]

            data_pa_i, data_all, data_node_i = subsample_autoregressive(
                datasets, target_links, context=c, regime=r,
                contexts=contexts[node], regimes=[r],
                target=node, windows_T=[(skip, params['T'])])
            model = gps[node][c][r][1]
            K_trans = model.kernel_(data_pa_i, model.X_train_)
            y_mean = K_trans @ model.alpha_
            y_mean = model._y_train_std * y_mean + model._y_train_mean
            sigma = 1
            neg_log_lik = math.log(2 * math.pi * (sigma ** 2)) / 2 + ((data_node_i.reshape(-1, 1) - y_mean) ** 2) / (
                    2 * (sigma ** 2))
            error[node][c] = np.sum(neg_log_lik.reshape((sum(contexts[node] == c), -1)), axis=0)
    return error
