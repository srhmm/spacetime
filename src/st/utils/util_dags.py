import numpy as np

from cdt.metrics import SHD, SID
from st.sttypes import MethodType


def compare_adj_to_links(pooled, res_adj, true_links, method: MethodType, tau_max, enable_SID_call, lg, vb):
    return _compare_adj_to_links(False, res_adj, true_links, method, pooled, tau_max, enable_SID_call, lg, vb)


def compare_timed_adj_to_links(pooled, res_adj, true_links, method: MethodType, tau_max, enable_SID_call, lg, vb):
    return _compare_adj_to_links(True, res_adj, true_links, method, pooled, tau_max, enable_SID_call, lg, vb)


def _compare_adj_to_links(timed, res_adj, true_links, method: MethodType, pooled, tau_max, enable_SID_call, lg, vb,
                          count_self_links=True):
    N = res_adj.shape[1]

    if timed:
        assert res_adj.shape[0] == N * (tau_max + 1)
        true_adj = np.zeros((N * (tau_max + 1), N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                assert (lag <= 0 and -lag <= tau_max)
                if (not count_self_links and i != j) or (count_self_links and (not (lag == 0 and i == j))):
                    index = N * -lag + i
                    true_adj[index][j] = 1
        res = eval_dag(true_adj, res_adj, N, enable_SID_call, True, pooled)
    else:
        assert res_adj.shape[0] == N
        true_adj = np.zeros((N, N))
        for j in range(N):
            for entry in true_links[j]:
                ((i, lag), _, _) = entry
                if i != j:
                    true_adj[i][j] = 1
        res = eval_dag(true_adj, res_adj, N, enable_SID_call, False, pooled)

    name = method.get_name(timed=timed)
    suffix = '-timed' if timed else ''
    suffix = suffix + '-pooled' if pooled else suffix

    if vb > 0:
        lg.info(
            f"\t\t{name}:\t\t(f1={np.round(res['f1' + suffix], 2)}\t(shd={np.round(res['shd' + suffix], 2)}, "
            f"sid={np.round(res['sid' + suffix], 2)})\t(tp={res['tp' + suffix]}, tn={res['tn' + suffix]}, fp={res['fp' + suffix]}, fn={res['fn' + suffix]})")

    return_dict = dict()

    for name, val in res.items():
        return_dict[name] = val
    return return_dict


def directional_f1(true_dag, test_dag):
    tp = sum([sum([1 if (test_dag[i][j] != 0 and true_dag[i][j] != 0) else 0
                   for j in range(len(test_dag[i]))]) for i in range(len(test_dag))])
    tn = sum([sum([1 if (test_dag[i][j] == 0 and true_dag[i][j] == 0) else 0
                   for j in range(len(test_dag[i]))]) for i in range(len(test_dag))])
    fp = sum([sum([1 if (test_dag[i][j] != 0 and true_dag[i][j] == 0) else 0
                   for j in range(len(test_dag[i]))]) for i in range(len(test_dag))])
    fn = sum([sum([1 if (test_dag[i][j] == 0 and true_dag[i][j] != 0) else 0
                   for j in range(len(test_dag[i]))]) for i in range(len(test_dag))])
    den = tp + 1 / 2 * (fp + fn)
    f1 = tp / den if den > 0 else 1
    return f1, tp, tn, fn, fp


def eval_dag(true_dag, res_dag, N, enable_SID_call, timed, pooled):
    enable_SID_call = False
    suffix = '-timed' if timed else ''
    suffix = suffix + '-pooled' if pooled else suffix

    if timed:
        assert (res_dag.shape[1] % N == 0) and (true_dag.shape[1] % N == 0)
    else:
        assert (res_dag.shape[1] == N) and (true_dag.shape[1] == N)

    assert (true_dag.shape[1] == res_dag.shape[1])
    assert (true_dag.shape[0] == res_dag.shape[0])

    f1, tp, tn, fn, fp = directional_f1(true_dag, res_dag)

    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 1 if den == 0 else (tp * tn - fp * fn) / np.sqrt(den)

    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (tn + fp == 0) else fp / (tn + fp)
    fdr = 0 if (tp + fp == 0) else fp / (tp + fp)

    true_ones = np.array([[1 if i != 0 else 0 for i in true_dag[j]] for j in range(len(true_dag))])

    shd = SHD(true_ones, res_dag)
    if enable_SID_call and true_dag.shape[0] == true_dag.shape[1]:  # curr no sid for timed dags
        sid = np.float(SID(true_ones, res_dag))
    else:
        sid = -1  # call to R script not working on some machines
    checksum = tp + tn + fn + fp
    assert (checksum == true_dag.shape[0] * true_dag.shape[1])

    res = {
        'shd' + suffix: shd, 'sid' + suffix: sid,
        'f1' + suffix: f1, 'tpr' + suffix: tpr, 'fpr' + suffix: fpr, 'fdr' + suffix: fdr,
        'tp' + suffix: tp, 'fp' + suffix: fp, 'tn' + suffix: tn, 'fn' + suffix: fn,
        'mcc' + suffix: mcc
    }
    return res


def links_to_is_true_edge(t_links):
    """ For info during DAG search """

    def is_true_edge(parent):
        def fun(j):
            i, lag = parent
            info = ''
            if True in [t_links[j][k][0][0] == i and t_links[j][k][0][1] == -lag for k in
                        range(len(t_links[j]))]:
                info += '[caus]'
            elif i==j:
                info += '[auto]'
            elif True in [t_links[j][k][0][0] == i for k in
                          range(len(t_links[j]))]:
                info += '[caus-diff-lg]'
            elif True in [t_links[i][k][0][0] == j and t_links[i][k][0][1] == -lag for k in range(len(t_links[i]))]:
                info += '[rev]' # same lag
            elif True in [t_links[i][k][0][0] == j for k in range(len(t_links[i]))]:
                info += '[rev]' # any lag
            if len(info) == 0:
                info += '[spu]'
            return info

        return fun

    return is_true_edge