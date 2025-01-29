import logging
from enum import Enum
from itertools import product, combinations
from types import SimpleNamespace

import numpy as np

from exp.utils.context_model_with_regimes import ContextModelTSWithRegimes
from st.utils.util_regimes import partition_t, r_partition_to_windows_T


class Func(Enum):
    """ functional forms """
    LIN = "lin"
    NLIN = "nlin"
    TANH = "tanh"
    SINC = "sinc"

    @staticmethod
    def to_func(fun):
        functions = {
            Func.LIN: lambda x, b, regime: b * x,
            Func.NLIN: lambda x, b, regime: b * x + 5. * x ** 2 * np.exp(-x ** 2 / 20.),
            Func.TANH: lambda x, b, regime: b * np.tanh(x),
            Func.SINC: lambda x, b, regime: b * np.sinc(x),
        }
        return [(k, v) for k, v in functions.items() if k == fun][0]


class Noise(Enum):
    """ additive noise types """
    GAUSS = "gauss"
    UNIFORM = "unif"


def gen_timeseries(
        true_tau_max: int,
        true_min_dur: int,
        rstate = None,
        seed: int = 0,
        _depth=100,
        **kwargs):
    r""" Generates time series datasets in multiple contexts and regimes.

    :param seed: seed
    :param true_tau_max: max time lag
    :param true_min_dur: min regime duration
    :param _depth: recursion level if invalid data
    :Keyword Arguments:
    * *N* (``int``) -- n nodes
    * *T* (``int``) -- n time points
    * *D* (``int``) -- n datasets
    * *CPS* (``int``) -- n cutpoints over T
    * *R* (``int``) -- n regimes over T
    * *C* (``int``) -- n contexts over D
    * *I* (``int``) -- n intervened nodes
    * *IVS* (``int``) -- intervention strength
    * *NS* (``Noise``) -- additive noise
    * *F* (``Func``) -- functional form
    :return:
    """
    random_state = rstate if rstate is not None else np.random.default_rng(seed)

    lg: logging = kwargs.get("lg", None)
    vb = kwargs.get("vb", 0)
    hard_intervention = kwargs.get("hard_intervention", False)
    regime_drift = kwargs.get("regime_drift", False)
    #nonlinear_fun = (Func.NLIN, lambda x, coeff, regime: coeff * x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))


    # Parameters and their defaults
    params = {
        "N": kwargs.get("N", 5),
        "CPS": kwargs.get("CPS", 3),
        "R": kwargs.get("R", 2),
        "T": kwargs.get("T", 500),
        "C": kwargs.get("C", 2),
        "D": kwargs.get("D", 1),
        "I": kwargs.get("I", 2),
        "IVS": kwargs.get("IVS", 0.5),
        "NS": kwargs.get("NS", Noise.GAUSS),
        "F": kwargs.get("F", Func.NLIN),
    }
    params["F"] = Func.to_func(params["F"])
    nb_of_chunks = params['CPS'] + 1  # random_state.integers(params['R'], params['T'] / options.true_min_dur)
    assert nb_of_chunks in set(range(params['R'], int(np.floor(params['T'] / true_min_dur)) + 1))
    equal_dur = random_state.choice([True, False])
    nb_edges = random_state.integers(1, params['N'])  # number of (lagged) links between different variables
    nb_all_arcs = nb_edges + params['N']  # number of links between variables and self links

    params['IC'], params['IR'] = int(np.floor(params['I'] * nb_all_arcs)), int(np.floor(params['I'] * nb_all_arcs))
    skip = true_tau_max + 1  # nn of datapoints to skip at the beginning of a regime
    if vb > 0:
        lg.info(f"Data Gen: Nb chunks {str(nb_of_chunks)}, nb edges {nb_all_arcs}, nb intervened {params['IC']}")

    ### MODEL GENERATION
    weights, targets_c, targets_r = gen_dag(
        params, nb_edges, params['IC'], params['IR'], random_state, true_tau_max, hard_intervention)
    regimes_partition = partition_t(params['T'], params['R'], nb_of_chunks, true_min_dur, equal_dur)
    windows_T = r_partition_to_windows_T(regimes_partition, skip)
    truth = [r for (b, d, r) in regimes_partition]

    ### DATA GENERATION WITH TIGRAMITE
    cnode = params['N']
    snode = params['N'] + 1
    is_uniform_noise = params['NS'].value == Noise.UNIFORM.value

    node_classification = {
        cnode: 'time_context',
        snode: 'space_context'
    }
    for n in range(params['N']): node_classification[n] = 'system'
    noises = [random_state.standard_normal for _ in range(params['N'] + 2)] if not is_uniform_noise \
        else [lambda s: random_state.uniform(size=s) for _ in
              range(params['N'] + 2)]  # same for all regimes and contexts

    # added this for the case of hard interventions where the links in each (c,r) could be different
    all_arcs = weights[(0, 0)].arcs
    [all_arcs.update(weights[(r, c)].arcs) for c in range(params['C']) for r in range(params['R'])]

    func = dict.fromkeys(all_arcs)
    for k in func.keys():
        func[k] = params['F']

    datasets = dict()
    datasets_known_regimes = dict()
    datasets_without_dummynodes = dict()
    datasets_without_dummynodes_known_regimes = dict()

    cpt = 0
    cpt2 = 0
    invalid_data = False
    data = dict.fromkeys(set(product(set(range(params['C'])), set(range(params['D'])))))
    data_known_regimes = dict()  # .fromkeys(set(product(set(range(params['C'])), set(range(params['D'])))))

    true_links = {i: [] for i in range(params['N'])}
    for c in range(params['C']):
        links = dict.fromkeys(range(params['R']))
        for r in range(params['R']):
            dag = weights[(r, c)]
            links[r] = gen_links_from_lagged_dag(dag, params['N'], random_state, func)

        tr_links = {i: links[0][i] for i in range(params['N'])}
        # add link if it exists in any regime
        if hard_intervention:
            for r in range(params['R']):
                for i in range(params['N']):
                    for lnk in links[r][i]:
                        duplicate = False
                        for lnk2 in true_links[i]:
                            if lnk2[0][0] == lnk[0][0] and lnk2[0][1] == lnk[0][1]:
                                duplicate = True
                        if not duplicate:
                            true_links[i].append(lnk)
        else:
            true_links = tr_links

        n_drift = 0.1 * true_min_dur if regime_drift else 0
        contextmodel = ContextModelTSWithRegimes(
            links_regimes=links, node_classification=node_classification, noises=noises, seed=seed)
        data_ens, nonstationary = contextmodel.generate_data_with_regimes(
            params['D'], params['T'], regimes_partition, n_drift, is_uniform_noise)
        if nonstationary: invalid_data = True
        for d in range(params['D']):
            data[(c, d)] = data_ens[d]
            datasets[cpt] = data[(c, d)]
            datasets_without_dummynodes[cpt] = data[(c, d)][:, :params['N']]
            cpt += 1
            for r, (st, end) in enumerate(windows_T):
                data_known_regimes[(c, r, d)] = data_ens[d][st:end, :]
                datasets_known_regimes[cpt2] = data_known_regimes[(c, r, d)]
                datasets_without_dummynodes_known_regimes[cpt2] = data_known_regimes[(c, r, d)][:, :params['N']]
                cpt2 += 1
    if invalid_data:
        return gen_timeseries(true_tau_max, true_min_dur, seed + 1, _depth - 1, **kwargs)

    for ky in data:
        assert (data[ky].shape[1] == params['N'] + 2)

    # DATA
    data_summary = SimpleNamespace(
        data=data,
        data_known_regimes=data_known_regimes,
        datasets_with_dummynodes=datasets,
        datasets_with_dummynodes_known_regimes=datasets_known_regimes,
        datasets=datasets_without_dummynodes,
        datasets_known_regimes=datasets_without_dummynodes_known_regimes,
        cnode=cnode,
        snode=snode,
        node_classification=node_classification
    )
    # GROUND TRUTH
    truths = SimpleNamespace(
        windows_T=windows_T,
        true_links=true_links,
        is_true_edge=links_to_is_true_edge(tr_links),
        true_r_partition=truth,
        regime_mask=get_regime_mask(datasets_without_dummynodes[0], regimes_partition),
        regimes_partition=regimes_partition,
        skip=skip
    )
    return data_summary, truths

def get_regime_mask(dataset, regimes_partition):
    T, N = dataset.shape
    regime_mask = np.zeros(dataset.shape)
    for t in range(1, T):
        for st, nd, r in regimes_partition:
            if st <= t <= nd + st:
                regime_mask[[t, t - 1]] = r
                break
    return regime_mask

def gen_links_from_lagged_dag(dag, N, random_state, funs):
    """
    Converts DAG with lagged effects to link list (as used in tigramite data generation)

    :param dag: true DAG including self transitions
    :param N: n nodes
    :param random_state: rand
    :param funs: list of possible functional forms for each causal relationship
    """

    # TODO: each variable should have only one list (which is a problem in case of DAG ><><>< like (Jilles example))
    links = dict()
    cnode = dag.nodes[-2]
    snode = dag.nodes[-1]

    for i in range(N):
        links[i] = []
        # fun_pa = random_state.choice(funs, size=len(dag.parents_of(i)))

        # Add causal parents in dag
        for index_j, j in enumerate(dag.parents_of(i)):
            if j != cnode and j != snode:  # Skip context & spatial links
                lag = int(i // N - j // N)
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = j + lag * N
            else:  # context & spatial links
                lag = 0
                w = dag.weight_mat[j][i]
                assert (w != 0)
                j_t = N + (j % N) - 1
            links[i].append(((j_t, lag), w, funs[(j, i)][1]))

    links[N] = []  # context node
    links[N + 1] = []  # spatial node
    return links


def gen_dag(params, nb_edges, intervention_nb, intervention_nb_regimes, random_state,
            true_tau_max, hard_intervention):
    """ Samples a causal dag and weights.

    :param params: see keyword args
    :param nb_edges: number of edges
    :param intervention_nb: number of intervened nodes across space
    :param intervention_nb_regimes:  number of intervened nodes across time
    :param random_state:
    :param true_tau_max: max lag
    :param hard_intervention: set intervened weigts to zero
    :Keyword Arguments:
    * *N* (``int``) -- n nodes
    * *T* (``int``) -- n time points
    * *D* (``int``) -- n datasets
    * *CPS* (``int``) -- n cutpoints over T
    * *R* (``int``) -- n regimes over T
    * *C* (``int``) -- n contexts over D
    * *I* (``int``) -- n intervened nodes
    * *IVS* (``int``) -- intervention strength
    * *NS* (``Noise``) -- additive noise
    * *F* (``(Func, lambda x.)``) -- functional form
    :return:  weights, intervention_targets, intervention_targets_regimes
    """
    import causaldag as cd
    strength = params['IVS']

    ## Define DAG structure
    arcs = cd.rand.directed_erdos(((true_tau_max + 1) * params['N']) + 2, 0)
    pairs = random_state.choice(list(combinations(range(params['N']), 2)), nb_edges, replace=False)
    pairs = [(j, i) if random_state.random() > .5 else (i, j) for (i, j) in pairs]
    lags = random_state.choice(range(true_tau_max + 1), nb_edges)
    for i in range(nb_edges):
        arcs = add_edge(pairs[i][0], pairs[i][1], -lags[i], params['N'], arcs)

    for i in range(params['N']):
        # self links
        arcs = add_edge(i, i, -1, params['N'], arcs)

    ## For each context define intervention targets
    intervention_targets = dict.fromkeys(set(range(params['C'])))
    for c in intervention_targets.keys():
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets[c] = random_state.choice(list(arcs.arcs), size=intervention_nb,
                                                      replace=False)  # TODO: assert different from a context to another?
        intervention_targets[c] = list(tuple(l) for l in intervention_targets[c])

    ## For each regime define intervention targets
    intervention_targets_regimes = dict.fromkeys(set(range(params['R'])))
    for r in intervention_targets_regimes.keys():
        nb = intervention_nb_regimes  # random_state.integers(1, params['N']+1)
        # interventions shouldn't be on edges from spatial or context variable
        intervention_targets_regimes[r] = random_state.choice(list(arcs.arcs), size=nb,
                                                              replace=False)  # TODO: assert different from a context to another?
        intervention_targets_regimes[r] = list(tuple(l) for l in intervention_targets_regimes[r])

    ## For each regime, define general weights and special weights for intervention tagets in contexts
    weights = dict.fromkeys(set(product(set(range(params['R'])), set(range(params['C'])))))  # key = (regime, context)

    initial_weights = cd.rand.rand_weights(arcs)

    from graphical_models.rand import unif_away_zero
    c_weights = {c: {t: unif_away_zero()[0] for t in intervention_targets[c]} for c in intervention_targets.keys()}
    for r in range(params['R']):
        r_weights = dict()
        for t in intervention_targets_regimes[r]:
            if hard_intervention:
                r_weights[t] = 0
            else:
                w = unif_away_zero()[0]
                while abs(w - initial_weights.arc_weights[t]) < strength:
                    w = unif_away_zero()[0]
                r_weights[t] = w
        # r_weights = {t: unif_away_zero()[0] for t in intervention_targets_regimes[r]}
        for c in range(params['C']):
            weights[(r, c)] = cd.rand.rand_weights(arcs)
            for arc in initial_weights.arcs:
                if arc in intervention_targets_regimes[r] and arc in intervention_targets[c]:
                    if hard_intervention:
                        w = 0
                    else:
                        w = unif_away_zero()[0]
                        while abs(w - initial_weights.arc_weights[arc]) < strength and abs(
                                w - r_weights[arc]) < strength:
                            w = unif_away_zero()[0]
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], w)
                elif arc not in intervention_targets_regimes[r] and arc in intervention_targets[c]:
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], c_weights[c][arc])
                elif arc not in intervention_targets_regimes[r] and arc not in intervention_targets[c]:
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], initial_weights.arc_weights[arc])
                else:
                    assert (arc in intervention_targets_regimes[r] and arc not in intervention_targets[c])
                    weights[(r, c)].set_arc_weight(arc[0], arc[1], r_weights[arc])

    return weights, intervention_targets, intervention_targets_regimes


def add_edge(node_i_t, node_j_t, lag, N, arcs):
    """
    Add arc to the DAG from a parent to the child at time t

    :param node_i_t: parent node name at time t
    :param node_j_t: child node name at time t
    :param lag: lag (negative value)
    :param N: number of variables
    :param arcs: DAG
    :return: DAG
    """
    arcs.add_arc(node_i_t + (-lag * N), node_j_t)
    return arcs


def links_to_is_true_edge(rlinks):
    """ For info during DAG search """

    def is_true_edge(parent):
        def fun(j):
            i, lag = parent
            info = ''
            if True in [rlinks[j][k][0][0] == i and rlinks[j][k][0][1] == -lag for k in
                        range(len(rlinks[j]))]:
                info += '[caus]'
            elif True in [rlinks[j][k][0][0] == i for k in
                          range(len(rlinks[j]))]:
                info += '[caus-any-lg]'
            if True in [rlinks[i][k][0][0] == j and rlinks[i][k][0][1] == -lag for k in range(len(rlinks[i]))]:
                info += '[rev-same-lg]'
            elif True in [rlinks[i][k][0][0] == j for k in range(len(rlinks[i]))]:
                info += '[rev-any-lag]'
            if len(info) == 0:
                info += '[spu]'
            return info

        return fun

    return is_true_edge
