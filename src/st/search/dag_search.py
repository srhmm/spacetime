import itertools
import logging
import time
from collections import defaultdict

import numpy as np
from itertools import product

from st.dag import DAG, is_insignificant
from st.dag_time import TimeDAG
from st.utils.upq import UPQ
from st.utils.util import dag_n_edges


""" Window Causal Graph Search """


def dag_exhaustive_search(
        data_C: dict,
        max_lag: int,
        min_dur: int,
        **optargs) -> DAG:
    """ Exhaustive search for causal DAGs (proof of concept, not scalable)."""
    q = UPQ()
    dag_model = TimeDAG(data_C, max_lag, min_dur, **optargs)
    dag_model.verbosity = 0
    q = dag_model.initial_edges(q, skip_insignificant=True)
    return _dag_exhaustive_phase(q, dag_model, optargs.get("lg", None), optargs.get("verbosity", 0))


def dag_tree_search(
        data: dict,
        max_lag: int,
        min_dur: int,
        **optargs
) -> TimeDAG:
    r""" Greedy tree search for causal DAGs, adapts the GLOBE algorithm (Mian et al. 2021) for window causal DAGs.

    :param data: time series datasets, `data[i]` of shape `TxN` for `T` time points, `N` nodes, in dataset `i`.
    :param max_lag: maximum time lag
    :param min_dur: minimum regime duration
    :param optargs:

    :Keyword Arguments:
    * *scoring_function* (``TimeseriesScoringFunction``) -- scoring function, regressor type
    * *is_true_edge* (``(i: int,int) -> (j: int) -> str``) -- ground truths for logging
    * *windows* (``list``) -- regime windows
    * *partition* (``list``) -- regime partition
    * *hybrid_cps_dag_discovery* (``bool``) -- ignore windows, regime partition and discover them per each edge
    * *lg* (``logging``) -- logger if verbosity>0
    * *verbosity* (``int``) -- verbosity level
    :return:
    """
    verbosity = optargs.get("verbosity", 0)
    lg = optargs.get("lg", None)
    _info = lambda st:(lg.info(st) if lg is not None else print(st)) if verbosity > 0 else None
    _info('*** DAG Search ***')

    q = UPQ()
    dag_model = TimeDAG(data, max_lag, min_dur, **optargs)
    _info('Phase 0: Scoring Edge Pairs')

    q = dag_model.initial_edges(q)
    q, dag_model = dag_forward_phase(q, dag_model, lg, verbosity)
    q, dag_model = dag_backward_phase(q, dag_model, lg, verbosity)

    n_n = data[0].shape[1]
    _info(f'DAG search result:' + ', '.join(
        f"{i}->{j}" for i, j in set(product(set(range(n_n)), set(range(n_n)))) if dag_model.get_adj()[i][j] != 0))
    return dag_model


def dag_forward_phase(q: UPQ, dag_model: TimeDAG, lg: logging, verbosity: int) -> (UPQ, TimeDAG):
    """ Greedy tree search -- forward phase greedily adding causal edges. """
    st = time.perf_counter()
    if verbosity > 0:
        lg.info('Phase 1: Forward')

    while q.pq:
        try:
            q, dag_model = _dag_next_edge(q, dag_model, lg, verbosity)
        except KeyError:  # empty or all remaining entries are tagged as removed
            pass
    if verbosity > 0:
        lg.info(f'Forward: {np.round(time.perf_counter() - st, 2)}s ')
    return q, dag_model


def _dag_update_children(child, lag, node, q: UPQ, dag_model: TimeDAG) -> (UPQ, TimeDAG):
    """ Greedy tree search, forward phase -- reconsider a causal child of a node upon adding an edge to that node. """
    ch = (child, lag)
    if not dag_model.is_edge(node, ch):
        return q, dag_model
    gain = dag_model.eval_edge_flip(node, ch)

    if is_insignificant(gain):
        return q, dag_model
    # Remove the edge and update the gain of both edges
    dag_model.remove_edge(node, child)  # ch)
    edge_fw = dag_model.pair_edges[node][ch][lag]
    edge_bw = dag_model.pair_edges[ch][node][lag]
    assert edge_fw.i == node and edge_fw.j == ch
    assert edge_bw.i == ch and edge_bw.j == node

    assert not (q.exists_task(edge_fw))  # since this was included in the model, i.e. removed from queue at some point
    if q.exists_task(edge_bw):
        q.remove_task(edge_bw)

    gain_bw = dag_model.eval_edge_addition(edge_bw.i, edge_bw.j)
    gain_fw = dag_model.eval_edge_addition(edge_fw.i, edge_fw.j)
    q.add_task(edge_bw, gain_bw * 100)
    q.add_task(edge_fw, gain_fw * 100)

    return q, dag_model


def _dag_next_edge(q: UPQ, dag_model: TimeDAG, logger: logging, verbosity: int) -> (UPQ, TimeDAG):
    """ Greedy tree search, forward phase -- evaluate the next causal edge in the priority queue. """
    pi_edge = q.pop_task()
    node, parent = pi_edge.j, pi_edge.pa

    # Check whether adding the edge would result in a cycle
    if dag_model.has_cycle(parent, node) or dag_model.exists_anticausal_edge(parent, node):
        if verbosity > 2:
            logger.info(
                f'\tSkip cyclic edge {parent} -> {node}, existing edge(s) {dag_model.parents_of(parent)} -> {parent} \t{dag_model.is_true_edge(parent)(node)}')
        return q, dag_model

    gain, score = dag_model.eval_edge_addition(node, parent, return_score=True)

    # Check whether gain is significant
    if is_insignificant(gain):
        if verbosity > 2:
            logger.info(
                f'\tSkip insig. edge {parent} -> {node}: s={np.round(gain, 2)} pa={dag_model.parents_of(node)} \t{dag_model.is_true_edge(parent)(node)}')
        return q, dag_model

    dag_model.add_edge(parent, node, score, gain)  # want true_adj[parent][node] > 0

    # Reconsider children under current model and remove if reversing the edge improves score
    for child in dag_model.nodes:
        for lag in range(dag_model.max_lag):
            q, dag_model = _dag_update_children(child, lag, node, q, dag_model)

    # Reconsider edges Xk->Xj in q given the current model as their score changed upon adding Xi->Xj
    for other_parent in dag_model.nodes:
        for lag in range(dag_model.max_lag):
            q, dag_model = _dag_update_parents(other_parent, lag, node, parent, q, dag_model)
    return q, dag_model


def _dag_update_parents(n, lag, node, parent, q: UPQ, dag_model: TimeDAG) -> (UPQ, TimeDAG):
    """ Greedy tree search, forward phase -- reconsider a node's parent set upon adding an edge to the node. """
    other_parent = (n, lag)
    # Do not consider Xi,Xj, or current parents/children of Xi
    if node == other_parent or parent == other_parent \
            or dag_model.is_edge(other_parent, node) or dag_model.is_edge(node, other_parent):
        return q, dag_model

    edge_candidate = dag_model.pair_edges[n][node][lag]
    gain_mom = dag_model.eval_edge_addition(node, other_parent)

    if q.exists_task(edge_candidate):  # ow. insignificant /skipped
        q.remove_task(edge_candidate)
        q.add_task(edge_candidate, gain_mom * 100)
    return q, dag_model


def dag_backward_phase(q: UPQ, dag_model: TimeDAG, lg: logging, verbosity: int) -> (UPQ, TimeDAG):
    """ Greedy tree search -- backward phase refining the result. """
    st = time.perf_counter()
    if verbosity > 0:
        lg.info('Phase 2: Backward')

    for j in dag_model.nodes:
        q, dag_model = _dag_refine_parentset(j, q, dag_model, lg, verbosity)

    for j in dag_model.nodes:
        q, dag_model = _dag_refine_lags(j, q, dag_model, lg, verbosity)

    if verbosity > 0:
        lg.info(f'Backward: {np.round(time.perf_counter() - st, 2)}s ')
    return q, dag_model


def _dag_refine_parentset(j: int, q: UPQ, dag_model: TimeDAG, lg: logging, verbosity: int) -> (UPQ, TimeDAG):
    """ Greedy tree search, backward phase -- refine a node's parent set. """
    parents = dag_model.parents_of(j)
    parent_indices = np.unique([p[0] for p in parents])
    lagged_parents = [(p, lg) for p in parent_indices for lg in range(dag_model.max_lag)]

    if len(lagged_parents) <= 1:
        return q, dag_model
    max_gain, min_score, arg_max = -np.inf, np.inf, None

    # Consider all graphs G' that use a subset of the target's current parents
    min_size = 1  # variant: min_size = 0 allowed
    for k in range(min_size, len(parents) + 1):
        parent_sets = itertools.combinations(parents, k)
        for parent_set in parent_sets:
            gain, new_score = dag_model.eval_edges(j, parent_set)
            if gain > max_gain:
                min_score, max_gain, arg_max = new_score, gain, parent_set
    if (arg_max is not None) and (not is_insignificant(max_gain)):
        if verbosity > 1:
            lg.info(f'\tupdating {parents} to {arg_max} -> {j}')
        dag_model.update_edges(j, arg_max)
    return q, dag_model

def _dag_refine_lags(j, q: UPQ, dag_search: TimeDAG, logger: logging, verbosity: int) -> (UPQ, TimeDAG):
    """ Greedy tree search, backward phase -- refine a node's parents' time lags. """
    # todo clean up duplications.
    parents = dag_search.parents_of(j)
    for i in dag_search.nodes:
        if i == j or not True in [dag_search.is_edge((i, lag), j) for lag in range(dag_search.max_lag)]:
            return q, dag_search

        best_lag = 0
        best_s = np.inf
        rng = range(1, dag_search.max_lag) if i == j else range(dag_search.max_lag)
        for lag in rng:
            other_parents = [(p, lg) for (p, lg) in dag_search.parents_of(j) if p != i]
            other_parents = other_parents.copy()
            other_parents.append((i, lag))
            s = dag_search.eval_edge(j, other_parents)
            if s < best_s:
                best_s = s
                best_lag = lag

        for lag in range(dag_search.max_lag):
            if (i, lag) in dag_search.parents_of(j) and lag != best_lag:
                dag_search.remove_edge((i, lag), j)

        for lag in range(dag_search.max_lag):
            if lag == best_lag:
                if (i, lag) not in dag_search.parents_of(j):
                    other_parents = [(p, lg) for (p, lg) in dag_search.parents_of(j) if p != i]
                    other_parents = other_parents.copy()
                    other_parents.append((i, lag))
                    s = dag_search.eval_edge(j, other_parents)
                    dag_search.add_edge((i, lag), j, s, [[0]])
    for j in dag_search.nodes:
        i = j
        best_lag = 1
        best_s = np.inf
        rng = range(1, dag_search.max_lag)
        for lag in rng:
            other_parents = [(p, lg) for (p, lg) in dag_search.parents_of(j) if p != i]
            other_parents = other_parents.copy()
            other_parents.append((i, lag))
            s = dag_search.eval_edge(j, other_parents)
            if s < best_s:
                best_s = s
                best_lag = lag

        for lag in range(dag_search.max_lag):
            if (i, lag) in dag_search.parents_of(j) and lag != best_lag:
                dag_search.remove_edge((i, lag), j)

        for lag in range(dag_search.max_lag):
            if lag == best_lag:
                if (i, lag) not in dag_search.parents_of(j):
                    other_parents = [(p, lg) for (p, lg) in dag_search.parents_of(j) if p != i]
                    other_parents = other_parents.copy()
                    other_parents.append((i, lag))
                    s = dag_search.eval_edge(j, other_parents)
                    dag_search.add_edge((i, lag), j, s, [[0]])
    return q, dag_search


def _dag_exhaustive_phase(q: UPQ, dag_model: DAG, lg: logging, verbosity: int) -> DAG:
    st = time.perf_counter()
    if verbosity > 0:
        lg.info('Exhaustive Phase ...')
    test_dags = _exhaustive_gen_tree_dags(dag_model, q)

    result = _score_dag_candidates(
        dag_model, test_dags, verbosity)

    if verbosity > 0:
        lg.info(f'Exhaustive: {np.round(time.perf_counter() - st, 2)}s ')
    return result


def _score_dag_candidates(
        dag_model: DAG,
        test_dags, verbosity=0, matching=lambda test_dag: False) -> DAG:
    """
    Scores candidate causal DAGs and returns the lowest (best) scoring one
    """

    results = dict()
    results_per_dag_size = defaultdict(dict)

    for i, test_dag in enumerate(test_dags):

        dag_score = dag_model.eval_other_dag(test_dag)

        n_edges = dag_n_edges(test_dag)
        if verbosity > 0:
            s = ''
            if matching(test_dag):
                s = '<- true'
            print(f'\t\tDAG {i + 1}/{len(test_dags)}: s={np.round(dag_score, 2)}\t|G|={n_edges}\t {s}')

        results[f'{i}'] = (dag_score, test_dag)
        results_per_dag_size[n_edges][f'{i}'] = (dag_score, test_dag)
    best_dag: DAG = results[sorted(results, key=lambda res: res[0])[0]][1]
    return best_dag


def _exhaustive_gen_tree_dags(dag_model: DAG, q: UPQ, max_depth=np.inf) -> list:
    """ Generates the search space over causal trees for a set of nodes in dag_model.

    :param dag_model:
    :param q:
    :return:
    """
    test_dags = []
    edges = q.all_entries
    depth = 0
    for i in range(1, len(edges) + 1):
        if depth > max_depth:
            break
        edge_combos = itertools.combinations(edges.values(), i)
        for combo in edge_combos:
            if depth > max_depth:
                break
            for edge in combo:
                node, parent = edge.j, edge.pa
                # Check whether adding the edge would result in a cycle
                if dag_model.has_cycle(parent, node):
                    continue
                gain, score = dag_model.eval_edge_addition(node, parent, return_score=True)

                # Optional to speed things up: Check whether gain is significant
                if is_insignificant(gain):
                    continue

                dag_model.add_edge(parent, node, score, gain)  # want true_adj[parent][node] > 0
            test_dags.append(dag_model.get_adj())
            dag_model.remove_all_edges()
            depth += 1
    return test_dags
