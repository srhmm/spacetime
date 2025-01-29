from itertools import product

import numpy as np

from st.dag import DAG
from st.sttypes import TimeseriesScoringFunction, SubsamplingApproach
from st.scoring.discrepancy_testing import DiscrepancyTestType
from st.scoring.time_space_scoring import hybrid_score_edge_time_space, score_edge_time_space
from st.utils.util import is_insignificant
from st.utils.upq import UPQ


class TimeDAG(DAG):
    def __init__(self, data: dict, max_lag: int, min_dur: int, **optargs):
        r""" Timed DAG model.


        :param data: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
        :param max_lag: maximum time lag, window size of the resulting DAG.
        :param min_dur: minimum regime duration
        :param optargs: optional arguments

        :Keyword Arguments:
        * *scoring_function* (``TimeseriesScoringFunction``) -- scoring function, regressor type
        * *subsampling_approach* (``SubsamplingApproach``) -- approach to collect subsamples for time series regression
        * *discrepancy_test_type* (``DiscrepancyTestType``) -- test whether regime chunks have the same gen. process
        * *is_true_edge* (``(i: int,int) -> (j: int) -> str``) -- ground truths for logging
        * *windows* (``list``) -- regime windows
        * *partition* (``list``) -- regime partition
        * *hybrid_cps_dag_discovery* (``bool``) -- ignore windows, regime partition and discover them per each edge
        * *lg* (``logging``) -- logger if verbosity>0
        * *verbosity* (``int``) -- verbosity level
        """
        super().__init__(data, **optargs)
        self.min_dur = min_dur
        self.max_lag = max_lag + 1  # using range for lags within TimeDag
        self.pair_edges = [[[None for _ in range(self.max_lag)] for _ in range(self.n_n)] for _ in range(self.n_n)]

        self.scoring_function = optargs.get("scoring_function", TimeseriesScoringFunction.GP)
        self.subsampling_approach = optargs.get("subsampling_approach", SubsamplingApproach.HORIZONTAL)
        self.discrepancy_test_type = optargs.get("discrepancy_test_type", DiscrepancyTestType.SKIP)
        self.partition = optargs.get("partition", None)
        self.windows = optargs.get("windows", None)
        self.hybrid_cps_dag_discovery = optargs.get("hybrid_cps_dag_discovery", False)

    def children_of(self, j):
        return [(i, lag) for lag in range(self.max_lag) for i in self.nodes if self.is_edge(j, (i, lag))]

    def parents_of(self, j):
        return [(i, lag) for lag in range(self.max_lag) for i in self.nodes if self.is_edge((i, lag), j)]

    def eval_edge(self, j: int, pa: list) -> int:
        """
        Evaluates scoring function for a causal relationship ``pa(Xj)->Xj``.

        :param j: node index of ``Xj`` (time lag suppressed, zero by convention)
        :param pa: parent nodes of ``Xj``; list of tuples ``(i, l)`` where ``i`` is the node index, ``l`` the time lag
        :return: ``score(pa(Xj)->Xj)``
        """
        return self.eval_edge_time(j, pa, return_scores_each_regime=False)

    def eval_edge_time(self, j: int, pa: list, return_scores_each_regime=False):
        """
        Evaluates scoring function for a causal relationship ``pa(Xj)->Xj`` in multi-context multi-regime time series.

        :param j: node index of ``Xj`` (time lag suppressed, zero by convention)
        :param pa: parent nodes of ``Xj``; list of tuples ``(i, l)`` where ``i`` is the node index, ``l`` the time lag
        :param return_scores_each_regime: return score for each regime-context chunk, otherwise sum.
        :return: ``score_cr(pa(Xj)->Xj)`` for each context ``c`` and regime ``r``
        """
        hash_key = f'j_{str(j)}_pa_{str(pa)}'
        info = "&". join([f"{self.is_true_edge(i)(j)}" for i in pa])

        if self.mdl_cache.__contains__(hash_key):
            scores_up = self.mdl_cache[hash_key]
            return scores_up

        # discover the regime windows adaptively for each edge
        if self.hybrid_cps_dag_discovery:
            scores_up, windows_T = hybrid_score_edge_time_space(
                self.data_C, covariates=pa, target=j, max_lag=self.max_lag, min_dur=self.min_dur,
                scoring_function=self.scoring_function, discrepancy_test_type=self.discrepancy_test_type,
                lg=self.lg, verbosity=self.verbosity - 1, edge_info=info)

        # use the regime windows provided as input
        else:
            windows_T = self.windows
            scores_up = score_edge_time_space(
                self.data_C, covariates=pa, target=j, windows_T=windows_T, regimes_R=self.partition,
                scoring_function=self.scoring_function, subsampling_approach=self.subsampling_approach,
                discrepancy_test_type=self.discrepancy_test_type,
                max_lag=self.max_lag, min_dur=self.min_dur,
                lg=self.lg, verbosity=self.verbosity - 1, edge_info=info)
        if not return_scores_each_regime:
            scores_up = float(sum(sum(np.array(scores_up))))
        self.mdl_cache[hash_key] = scores_up   # store for later use
        self.cps_cache[hash_key] = windows_T  # store for debug
        return scores_up

    """ Util """

    def get_links(self):
        """ Returns the timed causal links."""
        n_n = self.n_n
        all_parents = list(
            product(list(range(n_n)), list(range(0, -self.max_lag, -1))))

        causal_links = {
            0: {n: [((p, d), 1, None) for p, d in all_parents if
                    self.is_edge((p, -d), n) or (p == n and d == -1)]
                for n in range(n_n)}}
        return causal_links

    def get_timed_adj(self):
        adj = np.array([np.zeros(len(self.nodes)) for _ in range(len(self.nodes) * self.max_lag)])
        for i in self.nodes:
            for j in self.nodes:
                for lag in range(self.max_lag):
                    if self.is_edge((i, lag), j):
                        index = self.n_n * lag + i
                        adj[index][j] = 1
        return adj

    def get_timed_weights(self, c, r):
        adj = np.array([np.zeros(len(self.nodes)) for _ in range(len(self.nodes) * self.max_lag)])
        for i in self.nodes:
            for j in self.nodes:
                for lag in range(self.max_lag):
                    if self.is_edge((i, lag), j):
                        index = self.n_n * lag + i
                        adj[index][j] = self.eval_edge_time(j, [(i, lag)], return_scores_each_regime=True)[c][r][0][0]
        return adj

    def get_adj(self):
        adj = np.array([np.zeros(len(self.nodes)) for _ in range(len(self.nodes))])
        for i in self.nodes:
            for j in self.nodes:
                for lag in range(self.max_lag):
                    if self.is_edge((i, lag), j) and i != j:
                        adj[i][j] = 1
        return adj

    def get_weights(self, c, r):
        adj = np.array([np.zeros(len(self.nodes)) for _ in range(len(self.nodes))])
        for i in self.nodes:
            for j in self.nodes:
                for lag in range(self.max_lag):
                    if self.is_edge((i, lag), j) and i != j:
                        adj[i][j] = self.eval_edge_time(j, [(i, lag)], return_scores_each_regime=True)[c][r][0][0]
        return adj

    def exists_anticausal_edge(self, parent, node):
        for lag in range(self.max_lag):
            if self.is_edge((node, lag), parent[0]):
                return True
        return False

    def eval_other_dag(self, adj):
        """ Evaluate the MDL score for a given DAG *(regardless the edges in this DAG object).*"""
        mdl = 0
        for j in self.nodes:
            pa = [(i, 0) for i in self.nodes if adj[i][j] != 0] #lag zero here
            score_j = self.eval_edge(j, pa)
            mdl = mdl + score_j
        return mdl

    def initial_edges(self, q: UPQ, skip_insignificant=False) -> UPQ:
        for j in self.nodes:
            pa = []
            score = self.eval_edge(j, pa)
            # others = [i for i in self._nodes if not (i == j)]
            others = [i for i in self.nodes]
            for i in others:
                for lag in range(self.max_lag):
                    if i == j and lag == 0:
                        continue
                    score_ij = self.eval_edge(j, [(i, lag)])

                    edge_ij = QTimeEntry(i, j, lag, score_ij, score)

                    gain = score - score_ij
                    prio = -gain * 100
                    if (not skip_insignificant) or (not is_insignificant(gain)):
                        q.add_task(task=edge_ij, priority=prio)
                    self.pair_edges[i][j][lag] = edge_ij
        return q


class QTimeEntry:
    def __init__(self, i, j, t, score_ij, score_0):
        self.i = i
        self.j = j
        self.t = t
        self.pa = (i, t)

        # Score of edge i_t->j in the empty graph
        self.score_ij = score_ij
        # Score of []->j
        self.score_0 = score_0

    def __hash__(self):
        return hash((self.i, self.j, self.t))

    def __eq__(self, other):
        return (self.i == other.i
                & self.j == other.j
                & self.t == other.t)

    def __str__(self):
        return f'j_{str(self.i)}_i_{str(self.j)}_t_{str(self.t)}'
