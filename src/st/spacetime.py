import csv
import logging
import numpy as np
import pandas as pd

from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import List

from st.search.cps_search import mdl_error_cp_search, partition_regimes, partition_contexts_over_regimes
from st.search.dag_search import dag_tree_search, dag_exhaustive_search
from st.dag_time import TimeDAG
from st.scoring.discrepancy_testing import DiscrepancyTestType
from st.sttypes import CpsSearchStrategy, DAGSearchSpace, MethodType, CpsInitializationStrategy, \
    TimeseriesScoringFunction, SubsamplingApproach
from st.utils.util_dags import compare_adj_to_links, compare_timed_adj_to_links, links_to_is_true_edge
from st.utils.util_regimes import partition_t, r_partition_to_windows_T, windows_T_to_r_partition, \
    regimes_map_from_constraints
from st.utils.util_regimes import precision_recall_dist_cps, ari, nmi, safe_div


class STResult:
    """ Results of SPACETIME. see ``get_regime_changepoints``, ``get_regime_partition`` and ``get_causal_links``"""
    initial_links: dict
    initial_regimes: List
    prev_adj: np.array
    prev_regimes: List
    # Results
    cur_links: dict
    cur_regimes: List
    cur_contexts: List
    cur_dag_model: TimeDAG
    cur_c_each_node: List
    cur_r_each_node: List
    # Per interleaving iteration
    iters_metrics: List = []
    iters_adj: List = []
    iters_regimes: List = []
    iters_contexts: List = []

    def get_regime_changepoints(self, max_lag: int) -> List:
        return r_partition_to_windows_T(self.cur_regimes, max_lag)

    def get_partitions(self):
        return self.cur_regimes, self.cur_contexts

    def get_causal_links(self) -> dict:
        return self.cur_links


class SpaceTime:
    scoring_function: TimeseriesScoringFunction
    subsampling_approach: SubsamplingApproach
    cps_init_strategy: CpsInitializationStrategy
    method_type: MethodType
    interleave_its: int
    cps_convergence_tolerance: int
    hybrid_cps_dag_discovery: bool
    truths: SimpleNamespace
    results: STResult
    logger: logging
    verbosity: int
    to_file: bool
    cur_it: int
    out: str

    def __init__(
            self,
            max_lag: int,
            min_dur: int,
            **optargs):
        r""" SpaceTime (ST) algorithm to discover causal links and regime changepoints from multi-context time series.

        :param max_lag: maximum time lag
        :param min_dur: minimum regime duration
        :param optargs: optional arguments

        :Keyword Arguments:
        * *scoring_function* (``TimeseriesScoringFunction``) -- scoring function, regressor type
        * *subsampling_approach* (``SubsamplingApproach``) -- approach to subsample for regression in time series
        * *cps_init_strategy* (``CpsInitializationStrategy``) -- method to discover initial regime changepoints
        * *method_type* (``MethodType``) -- method, different oracles
        * *truths* (``dict``) -- ground truths for oracles and logging
        * *interleave_its* (``int``) -- num iterations to interleave DAG and changepoint search
        * *cps_convergence_tolerance* (``int``) -- threshold for considering cutpoints converged
        * *hybrid_cps_dag_discovery* (``bool``) -- variant of ST that combines DAG and changepoint search
        * *logger* (``logging``) -- logger if verbosity>0
        * *verbosity* (``int``) -- verbosity level
        * *to_file* (``int``) -- store outputs
        """
        self.defaultargs = {
            "scoring_function": TimeseriesScoringFunction.GP,
            "subsampling_approach": SubsamplingApproach.HORIZONTAL,
            "method_type": MethodType.ST_GP,
            "truths": SimpleNamespace(),
            "cps_init_strategy": MethodType.ST_GP.get_cps_init_strategy(),
            "interleave_its": 10,
            "cps_convergence_tolerance": 3,
            "hybrid_cps_dag_discovery": MethodType.ST_GP.is_hybrid(),
            "logger": None, "out": "", "verbosity": 0, "to_file": False}

        assert all([arg in self.defaultargs.keys() for arg in optargs.keys()])
        self.__dict__.update((k, v) for k, v in self.defaultargs.items() if k not in optargs.keys())
        self.__dict__.update((k, v) for k, v in optargs.items() if k in self.defaultargs.keys())

        # parameters
        self.max_lag = max_lag
        self.min_dur = min_dur
        self.search_space = self.method_type.get_search_space()
        self.cps_strategy = self.method_type.get_cps_strategy()
        self.interleave_its = 1 if self.cps_strategy == CpsSearchStrategy.SKIP else self.interleave_its

        # outputs
        if self.out is not None:
            Path(self.out).mkdir(parents=True, exist_ok=True)
        # results
        self.result = STResult()
        self.n_it_completed, self.cur_it, self.converged = 0, 0, False

    """Main method"""

    def run(self, data: dict):
        """ Run SpaceTime to discover a window causal graph, changepoints, and regime and context partitions

        :param data: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
        :return: see self.result.
        """
        self._info = lambda st: (self.logger.info(st) if self.logger is not None else print(st)) if self.verbosity > 0 else None
        self._info(f'*** SpaceTime ({self.method_type.value})***')

        # Checks
        (n_t, n_n), n_d = data[0].shape, len(data)
        assert all([data[i].shape == (n_t, n_n) for i in range(len(data))]) and n_n < n_t

        # Init regimes and causal links
        self._set_initial_regimes(data)
        self._set_initial_links(n_n)
        self._set_initial_results(n_n)
        _ = self.get_metrics()

        # Interleave DAG search and CPS search
        for self.cur_it in range(self.interleave_its):
            self.update(data)
            self.n_it_completed += 1
            metrics = self.get_metrics()
            self.result.iters_metrics.append(metrics)

            if self.hybrid_cps_dag_discovery or self.converged:
                break
        self._info(f'Completed run in {self.cur_it} iterations. ')
    def update(self, data: dict):
        """ Update step """
        self._info(f'Hybrid') if self.hybrid_cps_dag_discovery else \
                self._info(f'Interleaving Iteration {self.cur_it + 1}/max. {self.interleave_its}*')
        self.dag_search(data)
        self.cps_search(data)

    """Main Components"""

    def dag_search(self, data: dict):
        """ ST-DAG: Discover a window causal graph, given the current changepoints.
         *(If hybrid, discover DAG together with refitting the regime changepoints.)*

        :param data: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
        :return: *see self.result.cur_links for causal links per node; and self.result.cur_dag_search_result for TimeDAG object.*
        """

        # Oracle version: skip if the true causal links known
        if self.search_space == DAGSearchSpace.SKIP:
            return

        # Set hyperparameters
        is_hybrid = \
            self.hybrid_cps_dag_discovery or (
                    self.cps_init_strategy == CpsInitializationStrategy.HYBRID and self.cur_it == 0)
        scoring_parameters = dict(
            scoring_function=self.scoring_function,
            subsampling_approach=self.subsampling_approach,
            hybrid_cps_dag_discovery=is_hybrid,
            is_true_edge=links_to_is_true_edge(self.truths.true_links), #self.truths.is_true_edge, #
            windows=None if is_hybrid else r_partition_to_windows_T(self.result.cur_regimes, self.max_lag),
            partition=None if is_hybrid else self.result.cur_regimes,
            verbosity=self.verbosity, lg=self.logger
        )
        if self.verbosity > 0:
            self._info_dag_search(scoring_parameters)

        # DAG SEARCH
        if self.search_space == DAGSearchSpace.EXHAUSTIVE_SEARCH:
            dag_model = dag_exhaustive_search(
                data, self.max_lag, self.min_dur, **scoring_parameters)
        elif self.search_space == DAGSearchSpace.GREEDY_TREE_SEARCH:
            dag_model = dag_tree_search(
                data, self.max_lag, self.min_dur, **scoring_parameters)
        else:
            raise ValueError(self.search_space)

        self._update_dag(dag_model, data)

    def _update_dag(self, dag_model, data):
        """ Update and check for convergence """
        self.result.cur_dag_model = dag_model
        self.result.cur_links = dag_model.get_links()
        cur_adj = dag_model.get_adj()

        self.converged = self.adj_equals(self.result.prev_adj, cur_adj) or \
            any([self.adj_equals(seen_adj, cur_adj) for seen_adj in self.result.iters_adj])

        if self.converged and self.verbosity > 0:
            self.logger.info(f'DAG search converged!')
        self.result.iters_adj.append(cur_adj)
        self.result.prev_adj = cur_adj

        if self.to_file:
            self._dag_search_to_file(data)

    def cps_search(self, data: dict):
        """ ST-CPS: Discover regime changepoints under the current causal links.

        :param data: time series, ``data[i]``: ``np.array(T, N)`` for ``T`` time points, ``N`` nodes in context ``i``.
        :return: *see  self.results.*
        """

        # Oracle version: skip if the true regimes known
        if self.cps_strategy == CpsSearchStrategy.SKIP:
            return

        # CPS SEARCH using residual distributions and MDL
        elif self.cps_strategy == CpsSearchStrategy.MDL_ERROR:
            self._info('*** CPS Search ***')
            self.result.cur_regimes, self.result.cur_contexts, self.result.cur_c_each_node, self.result.cur_r_each_node = \
                mdl_error_cp_search(data, self.result.cur_links, self.max_lag, self.min_dur)
        else:
            raise ValueError('Invalid CpsSearchStrategy')
        self._update_cps()

    def _update_cps(self):
        """ Update and check for convergence """
        cur_regimes = self.result.cur_regimes

        cur_equal_previous_regimes = map(
            lambda seen_regimes: self.cps_equals(seen_regimes, cur_regimes, self.cps_convergence_tolerance),
                      [seen_regimes for seen_regimes in self.result.iters_regimes])
        self.converged = any(cur_equal_previous_regimes)

        if self.converged and self.verbosity > 0:
            self.logger.info(f'CPS search converged!')

        self.result.prev_regimes = self.result.cur_regimes
        self.result.iters_regimes.append(self.result.cur_regimes)
        self.result.iters_contexts.append(self.result.cur_contexts)

        if self.to_file:
            self._cps_search_to_file()

    """Util"""

    def _cps_search_to_file(self):
        np.save(f"{self.out}it_{self.cur_it}_regimes", self.result.cur_regimes)
        with open(f"{self.out}it_{self.cur_it}_regimes", 'w') as f:
            csv.writer(f, delimiter=' ').writerows(self.result.cur_regimes)

        np.save(f"{self.out}it_{self.cur_it}_contexts", self.result.cur_contexts)
        with open(f"{self.out}it_{self.cur_it}_contexts", 'w') as f:
            csv.writer(f, delimiter=' ').writerow(self.result.cur_contexts)

    def _dag_search_to_file(self, data: dict):
        dag_model = self.result.cur_dag_model
        cur_adj = dag_model.get_adj()
        cur_tadj = dag_model.get_timed_adj()

        # save causal weights per regime
        for cj in data.keys():
            for ri, (_, _) in enumerate(self.result.cur_regimes):
                cur_weights = self.result.cur_dag_model.get_weights(cj, ri)
                cur_tweights = self.result.cur_dag_model.get_timed_weights(cj, ri)

                for nm, obj in [(f'c{cj}_r{ri}_w', cur_weights), (f'c{cj}_r{ri}_tw', cur_tweights)]:
                    np.save(f"{self.out}it_{self.cur_it}_{nm}", obj)
                    with open(f"{self.out}it_{self.cur_it}_{nm}.csv", 'w') as f:
                        csv.writer(f, delimiter=' ').writerows(obj)
        # save adjacencies
        for nm, obj in [('adj', cur_adj), ('tadj', cur_tadj)]:
            np.save(f"{self.out}it_{self.cur_it}_{nm}", obj)
            with open(f"{self.out}it_{self.cur_it}_{nm}.csv", 'w') as f:
                csv.writer(f, delimiter=' ').writerows(obj)

    def _info_dag_search(self, params):
        info = f"use true regimes {str(params['windows'])}" if (self.cps_strategy == CpsSearchStrategy.SKIP) \
            else "(ignore regimes)" if params['hybrid_cps_dag_discovery'] \
            else f"use current regimes {str(params['windows'])}"
        info += f' using {self.cps_init_strategy}' if self.cur_it == 1 else ""
        #info += f' vs. true regimes {self.truths.windows_T} ' if self.truths is not None else ""
        self.logger.info(
            '\tInitialise DAG search:  ' + info
            + ', taumax ' + str(self.max_lag) + ', hybrid ' + str(params['hybrid_cps_dag_discovery']))

    @staticmethod
    def adj_equals(adj1, adj2):
        return all([all([adj1[i][j] == adj2[i][j] for j in range(len(adj1[i]))]) for i in range(len(adj1))])
    @staticmethod
    def cps_equals(cps1, cps2, tolerance):
        prec, rec, dist = precision_recall_dist_cps(
            cps1, cps2, max_dist=tolerance)
        f1 = 2 * safe_div(prec * rec, prec + rec)
        return f1==1

    """Initialization"""

    def _set_initial_regimes(self, data: dict) -> None:
        """ Set initial changepoints and regimes using CpsInitializationStrategy.

        :param data: time series
        :return:
        """
        if self.cps_strategy not in [CpsSearchStrategy.SKIP, CpsInitializationStrategy.HYBRID]:
            self._info('*** CPS Search ***')
        self.BIN_SIZE = self.min_dur
        n_t, n_n = data[0].shape
        n_bin_samples, n_bin_regions = self.BIN_SIZE, np.floor(n_t / self.BIN_SIZE)

        # Oracle version: use truths
        if self.cps_strategy == CpsSearchStrategy.SKIP:
            assert self.truths is not None
            assert self.truths.windows_T is not None and self.truths.regimes_partition is not None  # cleaner way
            self.result.initial_regimes = self.truths.regimes_partition

        # Hybrid: no initialization needed if the DAG and CPS are discovered at the same time
        elif self.cps_init_strategy == CpsInitializationStrategy.HYBRID:
            return

        # Bin: Segment the time domain into bins of eq duration
        elif self.cps_init_strategy == CpsInitializationStrategy.BINS:
            nb_chunks = int(np.floor(n_t / n_bin_samples))
            assumed_partition = partition_t(n_t, int(n_bin_regions), nb_chunks, n_bin_samples, True)
            self.result.initial_regimes = assumed_partition

        # Cps discovery under the empty graph
        elif self.cps_init_strategy == CpsInitializationStrategy.CPS_FROM_NOPARENTS:
            # only add self transitions and no other parents
            self_links = {0: {i: [((i, -1), None, None)] for i in range(n_n)}}
            found_r_partition, _, found_c_partitions, _ = (
                mdl_error_cp_search(data, self_links, self.max_lag, self.min_dur))
            self.result.initial_regimes = found_r_partition

        # Cps discovery under the fully connected graph
        elif self.cps_init_strategy == CpsInitializationStrategy.CPS_FROM_ALLPARENTS:
            all_parents = list(
                product(list(range(n_n)), list(range(0, -(self.max_lag + 1), -1))))
            full_links = {0: {n: [((p, d), 1, None) for p, d in all_parents] for n in range(n_n)}}

            found_r_partition, _, n_partitions = (
                mdl_error_cp_search(data, full_links, self.max_lag, self.min_dur))
            self.result.initial_regimes = found_r_partition
        else:
            raise ValueError(f"unknown CPS initialisation strategy {self.cps_init_strategy}")

        if self.cps_strategy not in [CpsSearchStrategy.SKIP, CpsInitializationStrategy.HYBRID]:
            self.result.iters_regimes.append(self.result.initial_regimes)# to check for convergence also of the initially found regimes

    def _set_initial_results(self, n_n):
        # Results of previous iteration to check for convergence
        self.result.prev_adj = np.zeros((n_n, n_n))
        self.result.prev_regimes = self.result.initial_regimes

        # Results of most recent iteration
        self.result.cur_regimes = self.result.initial_regimes
        self.result.cur_links = self.result.initial_links

    def _set_initial_links(self, n_n) -> None:
        """ Initialize the timed causal DAG. """
        if self.search_space == DAGSearchSpace.SKIP:
            assert self.truths is not None
            self.result.initial_links = {0: self.truths.true_links}  # convention: links in dict w key 0 for context 0

        else:
            self.result.initial_links = {0: {i: [((i, -1), None, None)] for i in range(n_n)}}


    def get_metrics(self):
        metrics = {}
        metrics_dag, metrics_dag_timed = {}, {}
        metrics_cps = {}
        CPS_TOL = 3

        eval_dag_search = self.search_space != DAGSearchSpace.SKIP and self.cur_it > 0 and self.truths is not None
        if eval_dag_search:
            is_hy = (self.hybrid_cps_dag_discovery or
                     (self.cps_init_strategy == CpsInitializationStrategy.HYBRID and self.cur_it == 0))
            self._info(
                f'\t> DAG Result (hybrid: {is_hy}, R*: {self.cps_strategy == CpsSearchStrategy.SKIP}):')
            metrics_dag = compare_adj_to_links(
                False, self.result.cur_dag_model.get_adj(),
                self.truths.true_links, self.method_type, self.max_lag, False, self.logger,  self.verbosity)
            metrics_dag_timed = compare_timed_adj_to_links(
                False, self.result.cur_dag_model.get_timed_adj(),
                self.truths.true_links, self.method_type, self.max_lag, False, self.logger,  self.verbosity)
        eval_cps_search = self.cps_strategy != CpsSearchStrategy.SKIP \
                          and self.truths is not None and self.truths.regimes_partition is not None
        if eval_cps_search:
            found_r_partition = self.result.cur_regimes

            self._info(f'\t> CPS Result ('
                       f"G*: {'given' if self.search_space == DAGSearchSpace.SKIP else 'unknown'}"
                       f"{f', initialization: {self.cps_init_strategy}' if self.cur_it==0 else''}): " 
                       f"{found_r_partition}")

            self._info(f'\t> CPS True: {self.truths.regimes_partition}')
            prec, rec, metrics_cps['cps-dist'] = precision_recall_dist_cps(
                self.truths.regimes_partition,found_r_partition, CPS_TOL)
            prec, rec, metrics_cps['cps-dist'] = float(prec), float(rec), float(metrics_cps['cps-dist'])
            f1 = 2 * safe_div(prec * rec, prec + rec)
            metrics_cps['cps-prec'], metrics_cps['cps-rec'], metrics_cps['cps-f1'] = prec, rec, f1
            metrics_cps['cps-ari'] = ari(self.truths.regimes_partition, found_r_partition)
            metrics_cps['cps-nmi'] = nmi(self.truths.regimes_partition, found_r_partition)
            # whether the ari and nmi should be counted or not
            metrics_cps['valid'] = 0 if f1 != 1 else 1

            self._info(f"\t> CPS F1: {metrics_cps['cps-f1']:.2} prec: {metrics_cps['cps-prec']:.2}, "
                       f"recall: {metrics_cps['cps-rec']:.2}, "
                       f"ari: {metrics_cps['cps-ari']:.3}, nmi: {metrics_cps['cps-nmi']:.3} "
                       f"(max. tol. {CPS_TOL})")
        for m in [metrics_dag, metrics_dag_timed, metrics_cps]:
            metrics.update(m)

        metrics['n_iterations'] = self.n_it_completed

        return metrics

    def util_score_given_links(self, data: dict, links: dict, windows_T: List = [], r_partition: List = [],
                               to_file=True):
        """ Causal weights (MDL scores) for each context and regime under a known causal graph

        :param data: time series
        :param links: window causal graph
        :param windows_T: custom time windows, set to initial windows if not provided
        :param r_partition: custom regime partition
        :param to_file:  write mdl weights to ..._{ci}_{ri}.csv for each context and regime

        :return:  scores dict for each context and regime
        """

        self.result.cur_links = self.result.initial_links
        # Set Regimes
        if len(windows_T) > 0 and len(r_partition) > 0:
            windows_to_use = windows_T
            partition_to_use = r_partition
        else:
            self._set_initial_regimes(data)
            windows_to_use = r_partition_to_windows_T(self.result.initial_regimes)
            partition_to_use = self.result.initial_regimes

        (n_t, n_n), n_d = data[0].shape, len(data)
        scoring_parameters = dict(
            scoring_function=self.scoring_function,
            discrepancy_test_type=DiscrepancyTestType.SKIP, #placeholder, discrep testing in CPS search
            hybrid_cps_dag_discovery=False,
            windows=windows_to_use, partition=partition_to_use,
            verbosity=self.verbosity-1, lg=self.logger
        )
        dag_search = TimeDAG(data, self.max_lag, self.min_dur, **scoring_parameters)
        scores = {}

        # Scores for each node in each regime in each context
        for node in range(n_n):
            scores[node] = {}
            parents = [pa_time for pa_time, _, _ in links[node]]
            for pa_time in parents:
                mdl = dag_search.eval_edge_time(node, [pa_time], return_scores_each_regime=True)
                others = [p_time for p_time, _, _ in links[node] if p_time[0] != pa_time[0] or p_time[1] != pa_time[1]]
                mdl_others = dag_search.eval_edge_time(node, others, return_scores_each_regime=True)
                scores[node][pa_time] = {}
                for ci in range(len(data)):
                    scores[node][pa_time][ci] = {}
                    for ri in range(len(windows_to_use)):
                        scores[node][pa_time][ci][ri] = mdl_others[ci][ri] - mdl[ci][ri]
        if not to_file:
            return scores

        # Save outputs
        for ci in range(len(data)):
            for ri in range(len(windows_to_use)):
                adj = np.zeros((n_n * 2, n_n))
                for node in range(n_n):
                    parents = [pa_time for pa_time, _, _ in links[node]]
                    for pa_time in parents:
                        pa, lg = pa_time
                        index = n_n * np.abs(lg) + pa
                        adj[index][node] = scores[node][pa_time][ci][ri]  # same score f each parent
                df = pd.DataFrame(adj)
                df.to_csv(self.out + f'_{ci}_{ri}.csv', header=False, index=False)
        return scores

    def util_partition_under_regimes(self, data, links, windows_T):
        """ Discover the regime partition for a given set of changepoints.

        :param data: time series
        :param links: causal links
        :param windows_T: given changepoints
        :return: r_partition, c_partition, regimes_per_node, contexts_per_node
        """
        # Parameters
        n_t, n_n = data[0].shape

        # Discover partitions for each node
        regimes_per_node = partition_regimes(data, links, windows_T, {'N': n_n, 'T': n_t})
        regimes = regimes_map_from_constraints(regimes_per_node)
        r_partition = windows_T_to_r_partition(windows_T, self.max_lag, regimes)

        contexts_per_node = partition_contexts_over_regimes(data, links, regimes, windows_T, {'N': n_n, 'T': n_t})
        c_partition = regimes_map_from_constraints(contexts_per_node)

        return r_partition, c_partition, regimes_per_node, contexts_per_node

