import logging
import numpy as np

from st.scoring.discrepancy_testing import DiscrepancyTestType
from st.search.cps_search import mdl_error_cp_search, eval_fit_partition
from st.sttypes import TimeseriesScoringFunction, SubsamplingApproach
from st.scoring.time_space_fitting import fit_functional_model_time_space
from st.utils.util_regimes import r_partition_to_windows_T


def hybrid_score_edge_time_space(
        data_C: dict, covariates: list, target: int, max_lag: int, min_dur: int,
        scoring_function: TimeseriesScoringFunction, discrepancy_test_type: DiscrepancyTestType,
        subsampling_approach: SubsamplingApproach = SubsamplingApproach.HORIZONTAL,
        lg: logging = None, verbosity: int = 0, edge_info=""):
    """First find the CPS and regime partition for node given its parents, then compute the score"""

    n_n = data_C[0].shape[0]
    target_links = _lagged_target_links(covariates, target)
    # Consider only the target and its parents
    links = {0: {i: target_links if i == target else [((i, -1), None, None)] for i in range(n_n)}}

    found_r_partition, _, _, _ = mdl_error_cp_search(data_C, links, max_lag, min_dur)

    windows_T = r_partition_to_windows_T(found_r_partition, max_lag)
    mdl_scores = score_edge_time_space(
        data_C, covariates, target, windows_T, found_r_partition, max_lag, min_dur,
        scoring_function, discrepancy_test_type, subsampling_approach,
        lg, verbosity, edge_info)
    if verbosity > 1:
        lg.info(f"\t\t\t-edge {target} <- {covariates}\ts: {[score for score in mdl_scores]}"
                f"\tcps: {windows_T}\ttruth: {edge_info}")

    return mdl_scores, windows_T


def score_edge_time_space(
        data_C: dict, covariates: list, target: int, windows_T: list, regimes_R: list, max_lag: int, min_dur: int,
        scoring_function: TimeseriesScoringFunction, discrepancy_test_type: DiscrepancyTestType,
        subsampling_approach: SubsamplingApproach = SubsamplingApproach.HORIZONTAL,
        lg: logging = None, verbosity: int = 0, edge_info="", ) -> np.array:
    """Compute the score for a node given its parents, using the provided CPS and regime partition"""
    target_links = _lagged_target_links(covariates, target)
    C = len(data_C)
    T = len(windows_T)
    only_one_regime_exists = len(np.unique([regime for _, _, regime in regimes_R])) == 1

    # Fit a different GP in each context and each time window/chunk, disregard the regime labels
    if only_one_regime_exists or discrepancy_test_type == DiscrepancyTestType.SKIP:
        # Fit a GP for each edge in each context and each time window
        scores_time_space = fit_functional_model_time_space(
            scoring_function, subsampling_approach, data_C, windows_T, target_links, target, return_models=False)

        scores = [[scores_time_space[c_i][w_i] for w_i in range(T)] for c_i in range(C)]
        if verbosity > 1:
            lg.info(
                f'\tEval Edge {covariates} -> {target}: {np.round(np.flatten(sum(sum(np.array(scores_time_space)))), 2)}\t{edge_info}')
        return scores
    # Fit a different GP in each context and same regime
    else:
        # only consider this target
        links = {0: {0: target_links}}
        # Given the current CPS, find the regime partition and sum up scores
        score, r_partition, contexts_per_node, regimes_per_node, gps, hist, scores = \
            eval_fit_partition(data_C, links, windows_T, max_lag, hist=None)
        return scores


def score_edge_continuous(
        data_C: dict, covariates: list, target: int,
        scoring_function, lg: logging = None, verbosity: int = 0, info: str = "") -> int:
    raise NotImplementedError("Potential support for non-time series scoring functions.")


def _lagged_target_links(covariates, i):
    """Generates the links for a single effect, with given time lags"""
    links = []
    links.append(((i, -1), 1, None))
    # Add causal parents
    for j, lag in covariates:
        links.append(((j, -lag), 1, None))
    return links
