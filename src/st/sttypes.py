from enum import Enum

from st.scoring.models.gpr_fourierf_mdl import GaussianProcessFourierRegularized
from st.scoring.models.gpr_mdl import GaussianProcessRegularized


class TimeseriesScoringFunction(Enum):
    GP = 'time_gaussian_process'
    GP_QFF = 'time_gaussian_process_qff'
    GLOBE = 'time_globe_splines'

    def get_model(self, kernel, alpha, n_restarts_optimizer):
        if self.value in [self.GP_QFF.value]:
            return GaussianProcessFourierRegularized(kernel=kernel, alpha=alpha,
                                                     n_restarts_optimizer=n_restarts_optimizer)
        else:
            return GaussianProcessRegularized(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)


class SubsamplingApproach(Enum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    JUMPING_WINDOW = 'jumping_window'


class ContinuousScoringFunction(Enum):
    GP = 'cont_gaussian_process'
    GLOBE = 'cont_globe_splines'


class ScoringFunction(Enum):
    TIME = TimeseriesScoringFunction
    CONTINUOUS = ContinuousScoringFunction


class DAGSearchSpace(Enum):
    SKIP = 0
    GREEDY_TREE_SEARCH = 1
    EXHAUSTIVE_SEARCH = 2

    def __str__(self):
        names = ["skip", "greedy_dag_search", "exhaustive_search"]
        return names[self.value]

    def __eq__(self, other):
        return self.value == other.value


class CpsSearchStrategy(Enum):
    SKIP = 0
    MDL_ERROR = 1

    def __str__(self):
        names = ["skip", "mdl_error"]
        return names[self.value]

    def __eq__(self, other):
        return self.value == other.value


class CpsInitializationStrategy(Enum):
    CPS_FROM_NOPARENTS = 0
    CPS_FROM_ALLPARENTS = 1
    BINS = 2
    HYBRID = 3

    def __str__(self):
        names = ["noparents", "allparents", "bins", "hybrid"]
        return names[self.value]

    def __eq__(self, other):
        return self.value == other.value


class MethodType(Enum):
    """ Time series causal discovery methods considered in the experiments.

    :Methods:
    * *ST_GP* -- interleaved regime cps search and dag search with Gaussian process regression
    * *ST_GP_SPLINE* -- ... with Spline regression and the GLOBE/SLOPE MDL score
    * *ST_GP_HYBRID* -- hybrid cps and dag search, refit the regime cps whenever scoring an edge
    * *ST_GP_QFF* -- *GP* with fourier feature approximation of GPs
    * *ST_GP_QFF_HYBRID*
    * *ST_GP_REGIMES* -- *GP* with regime cps known, to evaluate dag search
    * *JPCMCI* -- full method, apply over all contexts with dummy variables, ignoring the regimes
    * *JPCMCI_REGIMES* -- known regimes, apply JPCMCI on each regime (that regime window in all contexts)
    * *RPCMCI* -- full, find regimes (known number), applied in each context
    * *RPCMCI_REGIMES* -- known regime number and CPS via masking - avg contexts, equivalent to pcmci-regimes
    * *VARLINGAM* -- VarLingam.
    * *VARLINGAM_REGIMES* -- VarLingam, known regimes.
    * *PCMCIPLUS* -- PCMCIplus.
    * *PCMCIPLUS_REGIMES* -- PCMCIplus, known regimes.
    * *CDNOD* -- CD-NOD.
    * *CDNOD_REGIMES* -- CD-NOD, known regimes.
    * *DYNOTEARS* -- DyNOTEARS.
    * *DYNOTEARS_REGIMES* -- DyNOTEARS, known regimes.
    """
    ST_GP = 'st'
    ST_SPLINE = 'st_spline'
    ST_GP_QFF = 'st_qff'
    ST_GP_REGIMES = 'st_oracle_regimes'
    ST_GP_DAG = 'st_oracle_dag'
    ST_GP_HYBRID = 'st_hybrid'
    ST_GP_QFF_HYBRID = 'st_qff_hybrid'
    JPCMCI = 'JPCMCI'
    JPCMCI_REGIMES = 'JPCMCI_oracle_regimes'
    RPCMCI = 'RPCMCI'
    RPCMCI_REGIMES = 'RPCMCI_oracle_regimes'
    VARLINGAM = 'varlingam'
    VARLINGAM_REGIMES = 'varlingam_oracle_regimes'
    PCMCIPLUS = 'PCMCIP'
    PCMCIPLUS_REGIMES = 'PCMCIP_oracle_regimes'
    CDNOD = 'CDNOD'
    CDNOD_REGIMES = 'CDNOD_oracle_regimes'
    DYNOTEARS = 'DYNOTEARS'
    DYNOTEARS_REGIMES = 'DYNOTEARS_oracle_regimes'

    def __eq__(self, other):
        return self.value == other.value

    def is_spacetime(self):
        return self.value.startswith('st')

    def get_scoring_function(self):
        assert self.is_spacetime()
        return TimeseriesScoringFunction.GP_QFF if self.value in \
        [MethodType.ST_GP_QFF.value, MethodType.ST_GP_QFF_HYBRID.value] \
            else TimeseriesScoringFunction.GLOBE if self.value in [MethodType.ST_SPLINE.value]\
            else TimeseriesScoringFunction.GP

    def get_cps_init_strategy(self):
        assert self.is_spacetime()
        return CpsInitializationStrategy.CPS_FROM_NOPARENTS

    def get_search_space(self):
        assert self.is_spacetime()
        return DAGSearchSpace.SKIP if self.value == self.ST_GP_DAG.value \
            else DAGSearchSpace.GREEDY_TREE_SEARCH

    def get_cps_strategy(self):
        assert self.is_spacetime()
        return CpsSearchStrategy.SKIP if self.value == self.ST_GP_REGIMES.value \
            else CpsSearchStrategy.MDL_ERROR

    def is_hybrid(self):
        assert self.is_spacetime()
        return self.value in [self.ST_GP_HYBRID.value, self.ST_GP_QFF_HYBRID.value]

    def is_timed(self):
        """methods that can return a window causal graph, others can only return summary graph without time lags"""
        return self.value not in [self.CDNOD.value, self.CDNOD_REGIMES.value]

    def assumes_singlecontext(self):
        return self.value in [
            self.RPCMCI.value, self.RPCMCI_REGIMES.value,
            self.VARLINGAM.value, self.VARLINGAM_REGIMES.value,
            self.PCMCIPLUS.value, self.PCMCIPLUS_REGIMES.value,
            self.CDNOD.value, self.CDNOD_REGIMES.value,
            self.DYNOTEARS.value, self.DYNOTEARS_REGIMES.value
        ]

    def partitions_regimes(self):
        return self.is_spacetime() or self.value in [self.RPCMCI.value, self.RPCMCI_REGIMES.value]

    def get_name(self, timed=False, pooling_contexts=False):
        if timed:
            assert self.is_timed()
        if pooling_contexts:
            assert self.assumes_singlecontext()

        suff = '-timed' if timed else ''
        suff = suff + '-C' if pooling_contexts else suff
        return self.value + suff



