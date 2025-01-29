import math
import numpy as np
from tigramite.toymodels.structural_causal_processes import _Graph
from tigramite.toymodels.context_model import ContextModel


class ContextModelTSWithRegimes(ContextModel):
    def __init__(self, links_regimes, links=None, *args, **kwargs):
        links = links_regimes[0]  # dummy
        super().__init__(links=links, *args, **kwargs)
        self.links_regimes = links_regimes

    def generate_data_with_regimes(self, M, T, partition, n_drift, uniform_noise):
        links = {**self.links_tc, **self.links_sc, **self.links_sys}

        K_time = len(self.links_tc.keys())

        data = {}
        nonstationary = []

        time_seed = [1, self.seed]
        space_seed = [2, self.seed]
        system_seed = [3, self.seed]

        # first generate data for temporal context nodes
        data_tc_list, nonstat_tc = self._generate_temporal_context_data(self.links_tc, T, M, time_seed)

        # # generate spatial context data (constant in time)

        if not isinstance(self.noises, np.ndarray):
            data_sc_list, nonstat_sc = self._generate_spatial_context_data(self.links_sc,
                                                                           T, M,
                                                                           K_time + self.N,
                                                                           space_seed)
        else:
            data_sc_list = [{self.N + 1: self.noises[-1][int(self.transient_fraction * T):]}]
            nonstat_sc = False

        for m in range(M):
            data_context = dict(data_tc_list[m])
            data_context.update(data_sc_list[m])

            if self.noises is not None:
                noises_filled = self.noises
                if np.all([isinstance(el, np.ndarray) for el in self.noises]):
                    noises_filled = np.copy(self.noises)
                    for key in self.links_sc.keys():
                        # fill up any space-context noise to have T entries, then convert to numpy array
                        if uniform_noise:
                            noises_filled[key] = np.random.uniform(
                                size=len(self.noises[list(self.links_sys.keys())[0]]))
                        else:
                            noises_filled[key] = np.random.standard_normal(
                                len(self.noises[list(self.links_sys.keys())[0]]))
                    noises_filled = np.stack(noises_filled).transpose()
            else:
                noises_filled = None

            # generate system data that varies over space and time
            data_m, nonstat = structural_causal_process(links, T=T, links_regimes=self.links_regimes,
                                                        partition=partition, intervention=data_context,  # toys.
                                                        transient_fraction=self.transient_fraction,
                                                        seed=system_seed, noises=noises_filled, n_drift=n_drift)
            data[m] = data_m
            nonstationary.append(nonstat or nonstat_tc or nonstat_sc)
        return data, np.any(nonstationary)


def structural_causal_process(links, T, links_regimes, partition, noises=None,
                              intervention=None, intervention_type='hard',
                              transient_fraction=0.2,
                              seed=None, n_drift=0):
    """Returns a time series generated from a structural causal process.

    Allows lagged and contemporaneous dependencies and includes the option
    to have intervened variables or particular samples.

    The interventional data is in particular useful for generating ground
    truth for the CausalEffects class.

    In more detail, the method implements a generalized additive noise model process of the form

    .. math:: X^j_t = \\eta^j_t + \\sum_{X^i_{t-\\tau}\\in \\mathcal{P}(X^j_t)}
              c^i_{\\tau} f^i_{\\tau}(X^i_{t-\\tau})

    Links have the format ``{0:[((i, -tau), coeff, func),...], 1:[...],
    ...}`` where ``func`` can be an arbitrary (nonlinear) function provided
    as a python callable with one argument and coeff is the multiplication
    factor. The noise distributions of :math:`\\eta^j` can be specified in
    ``noises``.

    Through the parameters ``intervention`` and ``intervention_type`` the model
    can also be generated with intervened variables.

    Parameters
    ----------
    links : dict
        Dictionary of format: {0:[((i, -tau), coeff, func),...], 1:[...],
        ...} for all variables where i must be in [0..N-1] and tau >= 0 with
        number of variables N. coeff must be a float and func a python
        callable of one argument.
    T : int
        Sample size.
    noises : list of callables or array, optional (default: 'np.random.randn')
        Random distribution function that is called with noises[j](T). If an array,
        it must be of shape ((transient_fraction + 1)*T, N).
    intervention : dict
        Dictionary of format: {1:np.array, ...} containing only keys of intervened
        variables with the value being the array of length T with interventional values.
        Set values to np.nan to leave specific time points of a variable un-intervened.
    intervention_type : str or dict
        Dictionary of format: {1:'hard',  3:'soft', ...} to specify whether intervention is
        hard (set value) or soft (add value) for variable j. If str, all interventions have
        the same type.
    transient_fraction : float
        Added percentage of T used as a transient. In total a realization of length
        (transient_fraction + 1)*T will be generated, but then transient_fraction*T will be
        cut off.
    seed : int, optional (default: None)
        Random seed.

    Returns
    -------
    data : array-like
        Data generated from this process, shape (T, N).
    nonvalid : bool
        Indicates whether data has NaNs or infinities.

    """
    random_state = np.random.RandomState(seed)

    N = len(links.keys())
    if noises is None:
        noises = [random_state.randn for j in range(N)]

    if N != max(links.keys()) + 1:
        raise ValueError("links keys must match N.")

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction * T)), N):
            raise ValueError("noises.shape must match ((transient_fraction + 1)*T, N).")
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp_dag = _Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N - 1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort()

    if intervention is not None:
        if intervention_type is None:
            intervention_type = {j: 'hard' for j in intervention}
        elif isinstance(intervention_type, str):
            intervention_type = {j: intervention_type for j in intervention}
        for j in intervention.keys():
            if len(intervention[j]) != T:
                raise ValueError("intervention array for j=%s must be of length T = %d" % (j, T))
            if j not in intervention_type.keys():
                raise ValueError("intervention_type dictionary must contain entry for %s" % (j))

    transient = int(math.floor(transient_fraction * T))

    data = np.zeros((T + transient, N), dtype='float32')
    # n = dict()
    for j in range(N):  # initial noise for each variable
        if isinstance(noises, np.ndarray):
            data[:, j] = noises[:, j]
        else:
            data[:, j] = noises[j](T + transient)  # TODO: noise amplitude is huge
            # n[j] = np.copy(data[:, j])
            # data[0:2, j] = noises[j](1) # no noise except to set time series origin

    for t in range(max_lag, T + transient):
        if t >= transient:
            index = [i for i, (deb, dur, r) in enumerate(partition) if deb <= t - transient < deb + dur][0]
            regime = partition[index][2]  # [r for deb, dur, r in partition if deb <= t-transient < deb+dur][0]
            r1, r2 = regime, regime
            if n_drift != 0:
                if index != 0 and t - transient < partition[index][0] + (n_drift / 2):
                    r1 = partition[index - 1][2]
                    index_drift = int(t - transient - (partition[index][0] - (n_drift / 2)))
                if index != len(partition) - 1 and t - transient > partition[index + 1][0] - (n_drift / 2):
                    r2 = partition[index + 1][2]
                    index_drift = int(t - transient - (partition[index + 1][0] - (n_drift / 2)))
        else:
            r1 = r2 = regime = partition[0][2]  # during transient period, regime same as the first one
        for j in causal_order:

            if (intervention is not None and j in intervention and t >= transient
                    and np.isnan(intervention[j][t - transient]) == False):
                if intervention_type[j] == 'hard':
                    data[t, j] = intervention[j][t - transient]
                    # Move to next j and skip link_props-loop from parents below
                    continue
                else:
                    data[t, j] += intervention[j][t - transient]

            # This loop is only entered if intervention_type != 'hard'
            for ii, link_props in enumerate(links_regimes[regime][j]):
                var, lag = link_props[0]
                if r1 == r2:
                    coeff = link_props[1]
                else:
                    coeff = np.linspace(links_regimes[r1][j][ii][1], links_regimes[r2][j][ii][1], int(n_drift))[
                        index_drift]
                func = link_props[2]

                data[t, j] += func(data[t + lag, var], coeff, r1)

    data = data[transient:]

    nonvalid = (np.any(np.isnan(data)) or np.any(np.isinf(data)))

    return data, nonvalid
