import logging
import warnings

import numpy as np
from matplotlib import pyplot as plt

from exp.utils.gen_timeseries import gen_timeseries
from st.spacetime import SpaceTime
import exp.utils.plot_timeseries as stp
from st.sttypes import MethodType

if __name__ == "__main__":
    logging.basicConfig()
    lg = logging.getLogger("EXAMPLE")
    lg.setLevel("INFO")

    # DATA PARAMS
    true_tau_max = 1
    true_min_dur = 30
    hat_tau_max = 1
    hat_min_dur = 20
    seed = 42

    np.random.seed(seed)
    params = {'C': 2, 'R': 3, 'CPS': 2, 'T': 500, 'D': 1, 'N': 5, 'I': 0.5}
    data, truths = gen_timeseries(true_tau_max, true_min_dur, lg=lg, seed=seed, **params)
    stp.plot_timeseries_regimes_contexts(data, truths.regimes_partition)


    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = SpaceTime(hat_tau_max, hat_min_dur, truths=truths, logger=lg, verbosity=2)
        st.run(data.datasets)

