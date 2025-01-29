from joblib import Parallel, delayed
from numpy.random import SeedSequence

import numpy as np

from exp.utils.gen_timeseries import gen_timeseries
from exp.options import Options
from exp.utils.case_results import CaseReslts, write_cases
from st.spacetime import SpaceTime

""" Run ST on simulated data"""


def run(options: Options):
    """ Main Method for running our experiments """
    from datetime import datetime
    options.logger.info(f"RUN AT: {str(datetime.now())}")

    for exp in options.get_experiment_cases():
        exp_cases = options.get_cases()
        options.logger.info(f"Experiment {exp}")

        for case in exp_cases.items():
            run_case_safe(options, exp, case) if options.safe else run_case(options, exp, case)

        for attr in options.fixed:
            write_cases(options, None, attr, base_attributes=options.fixed)

def run_case_safe(options, exp, case):
    try:
        run_case(options, exp, case)
    except Exception as e:
        print(f"case failed {e}, skipping it")
        options.inc_seed()


def run_case(options, exp, test_case):
    """ Runs a case with fixed parameters (n nodes, n timepoints ...) """
    ss = SeedSequence(options.seed)
    cs = ss.spawn(options.reps)

    case, params = test_case
    for e,val in exp.items():
        params[e] = val
    run_rep = lambda rep: run_repetition(options, params, case, rep)

    if options.n_jobs != 1:
        reslts = Parallel(n_jobs=options.n_jobs)(delayed(
            run_rep)(rep_seed) for rep_seed in enumerate(cs))
    else:
        original_out_dir = options.out_dir
        options.out_dir = original_out_dir + 'intermediate/'
        reslts = []
        for rep_seed in enumerate(cs):
            res = run_rep(rep_seed)
            reslts.append(res)
        options.out_dir = original_out_dir

    case_results = CaseReslts(case)
    case_results.add_reps(reslts)
    case_results.write_case(params, exp, options)


def run_repetition(options, params, case, rep_seed):
    import warnings
    warnings.filterwarnings("ignore")

    rep_random_state = np.random.default_rng(rep_seed[1])
    options.logger.info(f'*** Rep {rep_seed[0] + 1}/{options.reps}, seed: {rep_seed[0]}***')
    options.logger.info(f'Params: {case}')

    # Generate Data
    data, truths = gen_timeseries(
          options.true_tau_max, options.true_min_dur, rstate=rep_random_state, **params)

    # Run methods
    return {method.value: run_method(options, params, data, truths, method) for method in options.methods}


def run_method(options, params, data, truths, method):
    import time
    options.logger.info(f'Running {str(method)}')
    start = time.time()
    result = {}
    if method.is_spacetime():
        result = run_method_spacetime(options, data, truths, method)
    else:
        raise ValueError(method)

    result['time'] = np.round(time.time() - start)
    options.logger.info(f'Done {str(method)} in {time.time() - start:.2f}s')

    return result


def run_method_spacetime(options, data, truths, method):
    assert method.is_spacetime()
    spct = SpaceTime(
        max_lag=options.assumed_max_lag,
        min_dur=options.assumed_min_dur,
        scoring_function=method.get_scoring_function(),
        method_type=method,
        logger=options.logger,
        verbosity=options.verbosity,
        out=options.out_dir,
        truths=truths
    )

    spct.run(data.datasets)
    final_metrics = spct.get_metrics()
    options.logger.info(f"N iterations: {final_metrics['n_iterations']}")
    return final_metrics
