from exp.utils.gen_timeseries import Func, Noise
from exp.options import Options
from exp.run import run
from st.sttypes import MethodType


if __name__ == "__main__":
    """
    >>> python run_exp.py --quick 
    """
    import sys
    import argparse
    import logging
    from pathlib import Path

    ap = argparse.ArgumentParser("ST")

    def enum_help(enm) -> str:
        return ','.join([str(e.value) + '=' + str(e) for e in enm])

    # Experiment Parameters
    ap.add_argument("-mtd", "--methods", default=['st'], nargs="+", help=f"valid methods: {enum_help(MethodType)}", type=str)
    ap.add_argument("-rep", "--reps", default=50, help="repetitions per experiment case", type=int)
    ap.add_argument("-sd", "--seed", default=42, help="seed", type=int)
    ap.add_argument("-nj", "--n_jobs", default=1, help="n parallel jobs", type=int)

    # Experiments to iterate over
    ap.add_argument("-f", "--functions", default=['nlin', 'lin'], nargs="+", help=f"valid functional forms: {enum_help(Func)}", type=str)
    ap.add_argument("-ns", "--noises", default=['gauss', 'unif'], nargs="+", help=f"valid noises: {enum_help(Noise)}", type=str)
    # Data generation parameters to iterate over per experiment
    ap.add_argument("-n", "--nodes", default=[5, 10], nargs="+", help="num nodes", type=int)
    ap.add_argument("-t", "--timepoints", default=[200, 500], nargs="+",
                    help="time series length (each dataset, over all regimes combined)", type=int)
    ap.add_argument("-d", "--datasets", default=[1], nargs="+", help="number of datasets (each with length T)",
                    type=int)
    ap.add_argument("-cps", "--changepoints", default=[2], nargs="+", help="num cutpoints", type=int)
    ap.add_argument("-r", "--regimes", default=[2], nargs="+", help="num regimes", type=int)
    ap.add_argument("-c", "--contexts", default=[2], nargs="+", help="num contexts", type=int)
    ap.add_argument("-i", "--interventions", default=[0.5], nargs="+",
                    help="frac intervened nodes per context/regime", type=int)
    ap.add_argument("-str", "--intervention_strength", default=[0.5], nargs="+",
                    help="strength of coefficient changes", type=int)
    # Data generation parameters held constant
    ap.add_argument("-ttm", "--true_tau_max", default=1, help="tau max in data generation", type=int)
    ap.add_argument("-tmd", "--true_min_dur", default=20, type=int)
    ap.add_argument("-rd", "--regime_drift", default=False, type=bool)
    ap.add_argument("-hi", "--hard_intervention", default=False, type=bool)
    ap.add_argument("--quick", action="store_true", help="run a shorter experiment for testing")
    ap.add_argument("--safe", action="store_true", help="catch exceptions and skip experiment cases")

    # Method hyperparameters
    ap.add_argument("-ataumax", "--assumed_max_lag", default=1, help="tau max that our algo assumes", type=int)
    ap.add_argument("-amindur", "--assumed_min_dur", default=20, type=int)
    ap.add_argument("-v", "--verbosity", default=2, help='use 1 to see outputs, >1 to see output of dag search', type=int)

    # Path
    ap.add_argument("-bd", "--base_dir", default="")
    ap.add_argument("-wd", "--write_dir", default="res/")

    argv = sys.argv[1:]
    nmsp = ap.parse_args(argv)

    logging.basicConfig()
    log = logging.getLogger("ST")
    log.setLevel("INFO")

    options = Options(**nmsp.__dict__)

    # Logging
    options.logger = log
    out_dir = nmsp.write_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"{out_dir}run.log")
    fh.setLevel(logging.INFO)
    options.logger.addHandler(fh)
    options.out_dir = out_dir

    import warnings
    warnings.filterwarnings("ignore")

    run(options)
