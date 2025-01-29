import logging
from typing import List

from exp.utils.gen_timeseries import Func, Noise
from st.sttypes import MethodType


class Options:
    logger: logging
    seed: int
    rep: int
    reps: int = 50
    n_jobs: int = 1
    quick: bool
    fixed: dict = {}
    functions: List = []
    exps: List = []

    parameters_to_change = {
        'N': 'nodes', 'T': 'timepoints', 'D': 'datasets', 'CPS': 'changepoints',
        'R': 'regimes', 'C': 'contexts', 'I': 'interventions', 'IVS': 'intervention_strength',
    }
    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.noises = [ns for ns in map(Noise, self.noises)]
        self.functions = [fun for fun in map(Func, self.functions)]

        self.methods = kwargs.get("methods", [MethodType.ST_GP.value])
        self.methods = [mtd for mtd in map(MethodType, self.methods)]

        self.attrs = self.get_attributes()
        self.exps = self.get_experiment_cases()

        self.reps = min(self.reps, 3) if self.quick else max(self.reps, 3) # for computing avgs etc

    def get_attributes(self):
        params = {short_nm: self.__dict__[long_nm] for (short_nm, long_nm) in self.parameters_to_change.items()}
        return params

    def get_cases(self):
        self.fixed = {attr: val[0] for (attr, val) in self.attrs.items()}
        # Keep one attribute fixed and get all combos of the others
        combos = [
            ({nm: (self.attrs[nm][i] if nm == fixed_nm else self.fixed[nm]) for nm in self.attrs})
            for fixed_nm in self.fixed
            for i in range(len(self.attrs[fixed_nm]))
        ]
        test_cases = {"_".join(f"{arg}_{val}" for arg, val in combo.items()): combo for combo in combos}
        # small runs first
        test_cases = dict(sorted(test_cases.items(), key=lambda dic: (dic[1]["N"], dic[1]["T"])))
        return test_cases

    def get_experiment_cases(self):
        attrs = {param: self.__dict__[nm]  for nm, param in  {('noises', 'NS'), ('functions', 'F')}}
        combos = [{'F': fun, 'NS': ns} for fun in attrs['F'] for ns in attrs['NS']]
        return combos