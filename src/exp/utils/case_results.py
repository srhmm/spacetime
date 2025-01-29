import os
import statistics
from collections import defaultdict
from tabulate import tabulate

import numpy as np
import pandas as pd


class CaseMethodReslts:
    def __init__(self, case, nm):
        self.case = case
        self.nm = nm
        self.metrics = defaultdict(list)

    def add_method_rep(self, method_results):
        for met in method_results:
            self.metrics[met] += [method_results[met]]


class CaseReslts:
    def __init__(self, case):
        self.case = case
        self.method_results = {}

    def add_reps(self, all_results):
        for rep in range(len(all_results)):
            for method_nm in all_results[rep]:
                one_result = all_results[rep][method_nm]
                self._add_rep(one_result, method_nm)

    def _add_rep(self, one_result, nm):
        if nm not in self.method_results:
            self.method_results[nm] = CaseMethodReslts(self.case, nm)
        self.method_results[nm].add_method_rep(one_result)


    def write_case(self, params, exp, options):
        options.logger.info("")
        options.logger.info(f"***RESULTS***  Case: {self.case}")
        table = self.method_results

        exp_name = '_'.join([str(val) for val in exp.values()])
        path = options.out_dir + exp_name + "/tikzfiles/all/"
        os.makedirs(path, exist_ok=True)

        # results of each run
        methods = table.keys()
        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(f'\t{str(mth)}_{met}')
            for r in range(options.reps):
                write_file.write(f'\n{r}')
                for met in table[mth].metrics.keys():
                    if len(table[mth].metrics[met]) >= r:
                        write_file.write(f'\t{table[mth].metrics[met][r]} ')
                    else:
                        write_file.write(f'\t{-1}')
            write_file.close()

        # Print averages of all runs for this case
        path = options.out_dir + exp_name +  "/tikzfiles/avg/"

        if not os.path.exists(path):
            os.makedirs(path)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(f'\t{str(mth)}_{met}_mn\t{str(mth)}_md\t{str(mth)}_{met}_var\t{str(mth)}_{met}_std')
            write_file.write(f'\n0')
            for met in table[mth].metrics.keys():
                relevant_entries = [x for x in table[mth].metrics[met] if x != -1] # -1 is placeholder for NaN
                if len(relevant_entries) > 2:
                    try:
                        write_file.write(
                            f'\t{statistics.mean(relevant_entries)}'
                            f'\t{statistics.median(relevant_entries)}'
                            f'\t{statistics.variance(relevant_entries)}'
                            f'\t{statistics.stdev(relevant_entries)}')
                    except:
                        write_file.write( f'\t-1\t-1\t-1\t-1')
                elif len(table[mth].metrics[met]) == 1:
                    write_file.write(f'\t{relevant_entries[0]}\t0\t0\t0')
                else:
                    write_file.write(f'\t{-1}\t0\t0\t0')
            write_file.close()

        # Print only means
        path = options.out_dir + exp_name + "/tikzfiles/avgmn/"

        if not os.path.exists(path):
            os.makedirs(path)

        for mth in methods:
            fl = os.path.join(path, f"{self.case}_m_{mth}.tsv")
            write_file = open(fl, 'w')
            write_file.write(f'X')

            for met in table[mth].metrics.keys():
                write_file.write(
                    f'\t{met}')
            write_file.write(f'\n0')
            for met in table[mth].metrics.keys():
                relevant_entries = [x for x in table[mth].metrics[met] if x != -1] # -1 is placeholder for NaN
                if len(relevant_entries) > 2:
                    try:
                        write_file.write(
                            f'\t{np.round(statistics.mean(relevant_entries), 2)}')
                    except:
                        write_file.write(f'\t-1')
                elif len(relevant_entries) == 1:
                    write_file.write(f'\t{relevant_entries[0]}')
                else:
                    write_file.write(f'\t{-1}')
            write_file.close()

            import csv
            from tabulate import tabulate

            options.logger.info(f"\tMethod: {mth}")

            with open(fl) as csv_file:
                reader = csv.reader(csv_file, delimiter='\t')
                rows = [row for row in reader]
                options.logger.info(tabulate(rows, tablefmt="pretty" ))


def write_cases(options, exp, relevant_attribute,
                base_attributes
                ):
    fixed_attributes = {ky: vl for (ky, vl) in base_attributes.items() if ky != relevant_attribute}
    in_path = options.out_dir + str(options.exp_type) + "_" + '_'.join(
        [f"{e}_{exp[e]}" for e in exp]) +  "/tikzfiles/avg/"
    out_path = options.out_dir + str(options.exp_type) + "_" + '_'.join(
        [f"{e}_{exp[e]}" for e in exp]) +  "/tikzfiles/vry/"

    os.makedirs(out_path, exist_ok=True)

    import glob

    fls = []
    for file in glob.glob(os.path.join(in_path, "*.tsv")):
        fls.append(file)

    relevant_per_method = {}
    relevant_metrics = {}
    found_attributes = {}

    # Extract all files that have
    # - one attribute A varying (e.g. number of latent vars MZ)
    # - all other attributes equal to base attribute values (e.g. the base case we study, 10 nodes, 1000 samples ...)
    # and store per method
    for fl in fls:
        suff = fl.split("/")[len(fl.split("/"))-1].split('.tsv')[0]
        parts = suff.split('_')

        mthd, relevant_attribute_val, contains_base_attributes = None, None, True

        # check all base attrs covered
        for ip, p in enumerate(parts):
            if p in fixed_attributes:
                contains_base_attributes = contains_base_attributes and parts[ip+1]==str(fixed_attributes[p])
        if not contains_base_attributes:
            continue

        # extract method and results
        for ip, p in enumerate(parts):
            if p == 'm':
                mthd = parts[ip+1]
                if mthd in relevant_per_method:
                    continue
                relevant_per_method[mthd] = []
                relevant_metrics[mthd] = {}
                found_attributes[mthd] = []
        for ip, p in enumerate(parts):
            if p == relevant_attribute:
                relevant_attribute_val = parts[ip+1]

                if relevant_attribute_val not in found_attributes[mthd]:
                    found_attributes[mthd].append(relevant_attribute_val)
        relevant_per_method[mthd].append(fl)

        print(fl)
        tb = pd.read_csv(fl, sep='\t')
        for metr in tb.columns:
            if metr == 'X': #this was some placeholder column
                continue
            if metr not in relevant_metrics[mthd]:
                relevant_metrics[mthd][metr] = {}
            relevant_metrics[mthd][metr][relevant_attribute_val] = tb[metr].iloc[0]


    for mthd in found_attributes:
        found_attributes[mthd] = sorted(found_attributes[mthd], key=float)

    # For each method, create the following file
    # cols: method_metric1_mn, method_metric1_var, method_metric1_std ... method_metricN_std
    # rows: value1(relevant_attribute) .... valueN (relevant_attribute)
    base_attribute_idf = '_'.join([f'{ky}_{vl}'for ky, vl in fixed_attributes.items()])
    for mthd in relevant_metrics:
        out_fl = os.path.join(out_path, f"{base_attribute_idf}_{mthd}.tsv")

        write_file = open(out_fl, 'w')
        write_file.write(f'{relevant_attribute}')

        for met in relevant_metrics[mthd].keys():
            if not np.all([attr_val in relevant_metrics[mthd][met] for attr_val in found_attributes[mthd]]):
                continue
            write_file.write(f'\t{str(met)}') #f'\t{str(mthd)}_{met}_mn\t{str(mthd)}_md\t{str(mthd)}_{met}_var\t{str(mthd)}_{met}_std')

        for attr_val in found_attributes[mthd]:
            write_file.write(f'\n{attr_val}')
            for met in relevant_metrics[mthd].keys():
                if attr_val in relevant_metrics[mthd][met] and (relevant_metrics[mthd][met][attr_val] is not None):
                    write_file.write(f'\t{relevant_metrics[mthd][met][attr_val]}')
                else:
                    continue

        write_file.close()

        # in addition log resulting file content
        import csv
        with open(out_fl) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            rows = [row for row in reader]
            options.logger.info(tabulate(rows, tablefmt="pretty"))
    print("X")


def getfolder(params, options, verbose=False):
    if not verbose:
        return ''

    fixed = {attr: val[0] for (attr, val) in options.attrs.items()}
    attrs = options.attrs
    s = 'base/'
    for attr in attrs:
        if params[attr] != fixed[attr]:
            s = f'change_{attr}/'
    return s

