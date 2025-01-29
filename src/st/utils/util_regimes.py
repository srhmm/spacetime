import random
import numpy as np
from itertools import combinations

import Rbeast as rb
import networkx as nx
import ruptures as rpt
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from scipy.optimize import linear_sum_assignment


def generate_cuts(T, n, min_dur=100, max_attempts=100):
    """
    Cur a range (0, T) into n bins of minimal width min_dur
    :param T: Length of the range
    :param n: Number of bins
    :param min_dur: Minimal bin width
    :param max_attempts: Maximal number of attempts to obtain the correct minimal width
    :return: The cutpoints, a flag indicating the success or failure of the task
    """
    attempts = 0
    durs = [False]
    while not all(durs) and attempts < max_attempts:
        cuts = [0] + sorted(random.sample(range(T), n - 1)) + [T]
        durs = [(cuts[i+1] - cuts[i]) >= min_dur for i in range(len(cuts)-1)]
        attempts += 1
    return cuts, all(durs)


def generate_seq(R, n):
    """
    Generate a sequence of length n from R symbols with each symbol at least once and no consecutive repetition
    :param R: Number of different symbols
    :param n: Length of the sequence
    :return:
    """
    assert n >= R
    seq = list(range(R))
    random.shuffle(seq)
    for i in range(R, n):
        choices = set(range(R))
        seq.append(random.choice(list(choices - {seq[i - 1]})))
    return seq


def r_partition_to_windows_T(r_partition, skip):
    """
    Convert an r_partition into a windows_T
    :param r_partition: [(begin, duration, regime)]
    :param skip:
    :return: [(begin, end)]
    """
    return [(b + skip, b + d) for (b, d, r) in r_partition]


def windows_T_to_r_partition(windows_T, skip, regimes=None):
    """
    Convert a windows_T (and a list of ordered regimes) into an r_partition
    :param windows_T: [(begin, end)]
    :param skip:
    :param regimes: list of ordered regimes (optional)
    :return: [(begin, duration, regime)]
    """
    if regimes is None: regimes = [None] * len(windows_T)
    return [(b-skip, windows_T[i+1][0]-b, regimes[i]) if i < len(windows_T)-1 else (b-skip, e-b+skip, regimes[i]) for i, (b, e) in enumerate(windows_T)]


def cuts_to_windows_T(cuts, skip):
    """
    Convert a list of cutpoints into a windows_T
    :param cuts: list of ordered cutpoints
    :param skip:
    :return: [(begin, end)]
    """
    return [(cuts[i]+skip, cuts[i+1]-1) if i < len(cuts)-2 else (cuts[i]+skip, cuts[i+1]) for i in range(len(cuts) - 1)]


def partition_t(T, R, n, min_dur=100, equal_dur=True):
    """
    Generate a partition of n chunks over R different regimes and T datapoints
    :param T: Total length of the time series
    :param R: Number of different regimes
    :param n: Number of chunks
    :param min_dur: Minimal duration of the chunks
    :param equal_dur: If all chunks should have the same length (except the last)
    :return:
    """
    success = True
    if not equal_dur:
        cuts, success = generate_cuts(T, n, min_dur)
        p = [(cuts[i], cuts[i+1] - cuts[i], None) for i in range(len(cuts)-1)]
    if equal_dur or not success:
        dur = T//n
        p = [(i*dur, dur, None) for i in range(n-1)]
        p.append(((n-1)*dur, dur+T%n, None))
    p = [(p[i][0], p[i][1], r) for i, r in enumerate(generate_seq(R, n))]
    return p


def regimes_map_from_constraints(regimes_per_node):
    regimes = np.zeros(len(regimes_per_node[0]))
    forbidden_pairs = list()
    for p1, p2 in combinations(list(range(len(regimes_per_node[0]))), 2):
        selected_columns = [p1, p2]
        array_2d = np.array(list(regimes_per_node.values()))
        selected_array = array_2d[:, selected_columns]
        any_different_values = np.any(np.diff(selected_array, axis=1) != 0)
        if any_different_values:
            forbidden_pairs.append((p1, p2))
    G = nx.complete_graph(len(regimes_per_node[0]))
    for p1, p2 in forbidden_pairs:
        G.remove_edge(p1, p2)
    clusters = list(nx.connected_components(G))
    for i, c in enumerate(clusters): regimes[list(c)] = i
    return regimes


def bayesian_changepoint_detection(data):
    o = rb.beast(data, season="none")
    rb.plot(o)
    return o.trend.cp


def pelt(data, min_dur, plot=False):
    model = "rbf"
    algo = rpt.Pelt(model=model, min_size=min_dur, jump=1).fit(data)
    my_bkps = algo.predict(pen=3)
    if plot:
        fig, ax_arr = rpt.display(data, my_bkps, figsize=(10, 6))
        plt.show()
    return my_bkps


def moving_average(data, min_dur):
    window = np.ones(min_dur) / min_dur
    return np.convolve(data, window, mode='valid')


def precision_recall_dist_cps(true_r_partition, found_r_partition, max_dist=10):
    """
    Compute the precision and recall.
    The precision correspond to the precision of the detection.
    :param true_r_partition: (beg, dur, regime)
    :param found_r_partition: (beg, dur, regime)
    :param max_dist: maximal distance with true cutpoint to be considered as found
    """
    true_cps = [b for b, d, r in true_r_partition[1:]]
    found_cps = [b for b, d, r in found_r_partition[1:]]

    tp = 0.0
    dist = 0.0
    for cps in true_cps:
        res = [c for c in found_cps if cps - max_dist <= c <= cps + max_dist]
        if len(res) < 1: continue
        tp += 1
        dist += abs(res[0] - cps)
    if len(true_cps) == 0:
        recall, precision, dist = (1, 1, 0) if len(found_cps)==0  else (0, 0, 0)
    else:
        recall = safe_div(tp, len(true_cps))
        precision = safe_div(tp, len(found_cps))
        dist = safe_div(dist,len(true_cps))

    return precision, recall, dist


def safe_div(x,y):
    if y == 0:
        return 0.0
    return x / y


def partitions_to_labels(partitions, length):
    labels = np.zeros(length)
    for start, length, label in partitions:
        labels[start:start + length] = label
    return labels


def relabel_partitions(target_labels, result_labels):
    max_label = int(max(target_labels.max(), result_labels.max()) + 1)
    cost_matrix = np.zeros((max_label, max_label))

    for i in range(max_label):
        for j in range(max_label):
            cost_matrix[i, j] = np.sum((target_labels == i) & (result_labels == j))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    new_result_labels = np.zeros_like(result_labels)

    for i, j in zip(row_ind, col_ind):
        new_result_labels[result_labels == j] = i

    return target_labels, new_result_labels


def ari(target_partitions, result_partitions):
    # If boundary precision is important, such as in time series segmentation
    # where the exact start and end points of segments matter, ARI may be more appropriate.
    # sensible to different cluster sizes
    # Convert partitions to labels
    target_length = sum([length for _, length, _ in target_partitions])
    result_length = sum([length for _, length, _ in result_partitions])
    max_length = max(target_length, result_length)

    target_labels = partitions_to_labels(target_partitions, max_length)
    result_labels = partitions_to_labels(result_partitions, max_length)

    # Relabel result partitions to match target partitions
    target_labels, result_labels = relabel_partitions(target_labels, result_labels)

    # Compute ARI
    ari_score = adjusted_rand_score(target_labels, result_labels)
    return ari_score


def nmi(target_partitions, result_partitions):
    # If label consistency is more important, such as in applications where the overall clustering pattern matters
    # more than exact boundaries, NMI may be a better choice.
    # good for when ground truth clustering is unbalanced and there exist small clusters
    # AMI is high when there are pure clusters in the clustering solution.
    # symmetric & favor pure clusters --> only ground truth clusters should matter :(
    # Convert partitions to labels
    target_length = sum([length for _, length, _ in target_partitions])
    result_length = sum([length for _, length, _ in result_partitions])
    max_length = max(target_length, result_length)

    target_labels = partitions_to_labels(target_partitions, max_length)
    result_labels = partitions_to_labels(result_partitions, max_length)

    target_labels, result_labels = relabel_partitions(target_labels, result_labels) # useless

    # Compute NMI
    nmi_score = normalized_mutual_info_score(target_labels, result_labels)
    nmi_score_adjusted = adjusted_mutual_info_score(target_labels, result_labels)
    return nmi_score_adjusted


def confusion_matrix(original_partitions, found_partitions):
    # Calculate the length of the label arrays
    target_length = sum([length for _, length, _ in original_partitions])
    result_length = sum([length for _, length, _ in found_partitions])
    max_length = max(target_length, result_length)

    # Convert partitions to labels
    target_labels = partitions_to_labels(original_partitions, max_length)
    result_labels = partitions_to_labels(found_partitions, max_length)

    # Construct confusion matrix
    conf_matrix = confusion_matrix(target_labels, result_labels)

    print("Confusion Matrix:")
    print(conf_matrix)
    return conf_matrix


def v_mesure(target_partitions, result_partitions):
    # symmetric
    target_length = sum([length for _, length, _ in target_partitions])
    result_length = sum([length for _, length, _ in result_partitions])
    max_length = max(target_length, result_length)

    target_labels = partitions_to_labels(target_partitions, max_length)
    result_labels = partitions_to_labels(result_partitions, max_length)

    target_labels, result_labels = relabel_partitions(target_labels, result_labels) # useless

    homogeneity_score(target_labels, result_labels)
    completeness_score(target_labels, result_labels)
    v_measure_score(target_labels, result_labels)


def fmi(target_partitions, result_partitions):
    # symmetric
    target_length = sum([length for _, length, _ in target_partitions])
    result_length = sum([length for _, length, _ in result_partitions])
    max_length = max(target_length, result_length)

    target_labels = partitions_to_labels(target_partitions, max_length)
    result_labels = partitions_to_labels(result_partitions, max_length)

    target_labels, result_labels = relabel_partitions(target_labels, result_labels)

    fowlkes_mallows_score(target_labels, result_labels)


