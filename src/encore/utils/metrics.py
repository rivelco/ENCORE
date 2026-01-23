import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_correlation_with_stimuli(ensembles_timecourse, data_stims):
    """
    Compute Pearson correlations between ensemble timecourses and stimuli.

    :param ensembles_timecourse: Ensemble activity array (ensembles, timepoints).
    :type ensembles_timecourse: numpy.ndarray
    :param data_stims: Stimulus timecourses (stimuli, timepoints).
    :type data_stims: numpy.ndarray
    :return: Correlation matrix (ensembles, stimuli).
    :rtype: numpy.ndarray
    """
    correlation = np.zeros((ensembles_timecourse.shape[0], data_stims.shape[0]))
    for e_idx, ensemble in enumerate(ensembles_timecourse):
        for s_idx, stim in enumerate(data_stims):
            similarity = np.corrcoef(ensemble, stim)[0, 1]
            correlation[e_idx][s_idx] =  similarity
    return correlation

# Function to calculate neuron overlap between two methods
def calculate_neuron_overlap_ratio(m1_neus_in_ens, m2_neus_in_ens):
    """
    Compute pairwise neuron overlap ratio between two ensemble sets.

    :param m1_neus_in_ens: Binary neuron membership for method 1.
    :type m1_neus_in_ens: numpy.ndarray
    :param m2_neus_in_ens: Binary neuron membership for method 2.
    :type m2_neus_in_ens: numpy.ndarray
    :return: Overlap ratio matrix.
    :rtype: numpy.ndarray
    """
    overlaps = np.zeros((m1_neus_in_ens.shape[0], m2_neus_in_ens.shape[0]))
    for e1_idx, ensemble1 in enumerate(m1_neus_in_ens):
        for e2_idx, ensemble2 in enumerate(m2_neus_in_ens):
            overlap = np.sum(ensemble1 & ensemble2) / np.sum(ensemble1 | ensemble2)
            overlaps[e1_idx, e2_idx] = overlap
    return overlaps

def calculate_neuron_overlap_shared(m1_neus_in_ens, m2_neus_in_ens):
    """
    Compute the number of shared neurons between ensembles from two methods.

    :param m1_neus_in_ens: Binary neuron membership for method 1.
    :type m1_neus_in_ens: numpy.ndarray
    :param m2_neus_in_ens: Binary neuron membership for method 2.
    :type m2_neus_in_ens: numpy.ndarray
    :return: Matrix of shared neuron counts.
    :rtype: numpy.ndarray
    """
    shared = np.zeros((m1_neus_in_ens.shape[0], m2_neus_in_ens.shape[0]))
    for e1_idx, ensemble1 in enumerate(m1_neus_in_ens):
        for e2_idx, ensemble2 in enumerate(m2_neus_in_ens):
            overlap = np.sum(ensemble1 & ensemble2)
            shared[e1_idx, e2_idx] = overlap
    return shared

def compute_correlation_inside_ensemble(activity_neus_in_ens):
    """
    Compute correlation matrix between neurons within an ensemble.

    :param activity_neus_in_ens: Neuronal activity matrix.
    :type activity_neus_in_ens: numpy.ndarray
    :return: Neuron-to-neuron correlation matrix.
    :rtype: numpy.ndarray
    """
    correlation = np.corrcoef(activity_neus_in_ens)
    return correlation

def compute_correlation_between_ensembles(ensembles_timecourse):
    """
    Compute correlation matrix between ensemble timecourses.

    :param ensembles_timecourse: Ensemble activity timecourses.
    :type ensembles_timecourse: numpy.ndarray
    :return: Ensemble-to-ensemble correlation matrix.
    :rtype: numpy.ndarray
    """
    correlation = np.corrcoef(ensembles_timecourse)
    return correlation

def compute_auc_roc_ensemble_stimuli(ensemble_timecourse, stimuli):
    """
    Compute ROC curve and AUC between ensemble activity and stimuli.

    :param ensemble_timecourse: Ensemble activity signal.
    :type ensemble_timecourse: numpy.ndarray
    :param stimuli: Binary or continuous stimulus signal.
    :type stimuli: numpy.ndarray
    :return: False positive rate, true positive rate, thresholds, and AUC.
    :rtype: tuple
    """
    fpr, tpr, thresholds = roc_curve(stimuli, ensemble_timecourse)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def compute_cross_correlations(ensemble_timecourse, stimuli):
    """
    Compute cross-correlation between ensemble activity and stimuli.

    :param ensemble_timecourse: Ensemble activity signal.
    :type ensemble_timecourse: numpy.ndarray
    :param stimuli: Stimulus signal.
    :type stimuli: numpy.ndarray
    :return: Cross-correlation values and corresponding lags.
    :rtype: tuple
    """
    cross_correlation = np.correlate(ensemble_timecourse, stimuli, mode='full')
    lags = np.arange(-len(ensemble_timecourse) + 1, len(stimuli))
    return cross_correlation, lags