import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_correlation_with_stimuli(ensembles_timecourse, data_stims):
    correlation = np.zeros((ensembles_timecourse.shape[0], data_stims.shape[0]))
    for e_idx, ensemble in enumerate(ensembles_timecourse):
        for s_idx, stim in enumerate(data_stims):
            similarity = np.corrcoef(ensemble, stim)[0, 1]
            correlation[e_idx][s_idx] =  similarity
    return correlation

# Function to calculate neuron overlap between two methods
def calculate_neuron_overlap_ratio(m1_neus_in_ens, m2_neus_in_ens):
    overlaps = np.zeros((m1_neus_in_ens.shape[0], m2_neus_in_ens.shape[0]))
    for e1_idx, ensemble1 in enumerate(m1_neus_in_ens):
        for e2_idx, ensemble2 in enumerate(m2_neus_in_ens):
            overlap = np.sum(ensemble1 & ensemble2) / np.sum(ensemble1 | ensemble2)
            overlaps[e1_idx, e2_idx] = overlap
    return overlaps

def calculate_neuron_overlap_shared(m1_neus_in_ens, m2_neus_in_ens):
    shared = np.zeros((m1_neus_in_ens.shape[0], m2_neus_in_ens.shape[0]))
    for e1_idx, ensemble1 in enumerate(m1_neus_in_ens):
        for e2_idx, ensemble2 in enumerate(m2_neus_in_ens):
            overlap = np.sum(ensemble1 & ensemble2)
            shared[e1_idx, e2_idx] = overlap
    return shared

def compute_correlation_inside_ensemble(activity_neus_in_ens):
    correlation = np.corrcoef(activity_neus_in_ens)
    return correlation

def compute_correlation_between_ensembles(ensembles_timecourse):
    correlation = np.corrcoef(ensembles_timecourse)
    return correlation

def compute_auc_roc_ensemble_stimuli(ensemble_timecourse, stimuli):
    fpr, tpr, thresholds = roc_curve(stimuli, ensemble_timecourse)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc

def compute_cross_correlations(ensemble_timecourse, stimuli):
    cross_correlation = np.correlate(ensemble_timecourse, stimuli, mode='full')
    lags = np.arange(-len(ensemble_timecourse) + 1, len(stimuli))
    return cross_correlation, lags