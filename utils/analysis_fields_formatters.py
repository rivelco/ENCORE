MAX_INTEGER = 2147483647

def initialize_svd_fields(widgets, defaults):
    """This function sets the limit for the input widgets of the SVD analysis and 
    sets the default value of the widget.

    Args:
        widgets (dict): Directory of the widgets for the analysis.
        defaults (dict): Directory with the default values for each widget
    """
    
    widgets['svd_edit_pks'].setRange(0, MAX_INTEGER)
    widgets['svd_edit_pks'].setValue(defaults['pks'])

    widgets['svd_edit_scut'].setRange(0.00001, 1)
    widgets['svd_edit_scut'].setValue(defaults['scut'])

    widgets['svd_edit_hcut'].setRange(0.00001, 1)
    widgets['svd_edit_hcut'].setValue(defaults['hcut'])

    widgets['svd_edit_statecut'].setRange(1, MAX_INTEGER)
    widgets['svd_edit_statecut'].setValue(defaults['state_cut'])

    widgets['svd_edit_csistart'].setRange(0, MAX_INTEGER)
    widgets['svd_edit_csistart'].setValue(defaults['csi_start'])

    widgets['svd_edit_csistep'].setRange(0, MAX_INTEGER)
    widgets['svd_edit_csistep'].setValue(defaults['csi_step'])

    widgets['svd_edit_csiend'].setRange(0, MAX_INTEGER)
    widgets['svd_edit_csiend'].setValue(defaults['csi_end'])

def initialize_pca_fields(widgets, defaults):
    """This function sets the limit for the input widgets of the PCA analysis and 
    sets the default value of the widget.

    Args:
        widgets (dict): Directory of the widgets for the analysis.
        defaults (dict): Directory with the default values for each widget
    """
    
    widgets['pca_edit_dc'].setRange(0.000001, MAX_INTEGER)
    widgets['pca_edit_dc'].setValue(defaults['dc'])
    
    widgets['pca_edit_npcs'].setRange(1, MAX_INTEGER)
    widgets['pca_edit_npcs'].setValue(defaults['npcs'])
    
    widgets['pca_edit_minspk'].setRange(1, MAX_INTEGER)
    widgets['pca_edit_minspk'].setValue(defaults['minspk'])
    
    widgets['pca_edit_nsur'].setRange(1, MAX_INTEGER)
    widgets['pca_edit_nsur'].setValue(defaults['nsur'])
    
    widgets['pca_edit_prct'].setRange(0.000001, 100)
    widgets['pca_edit_prct'].setValue(defaults['prct'])
    
    widgets['pca_edit_centthr'].setRange(0.000001, 100)
    widgets['pca_edit_centthr'].setValue(defaults['cent_thr'])
    
    widgets['pca_edit_innercorr'].setRange(0.01, MAX_INTEGER)
    widgets['pca_edit_innercorr'].setValue(defaults['inner_corr'])
    
    widgets['pca_edit_minsize'].setRange(1, MAX_INTEGER)
    widgets['pca_edit_minsize'].setValue(defaults['minsize'])

def initialize_ica_fields(widgets, defaults):
    """This function sets the limit for the input widgets of the ICA analysis and 
    sets the default value of the widget.

    Args:
        widgets (dict): Directory of the widgets for the analysis.
        defaults (dict): Directory with the default values for each widget
    """
    
    widgets['ica_edit_perpercentile'].setRange(0.01, 100.0)
    widgets['ica_edit_perpercentile'].setValue(defaults['threshold']['permutations_percentile'])
    
    widgets['ica_edit_percant'].setRange(1, MAX_INTEGER)
    widgets['ica_edit_percant'].setValue(defaults['threshold']['number_of_permutations'])
    
    widgets['ica_edit_iterations'].setRange(1, MAX_INTEGER)
    widgets['ica_edit_iterations'].setValue(defaults['Patterns']['number_of_iterations'])
    
def initialize_x2p_fields(widgets, defaults):
    """This function sets the limit for the input widgets of the X2P analysis and 
    sets the default value of the widget.

    Args:
        widgets (dict): Directory of the widgets for the analysis.
        defaults (dict): Directory with the default values for each widget
    """
    
    widgets['x2p_edit_bin'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_bin'].setValue(defaults['network_bin'])

    widgets['x2p_edit_iterations'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_iterations'].setValue(defaults['network_iterations'])

    widgets['x2p_edit_significance'].setRange(0.00001, MAX_INTEGER)
    widgets['x2p_edit_significance'].setValue(defaults['network_significance'])

    widgets['x2p_edit_threshold'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_threshold'].setValue(defaults['coactive_neurons_threshold'])

    widgets['x2p_edit_rangestart'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_rangestart'].setValue(defaults['clustering_range_start'])

    widgets['x2p_edit_rangeend'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_rangeend'].setValue(defaults['clustering_range_end'])

    widgets['x2p_edit_fixed'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_fixed'].setValue(defaults['clustering_fixed'])

    widgets['x2p_edit_itensemble'].setRange(1, MAX_INTEGER)
    widgets['x2p_edit_itensemble'].setValue(defaults['iterations_ensemble'])

def initialize_sgc_fields(widgets, defaults):
    """This function sets the limit for the input widgets of the SGC analysis and 
    sets the default value of the widget.

    Args:
        widgets (dict): Directory of the widgets for the analysis.
        defaults (dict): Directory with the default values for each widget
    """
    
    widgets['sgc_edit_stdthreshold'].setRange(0.0001, MAX_INTEGER)
    widgets['sgc_edit_stdthreshold'].setValue(defaults['standard_deviations_threshold'])

    widgets['sgc_edit_shuff'].setRange(1, MAX_INTEGER)
    widgets['sgc_edit_shuff'].setValue(defaults['shuffling_rounds'])

    widgets['sgc_edit_sig'].setRange(0.000001, MAX_INTEGER)
    widgets['sgc_edit_sig'].setValue(defaults['coactivity_significance_level'])

    widgets['sgc_edit_monterounds'].setRange(1, MAX_INTEGER)
    widgets['sgc_edit_monterounds'].setValue(defaults['montecarlo_rounds'])

    widgets['sgc_edit_montesteps'].setRange(1, MAX_INTEGER)
    widgets['sgc_edit_montesteps'].setValue(defaults['montecarlo_steps'])

    widgets['sgc_edit_affthres'].setRange(0.000001, MAX_INTEGER)
    widgets['sgc_edit_affthres'].setValue(defaults['affinity_threshold'])

def initialize_encore_analysis_fields(gui, analysis_defaults):
    svd_widgets = {
        'svd_edit_pks': gui.svd_edit_pks,
        'svd_edit_scut': gui.svd_edit_scut,
        'svd_edit_hcut': gui.svd_edit_hcut,
        'svd_edit_statecut': gui.svd_edit_statecut,
        'svd_edit_csistart': gui.svd_edit_csistart,
        'svd_edit_csistep': gui.svd_edit_csistep,
        'svd_edit_csiend': gui.svd_edit_csiend
    }
    initialize_svd_fields(svd_widgets, analysis_defaults['svd_defaults'])
    
    pca_widgets = {
        'pca_edit_dc': gui.pca_edit_dc,
        'pca_edit_npcs': gui.pca_edit_npcs,
        'pca_edit_minspk': gui.pca_edit_minspk,
        'pca_edit_nsur': gui.pca_edit_nsur,
        'pca_edit_prct': gui.pca_edit_prct,
        'pca_edit_centthr': gui.pca_edit_centthr,
        'pca_edit_innercorr': gui.pca_edit_innercorr,
        'pca_edit_minsize': gui.pca_edit_minsize
    }
    initialize_pca_fields(pca_widgets, analysis_defaults['pca_defaults'])
    
    ica_widgets = {
        'ica_edit_perpercentile': gui.ica_edit_perpercentile,
        'ica_edit_percant': gui.ica_edit_percant,
        'ica_edit_iterations': gui.ica_edit_iterations
    }
    initialize_ica_fields(ica_widgets, analysis_defaults['ica_defaults'])
    
    x2p_widgets = {
        'x2p_edit_bin': gui.x2p_edit_bin,
        'x2p_edit_iterations': gui.x2p_edit_iterations,
        'x2p_edit_significance': gui.x2p_edit_significance,
        'x2p_edit_threshold': gui.x2p_edit_threshold,
        'x2p_edit_rangestart': gui.x2p_edit_rangestart,
        'x2p_edit_rangeend': gui.x2p_edit_rangeend,
        'x2p_edit_fixed': gui.x2p_edit_fixed,
        'x2p_edit_itensemble': gui.x2p_edit_itensemble
    }
    initialize_x2p_fields(x2p_widgets, analysis_defaults['x2p_defaults'])
    
    sgc_widgets = {
        'sgc_edit_stdthreshold': gui.sgc_edit_stdthreshold,
        'sgc_edit_shuff': gui.sgc_edit_shuff,
        'sgc_edit_sig': gui.sgc_edit_sig,
        'sgc_edit_monterounds': gui.sgc_edit_monterounds,
        'sgc_edit_montesteps': gui.sgc_edit_montesteps,
        'sgc_edit_affthres': gui.sgc_edit_affthres
    }
    initialize_sgc_fields(sgc_widgets, analysis_defaults['sgc_defaults'])