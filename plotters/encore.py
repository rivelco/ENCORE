import numpy as np
import plotters.encore_plots as encore_plots
import math
import matplotlib.pyplot as plt

def plot_svd(figures, answer):
    """
    Plots and saves the results of the SVD algorithm, including similarity maps, singular values, 
    components, and ensemble timecourses.

    :param answer: Dictionary containing the SVD output data from MATLAB, including matrices for 
                similarity maps, singular values, component vectors, ensemble timecourses, 
                and neuron groupings.
    :type answer: dict
    :return: None
    :rtype: None
    """
    # Similarity map
    simmap = np.array(answer['S_index_ti'])
    plot_widget = figures.get('svd_plot_similaritymap')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='jet', aspect='equal')
    
    # Binary similarity map
    bin_simmap = np.array(answer['S_indexp'])
    plot_widget = figures.get('svd_plot_binarysimmap')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, bin_simmap, xlabel="Significant population vector", ylabel="Significant population vector", cmap='gray', aspect='equal')
    
    # Singular values plot
    singular_vals = np.diagonal(np.array(answer['S_svd']))
    num_state = int(answer['num_state'])
    plot_widget = figures.get('svd_plot_singularvalues')
    if plot_widget:
        encore_plots.plot_singular_values(plot_widget, singular_vals, num_state)

    # Components from the decomposition
    singular_vals = np.array(answer['svd_sig'])
    plot_widget = figures.get('svd_plot_components')
    if plot_widget:
        rows = math.ceil(math.sqrt(num_state))
        cols = math.ceil(num_state / rows)
        rows = rows+1 if rows == 1 else rows
        cols = cols+1 if cols == 1 else cols
        plot_widget.set_subplots(rows, cols)
        plot_widget.canvas.setFixedHeight(400*rows)
        for state_idx in range(num_state):
            curent_comp = singular_vals[:, :, state_idx]
            row = state_idx // cols
            col = state_idx % cols
            encore_plots.plot_states_from_svd(plot_widget, curent_comp, state_idx, row, col)
        
    # Plot the ensembles timecourse
    ensembles_timecourse = answer['timecourse']
    plot_widget = figures.get('svd_plot_timecourse')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, ensembles_timecourse)

    # Plot neurons in ensembles
    neurons_in_ensembles = answer['neus_in_ens']
    plot_widget = figures.get('svd_plot_cellsinens')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, neurons_in_ensembles, xlabel="Cell")
        
def plot_pca(figures, answer):
    """
    Plots the results of the PCA algorithm, including eigen values, principal components, 
    rho and delta values, correlation of cells, core cells and ensembles time course.

    :param answer: Dictionary containing the PCA output data from MATLAB, including
                    eigen values, principal components, rho and delta values, correlation of cells, 
                    core cells and ensembles time course.
    :type answer: dict
    :return: None
    :rtype: None
    """
    ## Plot the eigs
    eigs = np.array(answer['exp_var'])
    plot_widget = figures.get('pca_plot_eigs')
    if plot_widget:
        encore_plots.plot_eigs(plot_widget, eigs, answer['seleig'])

    # Plot the PCA
    pcs = np.array(answer['pcs'])
    labels = np.array(answer['labels'])
    labels = labels[0] if len(labels) else None
    Nens = int(answer['Nens'])
    ens_cols = plt.cm.tab10(range(Nens * 2))
    plot_widget = figures.get('pca_plot_pca')
    try:
        if plot_widget:
            encore_plots.plot_pca(plot_widget, pcs, ens_labs=labels, ens_cols = ens_cols)
    except:
        raise RuntimeError("Error plotting the PCA. Check the other plots and console for more info.")

    # Plot the rhos vs deltas
    rho = np.array(answer['rho'])
    delta = np.array(answer['delta'])
    cents = np.array(answer['cents'])
    predbounds = np.array(answer['predbounds'])
    plot_widget = figures.get('pca_plot_rhodelta')
    if plot_widget:
        encore_plots.plot_delta_rho(plot_widget, rho, delta, cents, predbounds, ens_cols)
    
    # Plot corr(n,e)
    try:
        ens_cel_corr = np.array(answer['ens_cel_corr'])
        ens_cel_corr_min = np.min(ens_cel_corr)
        ens_cel_corr_max = np.max(ens_cel_corr)
        plot_widget = figures.get('pca_plot_corrne')
        if plot_widget:
            encore_plots.plot_core_cells(plot_widget, ens_cel_corr, [ens_cel_corr_min, ens_cel_corr_max])
    except:
        raise RuntimeError("Error plotting the correlation of cells vs ensembles. Check the other plots and console for more info.")

    # Plot core cells
    core_cells = np.array(answer['core_cells'])
    plot_widget = figures.get('pca_plot_corecells')
    if plot_widget:
        encore_plots.plot_core_cells(plot_widget, core_cells, [-1, 1])

    # Plot core cells
    try:
        ens_corr = np.array(answer["ens_corr"])[0]
        corr_thr = np.array(answer["corr_thr"])
        plot_widget = figures.get('pca_plot_innerens')
        if plot_widget:
            encore_plots.plot_ens_corr(plot_widget, ens_corr, corr_thr, ens_cols)
    except:
        raise RuntimeError("Error plotting the core cells. Check the other plots and console for more info.")

    # Plot ensembles timecourse
    plot_widget = figures.get('pca_plot_timecourse')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, np.array(answer["sel_ensmat_out"]))

    plot_widget = figures.get('pca_plot_cellsinens')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, np.array(answer["sel_core_cells"]).T)
        
def plot_ica(figures, answer):
    """
    Plots the results of the ICA algorithm, including assembly templates, time projections, 
    binary assembly templates, core cells and binary assemblies.

    :param answer: Dictionary containing the ICA output data from MATLAB, including
                    assembly templates, time projections, binary assembly templates, 
                    and binary assemblies.
    :type answer: dict
    :return: None
    :rtype: None
    """
    # Plot the assembly templates
    plot_widget = figures.get('ica_plot_assemblys')
    if plot_widget:
        plot_widget.set_subplots(answer['assembly_templates'].shape[0], 1)
        total_assemblies = answer['assembly_templates'].shape[0]
        for e_idx, ens in enumerate(answer['assembly_templates']):
            plot_xaxis = e_idx == total_assemblies-1
            encore_plots.plot_assembly_patterns(plot_widget, ens, e_idx, title=f"Ensemble {e_idx+1}", plot_xaxis=plot_xaxis)

    # Plot the time projection
    plot_widget = figures.get('ica_plot_activity')
    if plot_widget:
        encore_plots.plot_cell_assemblies_activity(plot_widget, answer['time_projection'])

    # Plot binary assembly templates
    plot_widget = figures.get('ica_plot_binary_patterns')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, answer['binary_assembly_templates'], xlabel="Cell")

    plot_widget = figures.get('ica_plot_binary_assemblies')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, answer['binary_time_projection'], xlabel="Timepoint")
        
def plot_x2p(figures, answer):
    """
    Plots the results of the X2P algorithm, including similarity map, EPI, 
    onsemble activity, offsemble activity, activity, onsembles neurons and offsemble neurons.

    :param answer: Dictionary containing the X2P output data from MATLAB, including similaty map,
                    EPI, onsemble activity, offsemble activity, activity, onnsembles neurons
                    and offsemble neurons.
    :type answer: dict
    :return: None
    :rtype: None
    """
    # Similarity map
    dataset = answer['similarity']
    plot_widget = figures.get('x2p_plot_similarity')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, dataset, xlabel="Vector #", ylabel="Vector #", cmap='jet', aspect='equal')
    # EPI
    dataset = answer['EPI']
    plot_widget = figures.get('x2p_plot_epi')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, dataset, xlabel="Neuron", ylabel="Ensemble", cmap='jet')
    # Onsemble activity
    dataset = answer['OnsembleActivity']
    plot_widget = figures.get('x2p_plot_onsemact')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, dataset, xlabel="Timepoint", ylabel="Ensemble", cmap='jet')
    # Offsemble activity
    dataset = answer['OffsembleActivity']
    plot_widget = figures.get('x2p_plot_offsemact')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, dataset, xlabel="Timepoint", ylabel="Ensemble", cmap='jet')
    # Activity
    dataset = answer['Activity']
    plot_widget = figures.get('x2p_plot_activity')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, dataset)
    # Onsemble neurons
    dataset = answer['OnsembleNeurons']
    plot_widget = figures.get('x2p_plot_onsemneu')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, dataset, xlabel="Cell")
    # Offsemble neurons
    dataset = answer['OffsembleNeurons']
    plot_widget = figures.get('x2p_plot_offsemneu')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, dataset, xlabel="Cell")

def plot_sgc(figures, answer):
    ensembles_timecourse = answer['timecourse']
    neurons_in_ensembles = answer['neus_in_ens']

    # Plot the cells in ensembles
    plot_widget = figures.get('sgc_plot_timecourse')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, neurons_in_ensembles, xlabel="Cell")
    # Plot the ensmbles activations
    plot_widget = figures.get('sgc_plot_cellsinens')
    if plot_widget:
        encore_plots.plot_ensembles_timecourse(plot_widget, ensembles_timecourse, xlabel="Timepoint")
        
def plot_example(figures, answer):
    raster = answer['raster']
    dFFo_single_neuron = answer['neuron_dFFo']
    secondary_line = answer['other_dFFo']
    many_dFFo = answer['many_neurons_dFFo']
    
    # Figure 1
    # Plot the raster using an already defined plotting function
    plot_widget = figures.get('example_plot_raster')
    if plot_widget:
        encore_plots.preview_dataset(plot_widget, raster, xlabel="Timepoint", ylabel="Ensembles")
    
    # Figure 2
    # Plot a line using a function in 'encore_plots'
    plot_widget = figures.get('example_plot_dFFo')
    if plot_widget:
        encore_plots.plot_for_example_simple_line(plot_widget, dFFo_single_neuron)
    
    # Figure 3
    # Create the same plot but using the functions here
    plot_widget = figures.get('example_plot_secondary_dFFo')
    if plot_widget:
        plot_widget.axes.clear() # Clear the figure
        # The plot_widget.axes is a waraper for Matplolib axes, use it the same
        plot_widget.axes.plot(dFFo_single_neuron)
        plot_widget.axes.set_xlabel('Time (timepoint)')
        plot_widget.axes.set_ylabel('dFFo')
        # Finish the figure
        plot_widget.canvas.figure.tight_layout()
        plot_widget.canvas.draw()
        plot_widget.canvas.flush_events()
    
    # Figure 4
    # Create subplots and plot one line on each subplot
    plot_widget = figures.get('example_plot_many_dFFo')
    if plot_widget:
        plot_widget.set_subplots(4, 1)  # Figure with 4 rows and 1 column
        plot_widget.canvas.setFixedHeight(1200) # Set the figure size to 1200px
        
        colors = ['red', 'blue', 'orange', 'pink']
        widths = [0.5, 2, 1.5, 1]
        for idx in range(many_dFFo.shape[0]):
            plot_widget.axes[idx].clear()
            plot_widget.axes[idx].plot(many_dFFo[idx,:], color=colors[idx], linewidth=widths[idx])
            
        plot_widget.canvas.figure.tight_layout()
        plot_widget.canvas.draw()
        plot_widget.canvas.flush_events()
    