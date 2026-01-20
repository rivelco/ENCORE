import numpy as np
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt

def preview_dataset(plot_widget, dataset, xlabel='Timepoint', ylabel='Data', title=None, cmap='hot', aspect='auto', yitems_labels=[]):
    plot_widget.axes.clear()
    n, t = dataset.shape
    plot_widget.axes.imshow(dataset, cmap=cmap, interpolation='nearest', aspect=aspect)
    if title != None:
        plot_widget.axes.set_title(title)
    plot_widget.axes.set_xlabel(xlabel)
    plot_widget.axes.set_ylabel(ylabel)
    plot_widget.axes.set_xlim([0, t])
    plot_widget.axes.set_ylim([-0.5, n-0.5])
    num_user_labels = len(yitems_labels)
    if num_user_labels > 0:
        plot_widget.axes.set_yticks(range(n))
        if num_user_labels == n:
            plot_widget.axes.set_yticklabels(yitems_labels)
        elif n > num_user_labels:
            new_labels = list(range(num_user_labels, n))
            new_labels = yitems_labels + new_labels
            plot_widget.axes.set_yticklabels(new_labels)
        elif n < num_user_labels:
            new_labels = yitems_labels[:n]
            plot_widget.axes.set_yticklabels(new_labels)
            
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def preview_coordinates2D(plot_widget, dataset):
    plot_widget.axes.clear()
    plot_widget.axes.scatter(dataset[:,0], dataset[:,1], c='blue', marker='o')
    
    plot_widget.axes.set_xlabel('X coordinates')
    plot_widget.axes.set_ylabel('Y coordinates')
    min_x = int(min(dataset[:,0])) - 10
    max_x = int(max(dataset[:,0])) + 10
    min_y = int(min(dataset[:,1])) - 10
    max_y = int(max(dataset[:,1])) + 10
    plot_widget.axes.set_ylim([min_y, max_x])
    plot_widget.axes.set_xlim([min_x, max_y])
    #plot_widget.axes.set_xticks([])
    #plot_widget.axes.set_yticks([])
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.axes.set_aspect('equal', adjustable='box')
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def raster_plot(plot_widget, data_neuronal_activity):
    plot_widget.axes.clear()
    
    n, t = data_neuronal_activity.shape
    for neuron in range(n):
        spike_times = np.where(data_neuronal_activity[neuron] == 1)[0]
        plot_widget.axes.vlines(spike_times, neuron + 0.5, neuron + 1.5, color='black')

    plot_widget.axes.set_xlim([0, t])
    plot_widget.axes.set_ylim([0, n+1.5])
    plot_widget.axes.set_yticklabels([])
    plot_widget.axes.set_xticklabels([])
    plot_widget.axes.set_xticks([])
    plot_widget.axes.set_yticks([])
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.axes.set_xlabel("Time")
    plot_widget.axes.set_ylabel("Neuron")

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

# Plots for the SVD analysis
def plot_singular_values(plot_widget, singulars, num_states):
    plot_widget.axes.clear()
    plot_widget.axes.semilogx(singulars, marker='o', linestyle='-', label='Singular values')
    # Add a vertical red dashed line at num_state
    plot_widget.axes.plot(num_states, singulars[num_states], 'rs', markersize=10, label=f"Cutoff at {num_states}")
    #plot_widget.axes.axvline(x=num_states, color='r', linestyle='--', linewidth=2, label=f'Cutoff at {num_states}')
    # Label the axes
    plot_widget.axes.set_xlabel('log(singular value)')
    plot_widget.axes.set_ylabel('Magnitude')
    # Set the title
    plot_widget.axes.set_title('Singular values')
    plot_widget.axes.legend()
    for side in ['top', 'right']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_states_from_svd(plot_widget, svd_sig, comp_n, row, col):
    plot_widget.axes[row][col].clear()
    # Plot the image, where svd_sig[:,:,n]==0 is the condition to be checked
    plot_widget.axes[row][col].imshow(svd_sig == 0, cmap='gray', aspect='equal')
    # Set the labels and title
    plot_widget.axes[row][col].set_xlabel('Population vector')
    plot_widget.axes[row][col].set_ylabel('Population vector')
    plot_widget.axes[row][col].set_title(f'Components ensemble {comp_n+1}')
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_ensembles_timecourse(plot_widget, timecourse, xlabel="Timepoint"):
    plot_widget.axes.clear()
    ensembles_cant = timecourse.shape[0]
    frames_cant = timecourse.shape[1]
    for ens in range(ensembles_cant):
        for frame in range(frames_cant):
            if timecourse[ens, frame]:
                plot_widget.axes.plot(frame+1, ens+1, '|', markerfacecolor='none', markeredgecolor='k', markersize=15)

    plot_widget.axes.set_xlabel(xlabel)                # Show the frame label
    plot_widget.axes.spines['top'].set_visible(False)   # Hide the top line for everyone except the first
    plot_widget.axes.spines['right'].set_visible(False) # Hide the rigth line for everyone except the first
    plot_widget.axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Show only integers in the y axis
    if ensembles_cant == 0:
        ensembles_cant = 1
    plot_widget.axes.set_ylim([0.5, ensembles_cant+0.5])       # Set the y axis limit
    plot_widget.axes.set_xlim([0, frames_cant]) # Set the x axis limit
    plot_widget.axes.set_ylabel('Ensemble') 

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_ens_seq(plot_widget, tt, ens_labs, ens_cols, sellabs):
    plot_widget.axes.clear()
    nens = np.max(ens_labs)
    try:
        if nens == 0:
            plot_widget.axes.plot([tt[0], tt[-1]], [0, 0], 'k-')
            return
        else:
            plot_widget.axes.plot(tt, ens_labs, 'k--')
            plot_widget.axes.plot(tt[ens_labs == 0], ens_labs[ens_labs == 0], 'ko')
            for e in range(1, nens + 1):
                plot_widget.axes.plot(tt[ens_labs == e], ens_labs[ens_labs == e], '.', markersize=25, color=ens_cols[e-1])
            plot_widget.axes.set_ylim([0, nens * 1.1])
    except Exception as e:
        print(f"Error occurred: {e}")
        plot_widget.axes.plot([tt[0], tt[-1]], [0, 0], 'k-')

    plot_widget.axes.set_xlabel('Time (s)')
    plot_widget.axes.set_ylabel('Ensemble')
    plot_widget.axes.set_yticks(range(1, nens + 1))
    plot_widget.axes.set_yticklabels(sellabs)
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

# Plots for the PCA 
def plot_eigs(plot_widget, eigs, seleig):
    plot_widget.axes.clear()

    plot_widget.axes.plot(np.arange(1, len(eigs) + 1), eigs, 'o--', label="Principal components")
    plot_widget.axes.plot(seleig, eigs[seleig - 1], 'rs', markersize=10, label=f"Cutoff at {seleig}")
    plot_widget.axes.set_xscale('log')
    plot_widget.axes.set_yscale('linear')
    plot_widget.axes.set_xlim(1, len(eigs))
    plot_widget.axes.set_xlabel('Principal components')
    plot_widget.axes.set_ylabel('% var')
    plot_widget.axes.legend()

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_pca(plot_widget, pcs, ens_labs=None, ens_cols=None):
    plot_widget.axes.clear()
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.axes.set_yticks([])
    plot_widget.axes.set_xticks([])
    np = pcs.shape[1]
    if ens_labs is None or ens_cols is None:  # plots raster with no color
        if np > 2:
            plot_widget.axes = plot_widget.canvas.figure.add_subplot(111, projection='3d')
            plot_widget.axes.plot3D(pcs[:, 0], pcs[:, 1], pcs[:, 2], 'k.')
        else:
            plot_widget.axes = plot_widget.canvas.figure.add_subplot(111)
            plot_widget.axes.plot(pcs[:, 0], pcs[:, 1], 'k.')
        plot_widget.axes.set_xlabel('PC1')
        plot_widget.axes.set_ylabel('PC2')
        if np > 2:
            plot_widget.axes.set_zlabel('PC3')
        plot_widget.axes.grid(True)
    else:
        nens = int(max(ens_labs))
        if np > 2:
            plot_widget.axes = plot_widget.canvas.figure.add_subplot(111, projection='3d')
            plot_widget.axes.plot3D(pcs[ens_labs == 0, 0], pcs[ens_labs == 0, 1], pcs[ens_labs == 0, 2], 'k.')
            for e in range(1, nens + 1):
                plot_widget.axes.plot3D(pcs[ens_labs == e, 0], pcs[ens_labs == e, 1], pcs[ens_labs == e, 2],
                        marker='.', color=ens_cols[e - 1], linestyle='none')
        else:
            plot_widget.axes = plot_widget.canvas.figure.add_subplot(111)
            plot_widget.axes.plot(pcs[ens_labs == 0, 0], pcs[ens_labs == 0, 1], 'k.')
            for e in range(1, nens + 1):
                plot_widget.axes.plot(pcs[ens_labs == e, 0], pcs[ens_labs == e, 1],
                        marker='.', color=ens_cols[e - 1], linestyle='none')
        plot_widget.axes.set_xlabel('PC1')
        plot_widget.axes.set_ylabel('PC2')
        if np > 2:
            plot_widget.axes.set_zlabel('PC3')
        plot_widget.axes.grid(True)

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_delta_rho(plot_widget, rho, delta, cents, predbounds, ens_cols):
    plot_widget.axes.clear()
    # Sort pred_id based on the first column of predbounds
    pred_id = np.argsort(predbounds[:, 0])
    # Plot pred bounds
    plot_widget.axes.plot(predbounds[pred_id, 0], predbounds[pred_id, 1], 'b--', linewidth=2)
    # Plot delta rho
    plot_widget.axes.plot(rho, delta, 'k.')
    # Plot points based on the cluster centers (cents)
    nens = np.sum(cents > 0)
    for e in range(1, nens + 1):
        plot_widget.axes.plot(rho[cents == e], delta[cents == e], '.', markersize=25, color=ens_cols[e - 1])
        for r, d in zip(rho[cents == e], delta[cents == e]):
            plot_widget.axes.text(r * 1.01, d * 1.01, str(e), fontsize=12)
    # Set x and y limits
    plot_widget.axes.set_xlim([0, np.max(rho[~np.isinf(rho)]) * 1.1])
    plot_widget.axes.set_ylim([0, np.max(delta[~np.isinf(delta)]) * 1.1])
    # Set labels
    plot_widget.axes.set_xlabel(r'$\rho$')
    plot_widget.axes.set_ylabel(r'$\delta$')
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_core_cells(plot_widget, core_cells, clims):
    # Clear the axes
    plot_widget.axes.clear()
    # Determine the size of core_cells
    N, nens = core_cells.shape
    # Plot the image
    cax = plot_widget.axes.imshow(core_cells, aspect='auto', cmap='bwr', vmin=clims[0], vmax=clims[1])
    # Set the color limits
    cax.set_clim(clims)
    # Set axis limits
    plot_widget.axes.set_xlim([-0.5, nens-0.5])
    plot_widget.axes.set_ylim([0, N])
    # Draw the lines
    for e in range(nens + 1):
        plot_widget.axes.plot([e - 0.5, e - 0.5], [0, N], 'k-')
    # Remove y-axis labels and ticks
    plot_widget.axes.set_yticks([])
    plot_widget.axes.set_yticklabels([])
    plot_widget.axes.set_xticks(range(0, nens))
    plot_widget.axes.set_xticklabels(range(1, nens+1))
    # Set y-axis label
    plot_widget.axes.set_ylabel('Cells')
    plot_widget.axes.set_xlabel('Ensemble')
    # Add a box around the plot
    plot_widget.axes.spines['top'].set_visible(True)
    plot_widget.axes.spines['right'].set_visible(True)
    plot_widget.axes.spines['bottom'].set_visible(True)
    plot_widget.axes.spines['left'].set_visible(True)
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_ens_corr(plot_widget, ens_corr, corr_thr, ens_cols):
    # Clear the axes
    plot_widget.axes.clear()
    nens = len(ens_corr)
    # Plot bars
    for e in range(nens):
        plot_widget.axes.bar(e + 1, ens_corr[e], color=ens_cols[e], edgecolor=ens_cols[e])
    # Plot threshold line
    plot_widget.axes.plot([0.5, nens + 0.5], [corr_thr, corr_thr], 'r--')
    # Set x and y limits
    plot_widget.axes.set_xlim([0.5, nens + 0.5])
    plot_widget.axes.set_ylim([0, max(max(ens_corr), corr_thr) * 1.1])
    # Set labels
    plot_widget.axes.set_xlabel('Ensemble Id.')
    plot_widget.axes.set_ylabel('Core-Cells Mean Correlation')
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

# Plots for the ICA
def plot_assembly_patterns(plot_widget, Patterns, row_idx, title=None, plot_xaxis=False):
    plot_widget.axes[row_idx].clear()
    plot_widget.axes[row_idx].stem(Patterns)
    if title != None:
        plot_widget.axes[row_idx].set_title(f"{title}")

    plot_widget.axes[row_idx].set_ylabel('Weight')
    for side in ['top', 'right', 'bottom']:
        plot_widget.axes[row_idx].spines[side].set_visible(False)
    if plot_xaxis:
        plot_widget.axes[row_idx].spines['bottom'].set_visible(True)
        plot_widget.axes[row_idx].set_xlabel('Cell')
        plot_widget.canvas.figure.tight_layout()
    else:
        plot_widget.axes[row_idx].set_xticks([])
    
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_cell_assemblies_activity(plot_widget, activities):
    plot_widget.axes.clear()

    for e_idx, ensemble in enumerate(activities):
        plot_widget.axes.plot(ensemble, label=f"Ensemble {e_idx+1}")

    for side in ['top', 'right']:
        plot_widget.axes.spines[side].set_visible(False)
    plot_widget.axes.set_xlabel('Timepoint')
    plot_widget.axes.set_ylabel('Assembly activity')

    plot_widget.axes.legend()
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

# Plot the ensembles compare
def enscomp_update_map(plot_widget, lims, members_idx, members_freqs, members_coords, members_colors, neuron_size):
    plot_widget.axes.clear()

    # Plot each point as two semi-circles
    for idx in range(len(members_freqs)):
        member_freq = int(members_freqs[idx])
        member_coords_x = members_coords[0][idx]
        member_coords_y = members_coords[1][idx]
        deg_start = 0
        deg_step = 360.0/member_freq
        deg_end = deg_step
        for it in range(member_freq):
            wedge = Wedge((member_coords_x, member_coords_y), neuron_size, deg_start, deg_end, color=members_colors[idx][it])  # First half
            plot_widget.axes.add_patch(wedge)
            deg_start += deg_step
            deg_end += deg_step

    for idx in range(len(members_idx)):
        member_idx = members_idx[idx]
        member_coords_x = members_coords[0][idx]
        member_coords_y = members_coords[1][idx]
        plot_widget.axes.text(member_coords_x, member_coords_y, str(member_idx+1), fontsize=6,
                        ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round', pad=0))
    
    # Adjust plot limits
    lim_offset = 1.5*neuron_size
    plot_widget.axes.set_xlim(0-lim_offset, lims[0]+lim_offset)
    plot_widget.axes.set_ylim(0-lim_offset, lims[1]+lim_offset)
    plot_widget.axes.set_xlabel("X coordinates")
    plot_widget.axes.set_ylabel("Y coordinates")
    plot_widget.axes.set_aspect('equal')
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def enscomp_update_timelines(plot_widget, ticks, cell_activities, ensemble_dffo, ensemble_timecourse, colors, limx):
    plot_widget.axes.clear()

    # Auxiliar variable to move up the analysis timelines
    stim_or_behav = 0

    # Iterate over the indices to create bands
    for acts in range(len(ensemble_timecourse)):
        current_label = ticks[acts]
        cells_acts = ensemble_timecourse[acts]
        act_lenght = len(cells_acts)
        time_axis = range(0, act_lenght)
        if act_lenght > 0:
            if "Behav" in current_label:
                cells_acts = cells_acts/np.max(cells_acts)
                plot_widget.axes.plot(time_axis, cells_acts, color=colors[acts], alpha=1)
                stim_or_behav += 1
            else:
                band_it = 0
                while band_it < act_lenght:
                    if cells_acts[band_it] == 1:
                        start = band_it
                        band_it = band_it + 1
                        while band_it < act_lenght and cells_acts[band_it] == 1:
                            band_it = band_it + 1
                        end = band_it
                        plot_widget.axes.fill_between(time_axis[start:end], acts, acts+1, color=colors[acts], alpha=1)
                    band_it = band_it + 1
            if "Stim" in current_label:
                stim_or_behav += 1

    # Plot the neurons activitiesS
    valid_activities = []
    first_flag = True
    use_colors = []
    for method_idx in range(len(cell_activities)):
        method = cell_activities[method_idx]
        if len(method) > 0:
            if first_flag:
                valid_activities = np.array([method.copy()])
                first_flag = False
            else:
                valid_activities = np.vstack([valid_activities, method])
            use_colors.append(colors[method_idx])

    if not first_flag:
        Fmi = np.min(valid_activities)
        Fma = np.max(valid_activities)
        num_ensembles = len(cell_activities)
        for acts in range(num_ensembles):
            cells_acts = cell_activities[acts]
            if len(cells_acts) > 0:
                cant_timepoints = len(cells_acts)
                cells_acts = ((cells_acts - Fmi) / (Fma - Fmi)) + stim_or_behav
                plot_widget.axes.plot(np.arange(1, cant_timepoints + 1), acts + cells_acts, linewidth=1, color='black', alpha=0.6)
                #plot_widget.axes.text(cant_timepoints * 1.02, ii, str(cell_names[ii]+1), fontsize=8)

    plot_widget.axes.set_xlim([0, limx])
    plot_widget.axes.set_ylim([0, len(ticks)])
    plot_widget.axes.set_xlabel('Time (timepoint)')
    labels = ticks
    ticks = [tick+0.5 for tick in range(len(ticks))]
    plot_widget.axes.set_yticks(ticks = ticks, labels = labels, rotation=90, va='center')
    #plot_widget.axes.set_yticklabels(ticks)
    for side in ['left', 'top', 'right']:
        plot_widget.axes.spines[side].set_visible(False)

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def enscomp_plot_similarity(plot_widget, matrix, labels, color):
    if hasattr(plot_widget, "colorbar"):
        try:
            plot_widget.colorbar.remove()
        except:
            pass
    plot_widget.axes.clear()

    cax = plot_widget.axes.imshow(matrix, aspect='equal', cmap=color)
    plot_widget.axes.set_xticks(ticks = np.arange(len(labels)), labels=labels, rotation=45, ha="right", rotation_mode="anchor")
    plot_widget.axes.set_yticks(ticks = np.arange(len(labels)), labels=labels, rotation=0)
    plot_widget.colorbar = plot_widget.canvas.figure.colorbar(cax, ax=plot_widget.axes, orientation='vertical')

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()


def plot_coordinates2D_highlight(plot_widget, coordinates, highlight_idxs, exclusives, only_ens, only_contours, show_numbers):
    plot_widget.axes.clear()

    all_cells = range(coordinates.shape[0])
    not_in_ens = [cell for cell in all_cells if cell not in highlight_idxs]

    # Plot all cells
    if not only_ens:
        if only_contours:
            plot_widget.axes.scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], edgecolor='blue', facecolors='none')
        else:
            plot_widget.axes.scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], c='blue', alpha=0.5)
        if show_numbers:
            for i in range(len(not_in_ens)):
                cell = not_in_ens[i]
                plot_widget.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
    
    # Highlight cells in ensemble
    not_exclusive = [cell for cell in highlight_idxs if cell not in exclusives]
    if len(not_exclusive) > 0:
        if only_contours:
            plot_widget.axes.scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], edgecolor='red', facecolors='none', label='Cells in ensemble')
        else:
            plot_widget.axes.scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], c='red', alpha=0.5, label='Cells in ensemble')
        if show_numbers:
            for i in range(len(not_exclusive)):
                cell = not_exclusive[i]
                plot_widget.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
    
    # Highlight the exclusive cells 
    if len(exclusives) > 0:
        if only_contours:
            plot_widget.axes.scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], edgecolor='yellow', facecolors='none', label='Exclusive cells')
        else:
            plot_widget.axes.scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], c='yellow', alpha=0.5, label='Exclusive cells')
        if show_numbers:
            for i in range(len(exclusives)):
                cell = exclusives[i]
                plot_widget.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
    
    # Add legend
    plot_widget.axes.legend(loc='lower left', bbox_to_anchor=(0, 1))

    # Set labels and title
    plot_widget.axes.set_xlabel('X coordinate')
    plot_widget.axes.set_ylabel('Y coordinate')
    plot_widget.axes.set_aspect('equal', adjustable='box')

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_ensemble_dFFo(plot_widget, dFFo_ens, cell_names, ens_activity):
    plot_widget.axes.clear()
    Fmi = np.min(dFFo_ens)
    Fma = np.max(dFFo_ens)
    cant_timepoints = dFFo_ens.shape[1]
    num_cells = len(cell_names)
    #cc = plt.cm.jet(np.linspace(0, 1, min(num_cells, 64)))
    #cc = np.maximum(cc - 0.3, 0)
    
    for ii in range(num_cells):
        f = dFFo_ens[ii, :]
        f = (f - Fmi) / (Fma - Fmi)
        plot_widget.axes.plot(np.arange(1, cant_timepoints + 1), ii + f, linewidth=1, color='black') #, color=cc[ii % 64]
        plot_widget.axes.text(cant_timepoints * 1.02, ii, str(cell_names[ii]+1), fontsize=8)
    
    # Iterate over the indices to create bands
    time_axis = range(0, cant_timepoints)
    band_it = 0
    while band_it < len(ens_activity):
        if ens_activity[band_it] == 1:
            start = band_it
            band_it = band_it + 1
            while band_it < len(ens_activity) and ens_activity[band_it] == 1:
                band_it = band_it + 1
            end = band_it
            plot_widget.axes.fill_between(time_axis[start:end], 0, num_cells+0.2, color='red', alpha=0.4)
        band_it = band_it + 1

    plot_widget.axes.set_xlim([1, cant_timepoints])
    plot_widget.axes.set_ylim([0, num_cells + 0.2])
    plot_widget.axes.set_xlabel('Time (timepoint)')
    plot_widget.axes.set_ylabel('Cell #')
    plot_widget.axes.set_yticks([])
    plot_widget.axes.set_xticks([])
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes.spines[side].set_visible(False)

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_all_dFFo(plot_widget, dFFo_ens, core_names, plot_ax):
    plot_widget.axes[plot_ax].clear()

    Fmi = np.min(dFFo_ens)
    Fma = np.max(dFFo_ens)
    cant_timepoints = dFFo_ens.shape[1]
    
    for ii in range(len(core_names)):
        f = dFFo_ens[ii, :]
        f = (f - Fmi) / (Fma - Fmi)
        plot_widget.axes[plot_ax].plot(np.arange(1, cant_timepoints + 1), ii + f, color='black', linewidth=1) #, color=cc[ii % 64]
        plot_widget.axes[plot_ax].text(cant_timepoints * 1.02, ii, str(core_names[ii]+1), fontsize=8)

    plot_widget.axes[plot_ax].set_xlim([1, cant_timepoints])
    plot_widget.axes[plot_ax].set_ylim([0, len(core_names) + 0.2])
    plot_widget.axes[plot_ax].set_xlabel('Time (timepoint)')
    plot_widget.axes[plot_ax].set_ylabel('Cell #')
    plot_widget.axes[plot_ax].set_title(f'Ensemble {plot_ax + 1}')
    plot_widget.axes[plot_ax].set_yticks([])
    plot_widget.axes[plot_ax].set_xticks([])
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes[plot_ax].spines[side].set_visible(False)

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_all_coords(plot_widget, coordinates, highlight_idxs, exclusives, row, col):
    plot_widget.axes[row, col].clear()
    all_cells = range(coordinates.shape[0])
    not_in_ens = [cell for cell in all_cells if cell not in highlight_idxs]

    # Plot all cells
    plot_widget.axes[row, col].scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], c='blue', alpha=0.5)
    
    # Highlight cells in ensemble
    not_exclusive = [cell for cell in highlight_idxs if cell not in exclusives]
    if len(not_exclusive) > 0:
        plot_widget.axes[row, col].scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], c='red', alpha=0.5, label='Cells in ensemble')
        
    # Highlight the exclusive cells 
    if len(exclusives) > 0:
        plot_widget.axes[row, col].scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], c='yellow', alpha=0.5, label='Exclusive cells')
    
    for i in range(coordinates.shape[0]):
        plot_widget.axes[row, col].text(coordinates[i, 0], coordinates[i, 1], str(i+1), fontsize=6, ha='left')
    
    # Add legend
    plot_widget.axes[row, col].legend(loc='lower left', bbox_to_anchor=(0, 1))

    # Set labels and title
    plot_widget.axes[row, col].set_xlabel('X coordinate')
    plot_widget.axes[row, col].set_ylabel('Y coordinate')
    plot_widget.axes[row, col].set_aspect('equal', adjustable='box')

    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_all_binary(plot_widget, bin_matrix, cells_names, ens_idx, plot_idx):
    plot_widget.axes[plot_idx].clear()
    n, t = bin_matrix.shape
    plot_widget.axes[plot_idx].imshow(bin_matrix, cmap="gray", interpolation='nearest', aspect='auto')
    plot_widget.axes[plot_idx].set_title(f"Ensemble {ens_idx+1}")
    plot_widget.axes[plot_idx].set_xlabel("Time (timepoint)")
    plot_widget.axes[plot_idx].set_ylabel("Cell")
    plot_widget.axes[plot_idx].set_xlim([0, t])
    plot_widget.axes[plot_idx].set_ylim([-0.5, n-0.5])
    plot_widget.axes[plot_idx].set_yticks(range(0, n))
    plot_widget.axes[plot_idx].set_yticklabels(cells_names)
    #plot_widget.axes[plot_idx].set_xticks([])
    #plot_widget.axes[plot_idx].set_yticks([])
    for side in ['left', 'top', 'right', 'bottom']:
        plot_widget.axes[plot_idx].spines[side].set_visible(False)
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_perf_correlations_ens_group(plot_widget, correlations, col_idx, title=None, xlabel="Group", group_labels=[]):
    plot_widget.axes[col_idx].clear()

    # Plot the correlation matrix as a heatmap
    cax = plot_widget.axes[col_idx].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    plot_widget.axes[col_idx].set_xlabel(xlabel)
    plot_widget.axes[col_idx].set_ylabel('Ensembles')

    plot_widget.canvas.figure.colorbar(cax, ax=plot_widget.axes[col_idx], orientation='vertical')

    if title != None:
        plot_widget.axes[col_idx].set_title(f"{title}")

    num_ens = correlations.shape[0]
    num_stim = correlations.shape[1]
    stim_labels = group_labels if len(group_labels) > 0 else range(1, num_stim+1)
    plot_widget.axes[col_idx].set_xticks(range(num_stim))
    plot_widget.axes[col_idx].set_yticks(range(num_ens))
    plot_widget.axes[col_idx].set_xticklabels(stim_labels)
    plot_widget.axes[col_idx].set_yticklabels(range(1, num_ens+1))

    #for ens in range(num_ens):
    #    for stim in range(num_stim):
    #        plot_widget.axes[col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
    #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
    
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_perf_correlations_cells(plot_widget, correlations, cells_names, col_idx, row_idx, title=None):
    plot_widget.axes[row_idx][col_idx].clear()

    # Plot the correlation matrix as a heatmap
    cax = plot_widget.axes[row_idx][col_idx].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    plot_widget.axes[row_idx][col_idx].set_xlabel('Cell')
    plot_widget.axes[row_idx][col_idx].set_ylabel('Cell')

    plot_widget.canvas.figure.colorbar(cax, ax=plot_widget.axes[row_idx][col_idx], orientation='vertical')

    if title != None:
        plot_widget.axes[row_idx][col_idx].set_title(f"{title}")

    num_cells = correlations.shape[0]
    plot_widget.axes[row_idx][col_idx].set_xticks(range(num_cells))
    plot_widget.axes[row_idx][col_idx].set_yticks(range(num_cells))
    plot_widget.axes[row_idx][col_idx].set_xticklabels(cells_names)
    plot_widget.axes[row_idx][col_idx].set_yticklabels(cells_names)

    #for ens in range(num_cells):
    #    for stim in range(num_cells):
    #        plot_widget.axes[row_idx][col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
    #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
    
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()

def plot_perf_cross_ens_stims(plot_widget, cross_corrs, lags, col_idx, row_idx, group_prefix="Group", title=None, group_labels=[]):
    plot_widget.axes[row_idx][col_idx].clear()

    # Plot the correlation matrix as a heatmap
    num_stim = cross_corrs.shape[0]
    stim_labels = group_labels if len(group_labels) > 0 else range(1, num_stim+1)
    for c_idx, cross_corr in enumerate(cross_corrs):
        plot_widget.axes[row_idx][col_idx].plot(lags, cross_corr, label=f"{group_prefix} {stim_labels[c_idx]}")
    plot_widget.axes[row_idx][col_idx].axhline(0, color='black', linestyle='--', linewidth=0.5)
    plot_widget.axes[row_idx][col_idx].set_xlabel('Lag')
    plot_widget.axes[row_idx][col_idx].set_ylabel('Cross correlation')
    plot_widget.axes[row_idx][col_idx].legend()

    if title != None:
        plot_widget.axes[row_idx][col_idx].set_title(f"{title}")

    #for ens in range(num_ens):
    #    for stim in range(num_stim):
    #        plot_widget.axes[row_idx][col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
    #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
    
    plot_widget.canvas.figure.tight_layout()
    plot_widget.canvas.draw()
    plot_widget.canvas.flush_events()
