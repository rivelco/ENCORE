from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

class MatplotlibWidget(QWidget):
    def __init__(self, rows=1, cols=1, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        # Create subplots based on rows and cols
        self.set_subplots(rows, cols)

    def set_subplots(self, rows, cols):
        # Clear existing subplots
        self.canvas.figure.clf()
        # Create new subplots
        self.axes = self.canvas.figure.subplots(rows, cols)
        if rows == 1 and cols == 1:
            self.axes.axis('off')
        elif rows == 1 or cols == 1:
            tmp = max(rows, cols)
            for idx in range(tmp):
                self.axes[idx].axis('off')
            self.canvas.figure.tight_layout()
        elif rows > 1 and cols > 1:
            for row in range(rows):
                for col in range(cols):
                    self.axes[row][col].axis('off')
            self.canvas.figure.tight_layout()
    
    def reset(self, default_text="Nothing to show", rows=1, cols=1):
        self.set_subplots(rows, cols)
        self.axes.text(0.5, 0.5, f'{default_text}', 
        horizontalalignment='center', 
        verticalalignment='center', 
        fontsize=8, 
        transform=self.axes.transAxes)
        self.axes.set_axis_off()
        self.canvas.draw()

    def preview_dataset(self, dataset, xlabel='Timepoint', ylabel='Data', title=None, cmap='hot', aspect='auto'):
        self.axes.clear()
        n, t = dataset.shape
        self.axes.imshow(dataset, cmap=cmap, interpolation='nearest', aspect=aspect)
        if title != None:
            self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlim([0, t])
        self.axes.set_ylim([-0.5, n-0.5])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes.spines[side].set_visible(False)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def preview_coordinates2D(self, dataset):
        self.axes.clear()
        self.axes.scatter(dataset[:,0], dataset[:,1], c='blue', marker='o')
        
        self.axes.set_xlabel('X coordinates')
        self.axes.set_ylabel('Y coordinates')
        self.axes.set_xlim([min(dataset[:,0]) - 10, max(dataset[:,0]) + 10])
        self.axes.set_ylim([min(dataset[:,1]) - 10, max(dataset[:,1]) + 10])
        #self.axes.set_xticks([])
        #self.axes.set_yticks([])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes.spines[side].set_visible(False)
        self.axes.set_aspect('equal', adjustable='box')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def raster_plot(self, data_neuronal_activity):
        self.axes.clear()
        
        n, t = data_neuronal_activity.shape
        for neuron in range(n):
            spike_times = np.where(data_neuronal_activity[neuron] == 1)[0]
            self.axes.vlines(spike_times, neuron + 0.5, neuron + 1.5, color='black')

        self.axes.set_xlim([0, t])
        self.axes.set_ylim([0, n+1.5])
        self.axes.set_yticklabels([])
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes.spines[side].set_visible(False)
        self.axes.set_xlabel("Time")
        self.axes.set_ylabel("Neuron")

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # Plots for the SVD analysis
    def plot_singular_values(self, singulars, num_states):
        self.axes.clear()
        self.axes.semilogx(singulars, marker='o', linestyle='-', label='Singular values')
        # Add a vertical red dashed line at num_state
        self.axes.plot(num_states, singulars[num_states], 'rs', markersize=10, label=f"Cutoff at {num_states}")
        #self.axes.axvline(x=num_states, color='r', linestyle='--', linewidth=2, label=f'Cutoff at {num_states}')
        # Label the axes
        self.axes.set_xlabel('log(singular value)')
        self.axes.set_ylabel('Magnitude')
        # Set the title
        self.axes.set_title('Singular values')
        self.axes.legend()
        for side in ['top', 'right']:
            self.axes.spines[side].set_visible(False)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_states_from_svd(self, svd_sig, comp_n, row, col):
        self.axes[row][col].clear()
        # Plot the image, where svd_sig[:,:,n]==0 is the condition to be checked
        self.axes[row][col].imshow(svd_sig == 0, cmap='gray', aspect='equal')
        # Set the labels and title
        self.axes[row][col].set_xlabel('Population vector')
        self.axes[row][col].set_ylabel('Population vector')
        self.axes[row][col].set_title(f'Components ensemble {comp_n+1}')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ensembles_timecourse(self, timecourse, xlabel="Timepoint"):
        self.axes.clear()
        ensembles_cant = timecourse.shape[0]
        frames_cant = timecourse.shape[1]
        for ens in range(ensembles_cant):
            for frame in range(frames_cant):
                if timecourse[ens, frame]:
                    self.axes.plot(frame+1, ens+1, '|', markerfacecolor='none', markeredgecolor='k', markersize=15)

        self.axes.set_xlabel(xlabel)                # Show the frame label
        self.axes.spines['top'].set_visible(False)   # Hide the top line for everyone except the first
        self.axes.spines['right'].set_visible(False) # Hide the rigth line for everyone except the first
        self.axes.yaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Show only integers in the y axis
        self.axes.set_ylim([0.5, ensembles_cant+0.5])       # Set the y axis limit
        self.axes.set_xlim([0, frames_cant]) # Set the x axis limit
        self.axes.set_ylabel('Ensemble') 

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ens_seq(self, tt, ens_labs, ens_cols, sellabs):
        self.axes.clear()
        nens = np.max(ens_labs)
        try:
            if nens == 0:
                self.axes.plot([tt[0], tt[-1]], [0, 0], 'k-')
                return
            else:
                self.axes.plot(tt, ens_labs, 'k--')
                self.axes.plot(tt[ens_labs == 0], ens_labs[ens_labs == 0], 'ko')
                for e in range(1, nens + 1):
                    self.axes.plot(tt[ens_labs == e], ens_labs[ens_labs == e], '.', markersize=25, color=ens_cols[e-1])
                self.axes.set_ylim([0, nens * 1.1])
        except Exception as e:
            print(f"Error occurred: {e}")
            self.axes.plot([tt[0], tt[-1]], [0, 0], 'k-')

        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Ensemble')
        self.axes.set_yticks(range(1, nens + 1))
        self.axes.set_yticklabels(sellabs)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # Plots for the PCA 
    def plot_eigs(self, eigs, seleig):
        self.axes.clear()

        self.axes.plot(np.arange(1, len(eigs) + 1), eigs, 'o--', label="Principal components")
        self.axes.plot(seleig, eigs[seleig - 1], 'rs', markersize=10, label=f"Cutoff at {seleig}")
        self.axes.set_xscale('log')
        self.axes.set_yscale('linear')
        self.axes.set_xlim(1, len(eigs))
        self.axes.set_xlabel('Principal components')
        self.axes.set_ylabel('% var')
        self.axes.legend()

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_pca(self, pcs, ens_labs=None, ens_cols=None):
        self.axes.clear()
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes.spines[side].set_visible(False)
        self.axes.set_yticks([])
        self.axes.set_xticks([])
        np = pcs.shape[1]
        if ens_labs is None or ens_cols is None:  # plots raster with no color
            if np > 2:
                self.axes = self.canvas.figure.add_subplot(111, projection='3d')
                self.axes.plot3D(pcs[:, 0], pcs[:, 1], pcs[:, 2], 'k.')
            else:
                self.axes = self.canvas.figure.add_subplot(111)
                self.axes.plot(pcs[:, 0], pcs[:, 1], 'k.')
            self.axes.set_xlabel('PC1')
            self.axes.set_ylabel('PC2')
            if np > 2:
                self.axes.set_zlabel('PC3')
            self.axes.grid(True)
        else:
            nens = int(max(ens_labs))
            if np > 2:
                self.axes = self.canvas.figure.add_subplot(111, projection='3d')
                self.axes.plot3D(pcs[ens_labs == 0, 0], pcs[ens_labs == 0, 1], pcs[ens_labs == 0, 2], 'k.')
                for e in range(1, nens + 1):
                    self.axes.plot3D(pcs[ens_labs == e, 0], pcs[ens_labs == e, 1], pcs[ens_labs == e, 2],
                            marker='.', color=ens_cols[e - 1], linestyle='none')
            else:
                self.axes = self.canvas.figure.add_subplot(111)
                self.axes.plot(pcs[ens_labs == 0, 0], pcs[ens_labs == 0, 1], 'k.')
                for e in range(1, nens + 1):
                    self.axes.plot(pcs[ens_labs == e, 0], pcs[ens_labs == e, 1],
                            marker='.', color=ens_cols[e - 1], linestyle='none')
            self.axes.set_xlabel('PC1')
            self.axes.set_ylabel('PC2')
            if np > 2:
                self.axes.set_zlabel('PC3')
            self.axes.grid(True)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_delta_rho(self, rho, delta, cents, predbounds, ens_cols):
        self.axes.clear()
        # Sort pred_id based on the first column of predbounds
        pred_id = np.argsort(predbounds[:, 0])
        # Plot pred bounds
        self.axes.plot(predbounds[pred_id, 0], predbounds[pred_id, 1], 'b--', linewidth=2)
        # Plot delta rho
        self.axes.plot(rho, delta, 'k.')
        # Plot points based on the cluster centers (cents)
        nens = np.sum(cents > 0)
        for e in range(1, nens + 1):
            self.axes.plot(rho[cents == e], delta[cents == e], '.', markersize=25, color=ens_cols[e - 1])
            for r, d in zip(rho[cents == e], delta[cents == e]):
                self.axes.text(r * 1.01, d * 1.01, str(e), fontsize=12)
        # Set x and y limits
        self.axes.set_xlim([0, np.max(rho[~np.isinf(rho)]) * 1.1])
        self.axes.set_ylim([0, np.max(delta[~np.isinf(delta)]) * 1.1])
        # Set labels
        self.axes.set_xlabel(r'$\rho$')
        self.axes.set_ylabel(r'$\delta$')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_core_cells(self, core_cells, clims):
        # Clear the axes
        self.axes.clear()
        # Determine the size of core_cells
        N, nens = core_cells.shape
        # Plot the image
        cax = self.axes.imshow(core_cells, aspect='auto', cmap='bwr', vmin=clims[0], vmax=clims[1])
        # Set the color limits
        cax.set_clim(clims)
        # Set axis limits
        self.axes.set_xlim([-0.5, nens-0.5])
        self.axes.set_ylim([0, N])
        # Draw the lines
        for e in range(nens + 1):
            self.axes.plot([e - 0.5, e - 0.5], [0, N], 'k-')
        # Remove y-axis labels and ticks
        self.axes.set_yticks([])
        self.axes.set_yticklabels([])
        self.axes.set_xticks(range(0, nens))
        self.axes.set_xticklabels(range(1, nens+1))
        # Set y-axis label
        self.axes.set_ylabel('Cells')
        self.axes.set_xlabel('Ensemble')
        # Add a box around the plot
        self.axes.spines['top'].set_visible(True)
        self.axes.spines['right'].set_visible(True)
        self.axes.spines['bottom'].set_visible(True)
        self.axes.spines['left'].set_visible(True)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ens_corr(self, ens_corr, corr_thr, ens_cols):
        # Clear the axes
        self.axes.clear()
        nens = len(ens_corr)
        # Plot bars
        for e in range(nens):
            self.axes.bar(e + 1, ens_corr[e], color=ens_cols[e], edgecolor=ens_cols[e])
        # Plot threshold line
        self.axes.plot([0.5, nens + 0.5], [corr_thr, corr_thr], 'r--')
        # Set x and y limits
        self.axes.set_xlim([0.5, nens + 0.5])
        self.axes.set_ylim([0, max(max(ens_corr), corr_thr) * 1.1])
        # Set labels
        self.axes.set_xlabel('Ensemble Id.')
        self.axes.set_ylabel('Core-Cells Mean Correlation')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # Plots for the ICA
    def plot_assembly_patterns(self, Patterns, row_idx, title=None, plot_xaxis=False):
        self.axes[row_idx].clear()
        self.axes[row_idx].stem(Patterns)
        if title != None:
            self.axes[row_idx].set_title(f"{title}")

        self.axes[row_idx].set_ylabel('Weight')
        for side in ['top', 'right', 'bottom']:
            self.axes[row_idx].spines[side].set_visible(False)
        if plot_xaxis:
            self.axes[row_idx].spines['bottom'].set_visible(True)
            self.axes[row_idx].set_xlabel('Cell')
            self.canvas.figure.tight_layout()
        else:
            self.axes[row_idx].set_xticks([])
        
        self.canvas.draw()

    def plot_cell_assemblies_activity(self, activities):
        self.axes.clear()

        for e_idx, ensemble in enumerate(activities):
            self.axes.plot(ensemble, label=f"Ensemble {e_idx+1}")

        for side in ['top', 'right']:
            self.axes.spines[side].set_visible(False)
        self.axes.set_xlabel('Timepoint')
        self.axes.set_ylabel('Assembly activity')

        self.axes.legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    # Plot the ensembles compare
    def enscomp_update_map(self, lims, members_idx, members_freqs, members_coords, members_colors, neuron_size):
        self.axes.clear()

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
                self.axes.add_patch(wedge)
                deg_start += deg_step
                deg_end += deg_step

        for idx in range(len(members_idx)):
            member_idx = members_idx[idx]
            member_coords_x = members_coords[0][idx]
            member_coords_y = members_coords[1][idx]
            self.axes.text(member_coords_x, member_coords_y, str(member_idx+1), fontsize=6,
                            ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, boxstyle='round', pad=0))
        
        # Adjust plot limits
        lim_offset = 1.5*neuron_size
        self.axes.set_xlim(0-lim_offset, lims[0]+lim_offset)
        self.axes.set_ylim(0-lim_offset, lims[1]+lim_offset)
        self.axes.set_xlabel("X coordinates")
        self.axes.set_ylabel("Y coordinates")
        self.axes.set_aspect('equal')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def enscomp_update_timelines(self, ticks, cell_activities, ensemble_dffo, ensemble_timecourse, colors, limx):
        self.axes.clear()

        # Iterate over the indices to create bands
        for acts in range(len(ensemble_timecourse)):
            cells_acts = ensemble_timecourse[acts]
            act_lenght = len(cells_acts)
            time_axis = range(0, act_lenght)
            if act_lenght > 0:
                band_it = 0
                while band_it < act_lenght:
                    if cells_acts[band_it] == 1:
                        start = band_it
                        band_it = band_it + 1
                        while band_it < act_lenght and cells_acts[band_it] == 1:
                            band_it = band_it + 1
                        end = band_it
                        self.axes.fill_between(time_axis[start:end], acts, acts+1, color=colors[acts], alpha=1)
                    band_it = band_it + 1

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
                    cells_acts = (cells_acts - Fmi) / (Fma - Fmi)
                    self.axes.plot(np.arange(1, cant_timepoints + 1), acts + cells_acts, linewidth=1, color='black', alpha=0.6)
                    #self.axes.text(cant_timepoints * 1.02, ii, str(cell_names[ii]+1), fontsize=8)

        self.axes.set_xlim([0, limx])
        self.axes.set_ylim([0, len(ticks)])
        self.axes.set_xlabel('Time (timepoint)')
        self.axes.set_yticks([tick+0.5 for tick in range(len(ticks))])
        self.axes.set_yticklabels(ticks)
        for side in ['left', 'top', 'right']:
            self.axes.spines[side].set_visible(False)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def enscomp_plot_similarity(self, matrix, labels):
        self.axes.clear()

        self.axes.imshow(matrix, aspect='equal')
        self.axes.set_xticks(ticks = np.arange(len(labels)), labels=labels, rotation=45)
        self.axes.set_yticks(ticks = np.arange(len(labels)), labels=labels, rotation=0)

        self.canvas.figure.tight_layout()
        self.canvas.draw()


    def plot_coordinates2D_highlight(self, coordinates, highlight_idxs, exclusives, only_ens, only_contours, show_numbers):
        self.axes.clear()

        all_cells = range(coordinates.shape[0])
        not_in_ens = [cell for cell in all_cells if cell not in highlight_idxs]

        # Plot all cells
        if not only_ens:
            if only_contours:
                self.axes.scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], edgecolor='blue', facecolors='none')
            else:
                self.axes.scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], c='blue', alpha=0.5)
            if show_numbers:
                for i in range(len(not_in_ens)):
                    cell = not_in_ens[i]
                    self.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
        
        # Highlight cells in ensemble
        not_exclusive = [cell for cell in highlight_idxs if cell not in exclusives]
        if len(not_exclusive) > 0:
            if only_contours:
                self.axes.scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], edgecolor='red', facecolors='none', label='Cells in ensemble')
            else:
                self.axes.scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], c='red', alpha=0.5, label='Cells in ensemble')
            if show_numbers:
                for i in range(len(not_exclusive)):
                    cell = not_exclusive[i]
                    self.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
        
        # Highlight the exclusive cells 
        if len(exclusives) > 0:
            if only_contours:
                self.axes.scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], edgecolor='yellow', facecolors='none', label='Exclusive cells')
            else:
                self.axes.scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], c='yellow', alpha=0.5, label='Exclusive cells')
            if show_numbers:
                for i in range(len(exclusives)):
                    cell = exclusives[i]
                    self.axes.text(coordinates[cell, 0], coordinates[cell, 1], str(cell+1), fontsize=6, ha='right')
        
        # Add legend
        self.axes.legend(loc='lower left', bbox_to_anchor=(0, 1))

        # Set labels and title
        self.axes.set_xlabel('X coordinate')
        self.axes.set_ylabel('Y coordinate')
        self.axes.set_aspect('equal', adjustable='box')

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ensemble_dFFo(self, dFFo_ens, cell_names, ens_activity):
        self.axes.clear()
        Fmi = np.min(dFFo_ens)
        Fma = np.max(dFFo_ens)
        cant_timepoints = dFFo_ens.shape[1]
        num_cells = len(cell_names)
        #cc = plt.cm.jet(np.linspace(0, 1, min(num_cells, 64)))
        #cc = np.maximum(cc - 0.3, 0)
        
        for ii in range(num_cells):
            f = dFFo_ens[ii, :]
            f = (f - Fmi) / (Fma - Fmi)
            self.axes.plot(np.arange(1, cant_timepoints + 1), ii + f, linewidth=1, color='black') #, color=cc[ii % 64]
            self.axes.text(cant_timepoints * 1.02, ii, str(cell_names[ii]+1), fontsize=8)
        
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
                self.axes.fill_between(time_axis[start:end], 0, num_cells+0.2, color='red', alpha=0.4)
            band_it = band_it + 1

        self.axes.set_xlim([1, cant_timepoints])
        self.axes.set_ylim([0, num_cells + 0.2])
        self.axes.set_xlabel('Time (timepoint)')
        self.axes.set_ylabel('Cell #')
        self.axes.set_yticks([])
        self.axes.set_xticks([])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes.spines[side].set_visible(False)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_all_dFFo(self, dFFo_ens, core_names, plot_ax):
        self.axes[plot_ax].clear()

        Fmi = np.min(dFFo_ens)
        Fma = np.max(dFFo_ens)
        cant_timepoints = dFFo_ens.shape[1]
        
        for ii in range(len(core_names)):
            f = dFFo_ens[ii, :]
            f = (f - Fmi) / (Fma - Fmi)
            self.axes[plot_ax].plot(np.arange(1, cant_timepoints + 1), ii + f, color='black', linewidth=1) #, color=cc[ii % 64]
            self.axes[plot_ax].text(cant_timepoints * 1.02, ii, str(core_names[ii]+1), fontsize=8)

        self.axes[plot_ax].set_xlim([1, cant_timepoints])
        self.axes[plot_ax].set_ylim([0, len(core_names) + 0.2])
        self.axes[plot_ax].set_xlabel('Time (timepoint)')
        self.axes[plot_ax].set_ylabel('Cell #')
        self.axes[plot_ax].set_title(f'Ensemble {plot_ax + 1}')
        self.axes[plot_ax].set_yticks([])
        self.axes[plot_ax].set_xticks([])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes[plot_ax].spines[side].set_visible(False)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_all_coords(self, coordinates, highlight_idxs, exclusives, row, col):
        self.axes[row, col].clear()
        all_cells = range(coordinates.shape[0])
        not_in_ens = [cell for cell in all_cells if cell not in highlight_idxs]

        # Plot all cells
        self.axes[row, col].scatter(coordinates[not_in_ens, 0], coordinates[not_in_ens, 1], c='blue', alpha=0.5)
        
        # Highlight cells in ensemble
        not_exclusive = [cell for cell in highlight_idxs if cell not in exclusives]
        if len(not_exclusive) > 0:
            self.axes[row, col].scatter(coordinates[not_exclusive, 0], coordinates[not_exclusive, 1], c='red', alpha=0.5, label='Cells in ensemble')
            
        # Highlight the exclusive cells 
        if len(exclusives) > 0:
            self.axes[row, col].scatter(coordinates[exclusives, 0], coordinates[exclusives, 1], c='yellow', alpha=0.5, label='Exclusive cells')
        
        for i in range(coordinates.shape[0]):
            self.axes[row, col].text(coordinates[i, 0], coordinates[i, 1], str(i+1), fontsize=6, ha='left')
        
        # Add legend
        self.axes[row, col].legend(loc='lower left', bbox_to_anchor=(0, 1))

        # Set labels and title
        self.axes[row, col].set_xlabel('X coordinate')
        self.axes[row, col].set_ylabel('Y coordinate')
        self.axes[row, col].set_aspect('equal', adjustable='box')

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_all_binary(self, bin_matrix, cells_names, ens_idx, plot_idx):
        self.axes[plot_idx].clear()
        n, t = bin_matrix.shape
        self.axes[plot_idx].imshow(bin_matrix, cmap="gray", interpolation='nearest', aspect='auto')
        self.axes[plot_idx].set_title(f"Ensemble {ens_idx+1}")
        self.axes[plot_idx].set_xlabel("Time (timepoint)")
        self.axes[plot_idx].set_ylabel("Cell")
        self.axes[plot_idx].set_xlim([0, t])
        self.axes[plot_idx].set_ylim([-0.5, n-0.5])
        self.axes[plot_idx].set_yticks(range(0, n))
        self.axes[plot_idx].set_yticklabels(cells_names)
        #self.axes[plot_idx].set_xticks([])
        #self.axes[plot_idx].set_yticks([])
        for side in ['left', 'top', 'right', 'bottom']:
            self.axes[plot_idx].spines[side].set_visible(False)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_perf_correlations_ens_stim(self, correlations, col_idx, title=None):
        self.axes[col_idx].clear()

        # Plot the correlation matrix as a heatmap
        cax = self.axes[col_idx].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        self.axes[col_idx].set_xlabel('Stims')
        self.axes[col_idx].set_ylabel('Ensembles')

        self.canvas.figure.colorbar(cax, ax=self.axes[col_idx], orientation='vertical')

        if title != None:
            self.axes[col_idx].set_title(f"{title}")

        num_ens = correlations.shape[0]
        num_stim = correlations.shape[1]
        self.axes[col_idx].set_xticks(range(num_stim))
        self.axes[col_idx].set_yticks(range(num_ens))
        self.axes[col_idx].set_xticklabels(range(1, num_stim+1))
        self.axes[col_idx].set_yticklabels(range(1, num_ens+1))

        #for ens in range(num_ens):
        #    for stim in range(num_stim):
        #        self.axes[col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
        #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
        
        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    def plot_perf_correlations_cells(self, correlations, cells_names, col_idx, row_idx, title=None):
        self.axes[row_idx][col_idx].clear()

        # Plot the correlation matrix as a heatmap
        cax = self.axes[row_idx][col_idx].imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
        self.axes[row_idx][col_idx].set_xlabel('Cell')
        self.axes[row_idx][col_idx].set_ylabel('Cell')

        self.canvas.figure.colorbar(cax, ax=self.axes[row_idx][col_idx], orientation='vertical')

        if title != None:
            self.axes[row_idx][col_idx].set_title(f"{title}")

        num_cells = correlations.shape[0]
        self.axes[row_idx][col_idx].set_xticks(range(num_cells))
        self.axes[row_idx][col_idx].set_yticks(range(num_cells))
        self.axes[row_idx][col_idx].set_xticklabels(cells_names)
        self.axes[row_idx][col_idx].set_yticklabels(cells_names)

        #for ens in range(num_cells):
        #    for stim in range(num_cells):
        #        self.axes[row_idx][col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
        #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
        
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_perf_cross_ens_stims(self, cross_corrs, lags, col_idx, row_idx, title=None):
        self.axes[row_idx][col_idx].clear()

        # Plot the correlation matrix as a heatmap
        for c_idx, cross_corr in enumerate(cross_corrs):
            self.axes[row_idx][col_idx].plot(lags, cross_corr, label=f"Stim {c_idx+1}")
        self.axes[row_idx][col_idx].axhline(0, color='black', linestyle='--', linewidth=0.5)
        self.axes[row_idx][col_idx].set_xlabel('Lag')
        self.axes[row_idx][col_idx].set_ylabel('Cross correlation')
        self.axes[row_idx][col_idx].legend()

        if title != None:
            self.axes[row_idx][col_idx].set_title(f"{title}")

        #for ens in range(num_ens):
        #    for stim in range(num_stim):
        #        self.axes[row_idx][col_idx].text(stim, ens, f"{correlations[ens, stim]:.2f}",
        #                    ha="center", va="center", color="black" if abs(correlations[ens, stim]) < 0.5 else "white")
        
        self.canvas.figure.tight_layout()
        self.canvas.draw()
