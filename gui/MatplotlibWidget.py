from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
import numpy as np
import matplotlib.pyplot as plt

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self.axes = self.canvas.figure.add_subplot(111)
        self.axes.axis('off')
        self.canvas.figure.tight_layout()

    def preview_dataset(self, dataset, xlabel='Frame', ylabel='Data', title=None, cmap='hot', aspect='auto'):
        self.axes.clear()
        n, t = dataset.shape
        self.axes.imshow(dataset, cmap=cmap, interpolation='nearest', aspect=aspect)
        if title != None:
            self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlim([0, t])
        self.axes.set_ylim([-0.5, n-0.5])
        #self.axes.set_xticks([])
        #self.axes.set_yticks([])
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

    # Plot singular values
    def plot_singular_values(self, singulars, num_states):
        self.axes.clear()
        self.axes.semilogx(singulars, marker='o', linestyle='-', label='Singular values')
        # Add a vertical red dashed line at num_state
        self.axes.axvline(x=num_states, color='r', linestyle='--', linewidth=2, label=f'num_state = {num_states}')
        # Label the axes
        self.axes.set_xlabel('Singular value')
        self.axes.set_ylabel('Singular values idx')
        # Set the title
        self.axes.set_title('Singular values')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_states_from_svd(self, svd_sig, comp_n):
        self.axes.clear()

        # Plot the image, where svd_sig[:,:,n]==0 is the condition to be checked
        self.axes.imshow(svd_sig == 0, cmap='gray', aspect='equal')

        # Set the labels and title
        self.axes.set_xlabel('frame')
        self.axes.set_ylabel('frame')
        self.axes.set_title(f'Components ensemble {comp_n+1}')

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ensembles_timecourse(self, timecourse):
        self.axes.clear()

        ensembles_cant = timecourse.shape[0]
        frames_cant = timecourse.shape[1]

        for ens in range(ensembles_cant):
            for frame in range(frames_cant):
                if timecourse[ens, frame]:
                    self.axes.plot(frame+1, ens+1, '|', markerfacecolor='none', markeredgecolor='k', markersize=15)

        self.axes.set_xlabel('Frame')                # Show the frame label
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

    def plot_eigs(self, eigs, seleig):
        self.axes.clear()

        self.axes.plot(np.arange(1, len(eigs) + 1), eigs, 'o--')
        self.axes.plot(seleig, eigs[seleig - 1], 'rs', markersize=10)
        self.axes.set_xscale('log')
        self.axes.set_yscale('linear')
        self.axes.set_xlim(1, len(eigs))
        self.axes.set_xlabel('PCs')
        self.axes.set_ylabel('% var')
        self.axes.grid(True)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_pca(self, pcs, ens_labs=None, ens_cols=None):
        self.axes.clear()
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
        self.axes.set_xlim([0, nens])
        self.axes.set_ylim([0, N])

        # Draw the lines
        for e in range(nens + 1):
            self.axes.plot([e - 0.5, e - 0.5], [0, N], 'k-')

        # Remove y-axis labels and ticks
        self.axes.set_yticks([])
        self.axes.set_yticklabels([])
        
        # Set y-axis label
        self.axes.set_ylabel('')
        
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