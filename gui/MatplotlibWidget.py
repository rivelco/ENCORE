from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
import numpy as np
import matplotlib.pyplot as plt

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.axes = self.canvas.figure.add_subplot(111)
        self.axes.axis('off')
        self.canvas.figure.tight_layout()

    def preview_dataset(self, dataset, xlabel='Time', ylabel='Data', cmap='hot', aspect='auto'):
        self.axes.clear()
        n, t = dataset.shape
        self.axes.imshow(dataset, cmap=cmap, interpolation='nearest', aspect=aspect)
        #self.axes.set_title('Heatmap using Matplotlib')
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.set_xlim([0, t])
        self.axes.set_ylim([0, n-1])
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