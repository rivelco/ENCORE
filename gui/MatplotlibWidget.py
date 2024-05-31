from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
import numpy as np

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

    def preview_dataset(self, dataset):
        self.axes.clear()
        n, t = dataset.shape
        self.axes.imshow(dataset, cmap='hot', interpolation='nearest', aspect='auto')
        #self.axes.colorbar()  # Add a colorbar to show the scale
        #self.axes.set_title('Heatmap using Matplotlib')
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Data')
        self.axes.set_xlim([0, t])
        self.axes.set_ylim([0, n])
        self.axes.set_xticks([])
        self.axes.set_yticks([])
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
