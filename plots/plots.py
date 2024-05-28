from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtCore import QRectF, Qt
from matplotlib.figure import Figure
import numpy as np

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

def plot_raster(self):
    # Clear the scene
    self.scene.clear()

    n, t = self.data_neuronal_activity.shape
    pen = QPen(QColor(0, 0, 0))

    for neuron in range(n):
        for time in range(t):
            if self.data_neuronal_activity[neuron, time] == 1:
                x = time
                y = neuron
                self.scene.addLine(x, y, x, y+1, pen)

    # Adjust the view
    self.neuronal_raster_plot.fitInView(QRectF(0, 0, t, n))
