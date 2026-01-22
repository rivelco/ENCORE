from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib as mpl

class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)

    def save_figure(self, *args):
        # Add options for both SVG and PNG file types
        options = "SVG files (*.svg);;PNG files (*.png);;All files (*)"
        filename, filetype = QFileDialog.getSaveFileName(self, "Save Figure", "", options)
        
        if filename:
            # Check and append the correct file extension if not present
            if filetype == "SVG files (*.svg)" and not filename.endswith('.svg'):
                filename += '.svg'
            elif filetype == "PNG files (*.png)" and not filename.endswith('.png'):
                filename += '.png'

            # Set SVG font type to 'none' to preserve text as text in SVG
            if filename.endswith('.svg'):
                mpl.rcParams['svg.fonttype'] = 'none'
            else:
                mpl.rcParams['svg.fonttype'] = 'path'  # Default behavior for PNG
            
            # Save the figure with the selected format
            format = 'svg' if filename.endswith('.svg') else 'png'
            self.canvas.figure.savefig(filename, format=format)

class MatplotlibWidget(QWidget):
    def __init__(self, rows=1, cols=1, parent=None):
        super().__init__(parent)
        #plt.style.use('dark_background')
        self.canvas = FigureCanvas(Figure())
        self.toolbar = CustomNavigationToolbar(self.canvas, self)
        #self.toolbar = NavigationToolbar(self.canvas, self)
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
        self.canvas.flush_events()
