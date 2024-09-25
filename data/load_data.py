import h5py
from PyQt6.QtCore import Qt, QAbstractItemModel, QModelIndex
from PyQt6.QtGui import QStandardItemModel, QStandardItem
import numpy as np
import scipy.io
import csv

class FileTreeItem:
    def __init__(self, name, obj, mdl_type, parent=None):
        self.name = name
        self.obj = obj
        self.parent_item = parent
        self.child_items = []

        if mdl_type == "hdf5":
            if isinstance(obj, h5py.Group):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for key, val in obj.items():
                    self.child_items.append(FileTreeItem(key, val, mdl_type, self))
            elif isinstance(obj, h5py.Dataset):
                if len(obj.shape) == 0:
                    self.obj_type = "Scalar"
                    self.obj_size = np.array(obj)
                else:
                    self.obj_type = "Dataset"
                    self.obj_size = obj.shape
        
        if mdl_type == "pkl":
            #print(str(type(obj)))
            if isinstance(obj, dict):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for var_name, var_value in obj.items():
                    if not var_name.startswith('__'):
                        self.child_items.append(FileTreeItem(var_name, var_value, mdl_type, self))
            if isinstance(obj, np.ndarray):
                # This is an array or matrix.
                self.obj_type = "Dataset"
                self.obj_size = obj.shape
            elif isinstance(obj, str):
                # This is a struct.
                self.obj_type = "String"
                self.obj_size = -1
            elif isinstance(obj, float):
                # This is a struct.
                self.obj_type = "Scalar"
                self.obj_size = -1
            else:
                # Unknown type.
                self.obj_type = "Unknown"
                self.obj_size = -1

        if mdl_type == "mat":
            if isinstance(obj, dict):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for var_name, var_value in obj.items():
                    if not var_name.startswith('__'):
                        self.child_items.append(FileTreeItem(var_name, var_value, mdl_type, self))
            if isinstance(obj, np.ndarray):
                if obj.dtype == 'O':  # Object array (likely cell array)
                    #This is a cell array.
                    self.obj_type = "Group"
                    self.obj_size = obj.shape[0]
                    if not isinstance(obj, np.ndarray) or obj.dtype != 'O':
                        raise ValueError("The input is not a MATLAB cell array represented as a NumPy object array.")
                    for i in range(obj.shape[0]):
                        key = f"element_{i}"
                        self.child_items.append(FileTreeItem(key, obj[i], mdl_type, self))
                elif obj.size == 1:  # Scalar
                    # This is a scalar.
                    self.obj_type = "Scalar"
                    self.obj_size = obj[0]
                else:
                    # This is an array or matrix.
                    self.obj_type = "Dataset"
                    self.obj_size = obj.shape
            elif isinstance(obj, np.void):
                # This is a struct.
                self.obj_type = "Struct"
                self.obj_size = -1
            else:
                # Unknown type.
                self.obj_type = "Unknown"
                self.obj_size = -1

        if mdl_type == "csv":
            if self.name == "/":
                self.obj_type = "Group"
                self.obj_size = 1
                self.child_items.append(FileTreeItem("CSV_dataset", obj, mdl_type, self))
            else:
                self.obj_type = "Dataset"
                csvreader = csv.reader(obj)
                # Get the number of columns from the first row
                first_row = next(csvreader)
                num_columns = len(first_row)
                # Initialize row count
                num_rows = 1  # Already read the first row
                # Count the remaining rows
                for _ in csvreader:
                    num_rows += 1
                self.obj_size = (num_rows, num_columns)

    def child(self, row):
        return self.child_items[row]

    def child_count(self):
        return len(self.child_items)

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def column_count(self):
        return 1

    def data(self):
        return self.name
    
    def item_path(self):
        if self.parent_item:
            return self.parent_item.item_path() + "/" + self.name
        else:
            return self.name
    
    def item_type(self):
        return self.obj_type
    
    def item_size(self):
        return self.obj_size

    def parent(self):
        return self.parent_item

class FileTreeModel(QAbstractItemModel):
    def __init__(self, hdf5_file, model_type, parent=None):
        super(FileTreeModel, self).__init__(parent)
        self.root_item = FileTreeItem("/", hdf5_file, model_type)

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.child_count()

    def columnCount(self, parent=QModelIndex()):
        return self.root_item.column_count()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        item = index.internalPointer()
        return item.data()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return "Loaded File Structure"
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsEnabled
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)
    
    def data_name(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        item = index.internalPointer()
        return item.item_path()
    
    def data_type(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        item = index.internalPointer()
        return item.item_type()
    
    def data_size(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        item = index.internalPointer()
        return item.item_size()
