import h5py
from PyQt6.QtCore import Qt, QAbstractItemModel, QModelIndex
import numpy as np
import csv

class FileTreeItem:
    """
    Class to define a item for a FileTreeModel object.
    """
    def __init__(self, name: str, obj, mdl_type: str, MATLAB_available=False, parent=None):
        """
        Initializes a item for the file tree model. Recursively traverses structures with nested
        data if the model available and selected supports it.

        :param name: Name of the variable
        :type name: str
        :param obj: Variable as read by the loader.
        :type obj: object
        :param mdl_type: String with one of the supported models: "hdf5", "np_flatten", "pkl", "mat" or "csv"
        :type mdl_type: str
        :param MATLAB_available: True if MATLAB engine is available, defaults to False
        :type MATLAB_available: bool, optional
        :param parent: PArent of the current item for nested structures, defaults to None
        :type parent: FileTreeItem, optional
        :raises ValueError: Raises a ValueError if the data could not be loaded.
        """
        self.name = name
        self.obj = obj
        self.parent_item = parent
        self.child_items = []

        if mdl_type == "hdf5":
            if isinstance(obj, h5py.Group):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for key, val in obj.items():
                    self.child_items.append(FileTreeItem(key, val, mdl_type, MATLAB_available=False, parent=self))
            elif isinstance(obj, h5py.Dataset):
                if len(obj.shape) == 0:
                    try:
                        string_value = str(np.array(obj))
                        self.obj_type = f"Scalar with value {string_value}"
                        self.obj_size = np.array(obj)
                    except Exception:
                        self.obj_type = "Scalar"
                        self.obj_size = np.array(obj)
                else:
                    self.obj_type = "Dataset"
                    self.obj_size = obj.shape
        if mdl_type == "np_flatten":
            if isinstance(obj, dict):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for var_name, var_value in obj.items():
                    if not var_name.startswith('__'):
                        self.child_items.append(FileTreeItem(var_name, var_value, mdl_type, MATLAB_available=False, parent=self))
            elif isinstance(obj, np.ndarray):
                if len(obj.shape) < 1:
                    # Unknown type.
                    try:
                        string_value = str(obj)
                        self.obj_type = f"Unknown object with value {string_value}"
                        self.obj_size = -1
                    except Exception:
                        self.obj_type = f"Unknown object"
                        self.obj_size = -1
                else:
                    # This is an array or matrix.
                    self.obj_type = "Dataset"
                    self.obj_size = obj.shape
        if mdl_type == "pkl":
            if isinstance(obj, dict):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for var_name, var_value in obj.items():
                    if not var_name.startswith('__'):
                        self.child_items.append(FileTreeItem(var_name, var_value, mdl_type, MATLAB_available=False, parent=self))
            elif isinstance(obj, int):
                self.obj_type = "Scalar"
                self.obj_size = -1
            elif isinstance(obj, np.ndarray):
                # This is an array or matrix.
                self.obj_type = "Dataset"
                self.obj_size = obj.shape
            elif isinstance(obj, list):
                # This is an array or matrix.
                self.obj_type = "PythonList"
                self.obj_size = len(obj)
            elif isinstance(obj, str):
                # This is a struct.
                self.obj_type = "String"
                self.obj_size = -1
            elif isinstance(obj, float):
                # This is a struct.
                self.obj_type = "Scalar"
                self.obj_size = -1
            elif MATLAB_available:
                try:
                    print('trying')
                    import matlab
                    if isinstance(obj, matlab.double):
                        self.obj_type = "MatlabDouble"
                        self.obj_size = -1
                    elif isinstance(obj, matlab.logical):
                        self.obj_type = "MatlabLogical"
                        self.obj_size = -1 
                    else:
                        # Unknown type.
                        self.obj_type = f"Unknown {str(type(obj))}"
                        self.obj_size = -1
                except ImportError as exc:
                    # Unknown type.
                    self.obj_type = f"Unknown {str(type(obj))}"
                    self.obj_size = -1
            else:
                # Unknown type.
                self.obj_type = f"Unknown {str(type(obj))}"
                self.obj_size = -1
        if mdl_type == "mat":
            if isinstance(obj, dict):
                self.obj_type = "Group"
                self.obj_size = len(obj)
                for var_name, var_value in obj.items():
                    if not var_name.startswith('__'):
                        self.child_items.append(FileTreeItem(var_name, var_value, mdl_type, MATLAB_available=False, parent=self))
            if isinstance(obj, np.ndarray):
                #print(obj[0])
                if obj.dtype == 'O':  # Object array (likely cell array)
                    #This is a cell array.
                    self.obj_type = "Group"
                    self.obj_size = obj.shape[0]
                    if not isinstance(obj, np.ndarray) or obj.dtype != 'O':
                        raise ValueError("The input is not a MATLAB cell array represented as a NumPy object array.")
                    for i in range(obj.shape[0]):
                        key = f"element_{i}"
                        self.child_items.append(FileTreeItem(key, obj[i], mdl_type, MATLAB_available=False, parent=self))
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
                self.obj_type = f"Unknown {str(type(obj))}"
                self.obj_size = -1
        if mdl_type == "csv":
            if self.name == "/":
                self.obj_type = "Group"
                self.obj_size = 1
                self.child_items.append(FileTreeItem("CSV_dataset", obj, mdl_type, MATLAB_available=False, parent=self))
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
        """
        Returns the path of the variable in the file. Nested structures are separated
        using the character "/"

        :return: The path to the current variable
        :rtype: string
        """
        if self.parent_item:
            return self.parent_item.item_path() + "/" + self.name
        else:
            return self.name
    
    def item_type(self):
        return self.obj_type
    
    def item_size(self):
        """
        Returns the shape of the current variable. If the variable is not a matrix
        then the value -1 is returned.

        :return: Shape of the variable
        :rtype: tuple
        """
        return self.obj_size

    def parent(self):
        return self.parent_item

class FileTreeModel(QAbstractItemModel):
    """
    Generates an abstract item model to be used in a PyQt6 tree widget.
    This produces a model with the structure of the variables in a file.

    :param QAbstractItemModel: QAbstractItemModel from PyQt6
    :type QAbstractItemModel: QAbstractItemModel
    """
    def __init__(self, hdf5_file, model_type, parent=None, MATLAB_available=False):
        """
        Initializes the Tree model by passing the file, model type parent and MATLAB
        engine availability to the recursive function that generates each element in the tree.

        :param hdf5_file: Object from the data loader. May be a HDF5 file or different.
        :type hdf5_file: Object
        :param model_type: String with one of the supported models: "hdf5", "np_flatten", "pkl", "mat" or "csv"
        :type model_type: str
        :param parent: Initial parent for the first element in the tree, defaults to None
        :type parent: FileTreeItem, optional
        :param MATLAB_available: Boolean describing the availability of the MATLAB engine, defaults to False
        :type MATLAB_available: bool, optional
        """
        
        super(FileTreeModel, self).__init__(parent)
        self.root_item = FileTreeItem("/", hdf5_file, model_type, MATLAB_available)

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
