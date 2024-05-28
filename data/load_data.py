import h5py
from PyQt6.QtCore import Qt, QAbstractItemModel, QModelIndex

class HDF5TreeItem:
    def __init__(self, name, obj, parent=None):
        self.name = name
        self.obj = obj
        self.parent_item = parent
        self.child_items = []

        if isinstance(obj, h5py.Group):
            for key, val in obj.items():
                self.child_items.append(HDF5TreeItem(key, val, self))

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

    def parent(self):
        return self.parent_item

class HDF5TreeModel(QAbstractItemModel):
    def __init__(self, hdf5_file, parent=None):
        super(HDF5TreeModel, self).__init__(parent)
        self.root_item = HDF5TreeItem("/", hdf5_file)

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
            return "HDF5 File Structure"
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
