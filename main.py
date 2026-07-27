# ==============================================================================
# FLOWFI: FLOW CYTOMETRY FEATURE IMPORTANCE & DESIGN APPLICATION (v1.6.0)
# ==============================================================================


# ==============================================================================
# SYSTEM & GUI INITIALIZATION / CONFIGURATION UTILITIES
# ==============================================================================
import sys
import os
import csv
import json
import re
import time
import traceback
import ctypes

FLOWFI_VERSION = "1.6.0"

# Tell Windows this is a unique application, not just a generic Python script
if sys.platform == "win32":
    myappid = f'flowfi.cytometry.featureimportance.app.{FLOWFI_VERSION}'
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"Failed to set taskbar ID: {e}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QCheckBox, QPushButton, QProgressBar, QLabel, QFileDialog, QScrollArea, QFrame,
                             QAction, QMessageBox,QComboBox,QTabWidget,QTabBar,QSplitter, QFileSystemModel,QTreeView,QSlider,QMenu,QTextEdit,QSizePolicy,QDialog,QInputDialog,QActionGroup, QDialogButtonBox, QGridLayout, QProgressDialog, QSplashScreen)

from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QDir, QSize

from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QBrush, QPen, QColor,QIcon

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_app_dir():
    """Returns the base application directory (works for source and PyInstaller frozen app)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def get_user_data_dir():
    r"""Returns the OS-appropriate user data directory:
    - Windows: %APPDATA%/FlowFI (e.g. C:\Users\<user>\AppData\Roaming\FlowFI)
    - macOS: ~/Library/Application Support/FlowFI
    - Linux/UNIX: ~/.config/FlowFI or $XDG_CONFIG_HOME/FlowFI
    """
    if sys.platform == 'win32':
        app_data = os.environ.get('APPDATA', os.path.expanduser('~'))
        target_dir = os.path.join(app_data, 'FlowFI')
    elif sys.platform == 'darwin':
        target_dir = os.path.expanduser('~/Library/Application Support/FlowFI')
    else:
        # Linux / UNIX standard (XDG_CONFIG_HOME)
        xdg_config = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        target_dir = os.path.join(xdg_config, 'FlowFI')

    try:
        os.makedirs(target_dir, exist_ok=True)
        return target_dir
    except Exception:
        import tempfile
        return tempfile.gettempdir()

def get_config_path():
    """Returns path to flowfi_config.json, preferring user data dir in frozen/installed mode or if app dir is read-only."""
    try:
        app_dir = get_app_dir()
        if getattr(sys, 'frozen', False):
            return os.path.join(get_user_data_dir(), 'flowfi_config.json')

        primary_config = os.path.join(app_dir, 'flowfi_config.json')
        if os.path.exists(primary_config):
            return primary_config
        try:
            test_file = os.path.join(app_dir, '.writable_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return primary_config
        except Exception:
            return os.path.join(get_user_data_dir(), 'flowfi_config.json')
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), 'flowfi_config.json')

def load_flowfi_config():
    config_path = get_config_path()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_flowfi_config(config_dict):
    config_path = get_config_path()
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False

if __name__ == '__main__':
    # Initialize app and show splash screen before heavy imports
    app = QApplication(sys.argv)
    splash = None
    splash_path = resource_path('flowfi_logo_white.png')
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.showMessage(f"Loading FlowFI v{FLOWFI_VERSION}...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
        splash.show()
        app.processEvents()
else:
    app = None
    splash = None

from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import mutual_info_regression

from scipy.ndimage import binary_fill_holes, binary_erosion, distance_transform_edt

from skimage.filters import threshold_otsu, gaussian

from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects

from skimage.measure import label, regionprops

from skimage.feature import peak_local_max, canny



# ==============================================================================
# THIRD-PARTY SCIENTIFIC & IMAGE PROCESSING IMPORTS
# ==============================================================================
import numpy as np

import pandas as pd # Example: Though unused directly, good to have for clarity if debugging

from scipy.stats import kendalltau, skew, binned_statistic

from scipy.sparse import csr_matrix

from sklearn_extra.cluster import KMedoids

from sklearn.cluster import AgglomerativeClustering

import matplotlib

import matplotlib.colors as mcolors

import tifffile

import cv2

from skimage.segmentation import watershed



# ==============================================================================
# GLOBAL CONSTANTS & CONFIGURATION DEFAULTS
# ==============================================================================

EVAL = False

BOOT = 1000

CLUSTERS = 3

MEDS = CLUSTERS

BOOTSIZE = 200

THRESHOLD = 1e-5

alpha = 5

BOOTSTAT = 10000

FOOTPRINT = disk(4)

SQUARE = np.ones((4,4))

NOISETHRESHOLD = 0.0

LEFTCROP = 10

RIGHTCROP = 0

TOPCROP = 0

BOTTOMCROP = 0 

NOWAVEFRONT = 0

SMALL = 20

DISTANCES = [1,2]

ANGLES = [0,np.pi/2]

PROPERTIES = ['ASM']

AKERNEL = (5, 3) # OpenCV takes (width, height)

SIG_X = 5/2 # Suggested based on the 3x5 filter

SIG_Y = .8

PAR = 1.2

PAR /= 4. 

KERNEL = (3,3)

SIG2 = 3/2
# K = 5 # Recommended in original paper 2007
K = 15 # For similarity to standard UMAP parameters

matplotlib.use('Qt5Agg')

excludedcols = ['Saturated', 'Time', 'Sorted', 'Row', 'Column']

excludedcols += ['Protocol', 'EventLabel', 'Regions0', 'Regions1', 'Regions2',
       'Regions3', 'Gates', 'IndexSort', 'SaturatedChannels', 'PhaseOffset',
       'PlateLocationX', 'PlateLocationY', 'EventNumber0', 'EventNumber1',
       'DeltaTime0', 'DeltaTime1', 'DropId', 'SaturatedChannels1',
       'SaturatedChannels2', 'SpectralEventWidth', 'EventWidthInDrops',
       'SpectralUnmixingFlags', 'WaveformPresent',
       'sample_id', 'sample_ids', 'sample', 'id', 'filename', 'index', 'unnamed: 0']



# ==============================================================================
# FEATURE IMPORTANCE & STATISTICAL METRICS
# ==============================================================================
def get_similaritymatrix(X, k=15, t=1.0, mode='cosine', chunk_size=5000):
    n = X.shape[0]
    k_eff = min(k + 1, n - 1)
    use_sparse = n > 2000
    
    rows, cols, vals = [], [], []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        
        D_chunk = pairwise_distances(chunk, X, metric='cosine' if mode=='cosine' else 'euclidean')
        
        partition_indices = np.argpartition(D_chunk, k_eff, axis=1)[:, :k_eff]
        
        row_idx = np.arange(D_chunk.shape[0])[:, None]
        nearest_dists = D_chunk[row_idx, partition_indices]
        
        for i in range(end - start):
            global_row = start + i
            rows.extend([global_row] * k_eff)
            cols.extend(partition_indices[i])
            vals.extend(nearest_dists[i])

    rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
    if mode == 'heat':
        weights = np.exp(-vals**2 / (2 * t**2))
    else:
        weights = np.abs(1 - vals)

    if use_sparse:
        W = csr_matrix((weights, (rows, cols)), shape=(n, n))
        W = W.maximum(W.T)
        W.setdiag(0)
    else:
        W = np.zeros((n, n))
        W[rows, cols] = weights
        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0)
        
    return W


def lsRI_metric(X, numf, k=5, t=1.0, mode='cosine'):
    n = X.shape[0]
    W = get_similaritymatrix(X, k=k, t=t, mode=mode)
    
    d_vec = np.array(W.sum(axis=1)).flatten()
    D_sum = d_vec.sum()
    
    weighted_means = (X.T @ d_vec) / D_sum
    X_centered = X - weighted_means
    
    den = np.sum((X_centered**2).T * d_vec, axis=1)
    
    fWf = np.sum(X_centered * (W @ X_centered), axis=0)
    num = den - fWf
    
    ls = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return ls


def pRI_metric(sample, numf, threshold=0.8):    
    pca = PCA().fit(sample)    
    k = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= threshold) + 1    
    return np.dot(pca.explained_variance_ratio_[:k], np.abs(pca.components_[:k]))


def sRI_metric(sample, numf, grid_size=5, iterations=1000):
    from minisom import MiniSom
    som = MiniSom(grid_size, grid_size, numf, 
                  sigma=1.0, 
                  learning_rate=0.05, 
                  neighborhood_function='gaussian')
    
    som.train_batch(sample, iterations)
    
    node_weights = som.get_weights().reshape(grid_size*grid_size, numf)
    
    k_meta = 5 
    clusterer = AgglomerativeClustering(n_clusters=k_meta, linkage='ward')
    metacluster_indices = clusterer.fit_predict(node_weights)
    
    meta_medians = []
    meta_iqrs = []
    
    for c in range(k_meta):
        mask = metacluster_indices == c
        if np.any(mask):
            meta_medians.append(np.median(node_weights[mask], axis=0))
            meta_iqrs.append(np.percentile(node_weights[mask], 75, axis=0) - 
                             np.percentile(node_weights[mask], 25, axis=0))
    
    meta_medians = np.array(meta_medians)


    global_variability = np.std(meta_medians, axis=0)
    avg_within_cluster_spread = np.mean(meta_iqrs, axis=0) + 1e-10
    
    return global_variability / avg_within_cluster_spread


def miRI_metric(X, numf):
    mi_scores = np.zeros(numf)
    for i in range(numf):
        X_other = np.delete(X, i, axis=1)
        y_current = X[:, i]
        
        shared_info = mutual_info_regression(X_other, y_current, n_neighbors=15)
        
        mi_scores[i] = np.mean(shared_info)
        
    return mi_scores



# ==============================================================================
# AUXILIARY PYQT WIDGETS & WORKER THREADS
# ==============================================================================
class OperationHistory(QWidget):
    info_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_text_edit = QTextEdit()
        self.history_text_edit.setReadOnly(True)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Operation History:"))
        self.layout.addWidget(self.history_text_edit)
        self.setLayout(self.layout)

    def add_operation(self, operation_description):
        current_text = self.history_text_edit.toPlainText()
        new_text = f"{current_text}\n{operation_description}".strip()
        self.history_text_edit.setText(new_text)
        self.history_text_edit.verticalScrollBar().setValue(self.history_text_edit.verticalScrollBar().maximum())

    def update_info(self,info):
        self.info_updated.emit(info)


class BarWidget(QWidget):
    def __init__(self, mean, color, low_ci=None, upper_ci=None, stroke_color="black", parent=None):
        super().__init__(parent)
        self.mean = mean
        self.color = color
        self.low_ci = low_ci
        self.upper_ci = upper_ci
        self.stroke_color = stroke_color
        self.setFixedHeight(20)
        self.setFixedWidth(300)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Dimensions
        ci_h = 14
        solid_h = 6
        
        y_ci = (h - ci_h) // 2
        y_solid = (h - solid_h) // 2
        
        c = QColor(self.color)
        
        # Handle NaNs by checking if values are finite
        if self.low_ci is not None and self.upper_ci is not None and np.isfinite(self.low_ci) and np.isfinite(self.upper_ci):
            l = max(0, min(1, self.low_ci)) if np.isfinite(self.low_ci) else 0
            u = max(0, min(1, self.upper_ci)) if np.isfinite(self.upper_ci) else 0
            m = max(0, min(1, self.mean)) if np.isfinite(self.mean) else 0
            
            x_start = int(l * w)
            x_end = int(u * w)
            x_mean = int(m * w)
            
            # Draw solid bar from 0 to mean (thinner)
            painter.setBrush(QBrush(c))
            painter.setPen(Qt.NoPen)
            if x_mean > 0:
                painter.drawRect(0, y_solid, x_mean, solid_h)
            
            # Draw transparent bar from low_ci to upper_ci (thicker)
            c_trans = QColor(c)
            c_trans.setAlpha(100)
            painter.setBrush(QBrush(c_trans))
            width_ci = max(x_end - x_start, 2) # Ensure at least 2px width
            painter.drawRect(x_start, y_ci, width_ci, ci_h)
            
            # Draw Mean stroke
            s_c = QColor(self.stroke_color)
            painter.setPen(QPen(s_c, 3))
            painter.drawLine(x_mean, y_ci - 2, x_mean, y_ci + ci_h + 2)
            
        else:
            m = max(0, min(1, self.mean)) if np.isfinite(self.mean) else 0
            width_bar = int(m * w)
            painter.setBrush(QBrush(c))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, y_solid, width_bar, solid_h)


class WorkerThread(QThread):
    progress_update = pyqtSignal(int)
    intermediate_result = pyqtSignal(dict)
    result_ready = pyqtSignal()
    def __init__(self, data, boots=BOOT, bootsize=BOOTSIZE, conv_check=True, conv_threshold=THRESHOLD, metric_name="lsRI"):
        super().__init__()
        self.data = data
        self.metric_name = metric_name
        
        N = self.data.shape[0]
        self.n = bootsize
        self.boots = boots
        if N<self.n:
            self.n = int(max([N / 2, 2]))
            self.boots = N
        # self.k = max([int(self.n*KFRAC),1])
        self.k = K
        self.mode = 'cosine'
        self.t = 1
        self.progress = 0
        self.early = 0
        self.conv_check = conv_check
        self.conv_threshold = conv_threshold
        
        # Initialize accumulators for convergence check
        self.feature_averages = np.zeros((self.data.shape[1], self.boots))
        self.calculated = np.zeros((self.boots))
        self.medoids = np.zeros((self.data.shape[1], self.boots))
        self.memberships = np.zeros((self.data.shape[1], self.boots))
    def run(self):
        for i in range(self.boots):
            result = self.process_part(i)
            
            # Accumulate results
            value = result['value']
            self.medoids[list(result['medoids'].astype(int)), i] += 1
            self.memberships[:, i] = result['membership']
            self.feature_averages[:, i] = value
            self.calculated[i] = 1
            
            # Convergence check
            if self.conv_check and np.sum(self.calculated) > 10:
                non0 = self.calculated > 0
                imp_calculated = self.feature_averages[:, non0]
                isconv, inds1, inds2 = self.splittest(imp_calculated, th=self.conv_threshold)
                if isconv:
                    isclust = self.consensusclustering_test(inds1, inds2, th=self.conv_threshold)
                    if isclust:
                        self.early = 1

            self.intermediate_result.emit(result)
            self.progress += 1
            if self.early:
                break
        self.result_ready.emit()
    def process_part(self, i):
        sample_idx = np.random.choice(self.data.shape[0], self.n)
        Xsub = self.data[sample_idx, :]
        
        if self.metric_name == "pRI":
            val = pRI_metric(Xsub, self.data.shape[1], threshold=0.8)
        elif self.metric_name == "sRI":
            val = sRI_metric(Xsub, self.data.shape[1])
        elif self.metric_name == "miRI":
            val = miRI_metric(Xsub, self.data.shape[1])
        else:
            val = lsRI_metric(Xsub, self.data.shape[1], k=self.k, t=self.t, mode=self.mode)
            
        medoids, medlabels = self.kmedoids(Xsub.T)
        return {"value": val, "i": i, "medoids": medoids, "membership": medlabels}
    def getclust(self,mems):
        import leidenalg as la
        import igraph as ig
        memlabels = np.unique(mems.flatten())
        D = np.zeros([mems.shape[0],mems.shape[0]])
        for m in memlabels:
            mem = (mems == m)*1.
            D += mem @ mem.T
        np.fill_diagonal(D,0)
        return np.array(la.find_partition(ig.Graph.Adjacency(D), la.ModularityVertexPartition).membership)
    def kmedoids(self,X):
        if CLUSTERS*10>=self.data.shape[1]:
            clusters = CLUSTERS
        else:
            clusters = int(self.data.shape[1]/10)
        model = KMedoids(n_clusters=clusters,method='pam').fit(X)
        medoids = model.medoid_indices_
        medlabels = model.labels_
        return medoids,medlabels
    def splittest(self,data,th):
        shape = data.shape[1]
        inds = np.arange(shape)
        np.random.shuffle(inds)
        data = data[:,inds]
        splitat = int(shape/2)
        inds1 = inds[:splitat]
        inds2 = inds[splitat:]
        data1 = np.mean(data[:,inds1],axis=1)
        data2 = np.mean(data[:,inds2],axis=1)
        kt = kendalltau(data1,data2)
        kt = kt.statistic
        if 1-kt<=th:
            return True,inds1,inds2
        else:
            return False,inds1,inds2
    def consensusclustering_test(self,inds1,inds2,th):
        mems = self.memberships[:,self.calculated>0]
        mems1 = mems[:,inds1]
        mems2 = mems[:,inds2]
        membership1 = self.getclust(mems1)
        membership2 = self.getclust(mems2)
        ami = adjusted_mutual_info_score(membership1,membership2)
        return ami>th
    def consensusclustering_test(self,inds1,inds2,th):
        mems = self.memberships[:,self.calculated>0]
        mems1 = mems[:,inds1]
        mems2 = mems[:,inds2]
        membership1 = self.getclust(mems1)
        membership2 = self.getclust(mems2)
        ami = adjusted_mutual_info_score(membership1,membership2)
        return ami>th


class DynamicTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setExpanding(True)

    def tabSizeHint(self, index):
        if self.shape() in (QTabBar.RoundedWest, QTabBar.TriangularWest, QTabBar.RoundedEast, QTabBar.TriangularEast):
            total_height = max(180, self.height())
            count = max(1, self.count())
            tab_height = total_height // count
            font_metrics = self.fontMetrics()
            text_width = font_metrics.width(self.tabText(index))
            tab_width = max(180, text_width + 55)
            # For West/East vertical tabs, Qt transposes QSize(w, h)!
            # First param is vertical height, second param is horizontal width.
            return QSize(tab_height, tab_width)
        return super().tabSizeHint(index)



# ==============================================================================
# DIALOG CLASSES
# ==============================================================================
class AlphaDialog(QDialog):
    def __init__(self, parent=None, default_alpha=5.0, default_boots=10000):
        super().__init__(parent)
        self.setWindowTitle("Set CI Parameters")
        self.alpha = default_alpha
        self.boots = default_boots

        layout = QVBoxLayout(self)
        
        input_layout = QHBoxLayout()
        label = QLabel("Alpha (%):")
        self.alpha_edit = QLineEdit(str(self.alpha))
        input_layout.addWidget(label)
        input_layout.addWidget(self.alpha_edit)
        layout.addLayout(input_layout)

        boots_layout = QHBoxLayout()
        boots_label = QLabel("Bootstrap Size:")
        self.boots_edit = QLineEdit(str(self.boots))
        boots_layout.addWidget(boots_label)
        boots_layout.addWidget(self.boots_edit)
        layout.addLayout(boots_layout)

        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        self.setLayout(layout)

    def get_alpha(self):
        return self.alpha

    def get_boots(self):
        return self.boots

    def accept(self):
        try:
            val = float(self.alpha_edit.text())
            boots_val = int(self.boots_edit.text())
            if 0 < val < 50 and boots_val > 0:
                self.alpha = val
                self.boots = boots_val
                super().accept()
            else:
                QMessageBox.warning(self, "Invalid Input", "Alpha must be strictly between 0 and 50, and Bootstrap Size must be positive.", QMessageBox.Ok)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers.", QMessageBox.Ok)


def parse_tuple_input(text, is_int=False):
    cleaned = text.strip().lstrip('(').rstrip(')').strip()
    if not cleaned:
        raise ValueError("Empty input")
    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
    if is_int:
        vals = [int(p) for p in parts]
    else:
        vals = [float(p) for p in parts]
    
    if len(vals) == 1:
        return (vals[0], vals[0])
    elif len(vals) >= 2:
        return (vals[0], vals[1])
    else:
        raise ValueError("Invalid tuple values")


class GaussDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Blur Filter")
        self.params = {"kernel": [3, 3], "sigmaX": 2.0, "sigmaY": 2.0}

        layout = QVBoxLayout(self)

        # Label and Line Edit for Kernel Size tuple
        kernel_layout = QHBoxLayout()
        kernel_label = QLabel("Kernel Size (e.g. (3, 3)):")
        self.kernel_edit = QLineEdit("(3, 3)")
        self.kernel_edit.setPlaceholderText("e.g. (3, 3)")
        kernel_layout.addWidget(kernel_label)
        kernel_layout.addWidget(self.kernel_edit)
        layout.addLayout(kernel_layout)

        # Label and Line Edit for Sigma tuple
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma (e.g. (1.5, 0.5)):")
        self.sigma_edit = QLineEdit("(2.0, 2.0)")
        self.sigma_edit.setPlaceholderText("e.g. (1.5, 0.5) or (2.0, 2.0)")
        sigma_layout.addWidget(sigma_label)
        sigma_layout.addWidget(self.sigma_edit)
        layout.addLayout(sigma_layout)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addLayout(button_box)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        self.setLayout(layout)

    def get_values(self):
        return self.params

    def get_sigma(self):
        return self.params.get("sigmaX", 2.0)

    def accept(self):
        try:
            k_text = self.kernel_edit.text()
            s_text = self.sigma_edit.text()

            # Parse kernel tuple
            kw, kh = parse_tuple_input(k_text, is_int=True)
            if kw <= 0 or kh <= 0:
                QMessageBox.warning(self, "Invalid Input", "Kernel dimensions must be positive integers.", QMessageBox.Ok)
                return
            if kw % 2 == 0: kw += 1
            if kh % 2 == 0: kh += 1

            # Parse sigma tuple
            sigX, sigY = parse_tuple_input(s_text, is_int=False)
            if sigX <= 0 or sigY <= 0:
                QMessageBox.warning(self, "Invalid Input", "Sigma values must be greater than zero.", QMessageBox.Ok)
                return

            self.params = {
                "kernel": [kw, kh],
                "sigmaX": float(sigX),
                "sigmaY": float(sigY)
            }
            super().accept()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid tuple formats, e.g. (3, 3) for kernel and (1.5, 0.5) for sigmas.", QMessageBox.Ok)
            return


class MultiChannelDialog(QDialog):
    def __init__(self, channel_roles, num_channels, parent=None, disable_snr_checks=False):
        super().__init__(parent)
        self.setWindowTitle("Select Channels for Operation")
        self.channel_roles = channel_roles
        self.num_channels = num_channels
        self.channel_combos = {}
        self.snr_checkboxes = {}

        layout = QVBoxLayout(self)
        channel_options = [str(i + 1) for i in range(self.num_channels)]

        for role in self.channel_roles: # e.g. 'Signal', 'Mask', 'Global Mask (Optional)'
            row_layout = QHBoxLayout()
            label = QLabel(f"{role} Channel:")
            combo = QComboBox()
            
            if "(Optional)" in role:
                combo.addItems(["None"] + channel_options)
            else:
                combo.addItems(channel_options)

            self.channel_combos[role] = combo

            row_layout.addWidget(label)
            row_layout.addWidget(combo)

            if "Global Mask (Optional)" not in role and not disable_snr_checks:
                snr_check = QCheckBox("SNR Check")
                snr_check.setChecked(True)
                self.snr_checkboxes[role] = snr_check
                row_layout.addWidget(snr_check)

            layout.addLayout(row_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_channels(self):
        """Returns a dictionary of role -> channel_index and snr_check states."""
        selections = {}
        snr_checks = {}
        for role, combo in self.channel_combos.items():
            # Handle channel selection
            is_optional = "(Optional)" in role
            if is_optional and combo.currentIndex() == 0:
                selections[role] = None  # Store None for the "None" option
            else:
                # Adjust index for optional roles that have "None" at the start
                offset = 1 if is_optional else 0
                selections[role] = combo.currentIndex() - offset  # 0-indexed channel

            # Handle SNR checkbox state
            if role in self.snr_checkboxes:
                snr_checks[role] = self.snr_checkboxes[role].isChecked()
        selections['snr_checks'] = snr_checks
        return selections


class CropDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Crop Values")

        layout = QGridLayout(self)

        self.top_edit = QLineEdit("0")
        self.bottom_edit = QLineEdit("0")
        self.left_edit = QLineEdit("0")
        self.right_edit = QLineEdit("0")

        layout.addWidget(QLabel("Top:"), 0, 0)
        layout.addWidget(self.top_edit, 0, 1)
        layout.addWidget(QLabel("Bottom:"), 1, 0)
        layout.addWidget(self.bottom_edit, 1, 1)
        layout.addWidget(QLabel("Left:"), 2, 0)
        layout.addWidget(self.left_edit, 2, 1)
        layout.addWidget(QLabel("Right:"), 3, 0)
        layout.addWidget(self.right_edit, 3, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 4, 0, 1, 2)

    def get_values(self):
        try:
            top = int(self.top_edit.text())
            bottom = int(self.bottom_edit.text())
            left = int(self.left_edit.text())
            right = int(self.right_edit.text())
            return top, bottom, left, right
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for all crop values.")
            return None, None, None, None

    def accept(self):
        values = self.get_values()
        if all(v is not None for v in values):
            super().accept()


class RescaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Rescale Values")

        layout = QGridLayout(self)

        self.scale_x_edit = QLineEdit("1.0")
        self.scale_y_edit = QLineEdit("1.0")
        self.inter_combo = QComboBox()
        self.inter_combo.addItems(['Linear', 'Nearest', 'Area', 'Cubic', 'Lanczos4'])

        layout.addWidget(QLabel("Scale X:"), 0, 0)
        layout.addWidget(self.scale_x_edit, 0, 1)
        layout.addWidget(QLabel("Scale Y:"), 1, 0)
        layout.addWidget(self.scale_y_edit, 1, 1)
        layout.addWidget(QLabel("Interpolation:"), 2, 0)
        layout.addWidget(self.inter_combo, 2, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 3, 0, 1, 2)

    def get_values(self):
        try:
            scale_x = float(self.scale_x_edit.text())
            scale_y = float(self.scale_y_edit.text())
            interpolation = self.inter_combo.currentText()
            return scale_x, scale_y, interpolation
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for scale values.")
            return None, None, None

    def accept(self):
        values = self.get_values()
        if all(v is not None for v in values):
            super().accept()


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"FlowFI v{FLOWFI_VERSION} Help")
        self.setGeometry(200, 200, 850, 600)

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab_bar = DynamicTabBar()
        self.tabs.setTabBar(self.tab_bar)
        self.tabs.setTabPosition(QTabWidget.West)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)

        # --- Create Tabs ---
        self.tabs.addTab(self.create_text_widget(self.get_workflow_text()), "Workflow")
        self.tabs.addTab(self.create_text_widget(self.get_design_text()), "Design Tab")
        self.tabs.addTab(self.create_text_widget(self.get_refine_text()), "Refine Tab")

        main_layout.addWidget(self.tabs)
        self.update_tab_styles()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_tab_styles()
        self.tab_bar.updateGeometry()

    def update_tab_styles(self):
        font_size = max(13, min(18, self.height() // 35))
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                padding: 10px 16px;
                font-size: {font_size}px;
                font-weight: bold;
            }}
        """)

    def create_text_widget(self, html_content):
        """Creates a read-only QTextEdit widget with the given HTML content."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(html_content)
        return text_edit

    def get_workflow_text(self):
        return """
        <h2>FlowFI General Workflow (v1.6.0)</h2>
        <p><b>Version:</b> 1.6.0</p>
        <p>FlowFI is a dual-purpose tool for flow and imaging cytometry analysis, combining feature ranking and feature engineering into one application.</p>
        
        <h3>Core Components:</h3>
        <ul>
            <li><b>Refine Tab:</b> Analyzes existing tabular flow cytometry data (<code>.fcs</code> files) to identify and rank the most important measurement channels (features) for describing the data's structure.</li>
            <li><b>Design Tab:</b> Provides an interactive environment to create <i>new</i> quantitative features from imaging cytometry data (<code>.tiff</code> files) by building custom image processing pipelines.</li>
        </ul>

        <h3>A Typical Use Case:</h3>
        <ol>
            <li>A researcher uses the <b>Design</b> tab to engineer a novel biological feature, such as "the symmetry of a protein signal relative to the cell's nucleus."</li>
            <li>They save or apply their preset processing pipeline to a folder of cell images, exporting the results as a new parameter in their main <code>.fcs</code> dataset via the <b>Parameters</b> menu.</li>
            <li>They then switch to the <b>Refine</b> tab, load the newly augmented <code>.fcs</code> file, and run the analysis to see how important their custom-designed feature is compared to the standard, instrument-provided measurements.</li>
        </ol>

        <p>This workflow allows for a powerful cycle of hypothesis generation (Design) and a means for quick testing of these hypotheses against existing parameters (Refine).</p>
        """

    def get_refine_text(self):
        return """
        <h2>Refine Tab Guide</h2>
        <p>This tab is used to analyze a standard flow cytometry <code>.fcs</code> file to determine the importance of its features.</p>
        
        <h3>How to Use:</h3>
        <ol>
            <li>Enter the <code>.fcs</code> file path manually or click <b>Browse</b> to select a file.</li>
            <li>Use the checkboxes at the top to include or exclude broad categories of features from the analysis.</li>
            <li>Select your desired <b>RI Metric</b> from the Refine menu (details below).</li>
            <li>Click <b>Execute</b> to start the analysis. The process involves bootstrapping and may take some time, with progress shown in the progress bar.</li>
            <li>Results will be displayed in the main panel, ranked by importance by default.</li>
        </ol>

        <h3>Relative Importance (RI) Metrics:</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>Method</th><th>Local-Global Structure</th><th>Importance Methodology</th></tr>
            <tr><td>Laplace-Scoring (lsRI)</td><td>Intermediate</td><td>Neighbourhood preservation</td></tr>
            <tr><td>PCA (pRI)</td><td>Global</td><td>Linear variability explained</td></tr>
            <tr><td>SOM (sRI)</td><td>Local</td><td>Cluster identity preservation</td></tr>
            <tr><td>Mutual Information (miRI)</td><td>Global</td><td>Maximum informational dependence</td></tr>
        </table>
        <br>

        <h3>Interpreting the Results:</h3>
        <ul>
            <li><b>Feature Name:</b> The name of the channel from the <code>.fcs</code> file.</li>
            <li><b>Importance Bar:</b> The length of the colored bar indicates the relative importance of the feature. Longer bars are more important.</li>
            <li><b>Sorting:</b> Use the dropdown menu to sort features by different criteria:
                <ul>
                    <li><b>Importance:</b> (Default) Ranks features by their chosen Relative Importance (RI) score.</li>
                    <li><b>Type:</b> Groups features by their category (e.g., UV, V, B).</li>
                    <li><b>Cluster:</b> Groups features that are algorithmically determined to be similar to each other. The border color indicates cluster membership.</li>
                    <li><b>Centrality:</b> Ranks features by how representative they are of their assigned cluster. Central features are underlined.</li>
                    <li><b>Change from Previous:</b> Compares the current run's rankings to a previously loaded CSV file.</li>
                </ul>
            </li>
        </ul>
        
        <h3>Menu Options (Refine -> ...):</h3>
        <ul>
            <li><b>RI Metric:</b> Choose the algorithm used to evaluate feature importance (lsRI, pRI, sRI, or miRI).</li>
            <li><b>Calculate Importance CIs:</b> Enables bootstrap confidence interval estimation (alpha level and bootstrap sample size).</li>
            <li><b>Save Output as CSV:</b> Saves the full results table, including raw scores, relative importance, cluster memberships, and centralities to a CSV file.</li>
            <li><b>Load Output CSV for Comparison:</b> Loads a previously saved run to enable the "Sort by: Change from Previous" comparison option.</li>
            <li><b>Preferences...:</b> Configures bootstrap iterations (BOOT), sample size (BOOTSIZE), dataset size (N), and convergence parameters.</li>
        </ul>
        """

    def get_design_text(self):
        return """
        <h2>Design Tab Guide</h2>
        <p>This tab is a workbench for creating new features from multi-channel <code>.tiff</code> images.</p>

        <h3>Basic Workflow:</h3>
        <ol>
            <li>Use the file tree on the left to navigate to and double-click a <code>.tif</code> or <code>.tiff</code> file to load it.</li>
            <li>The original image for the selected channel appears in the top-left panel. The top-right panel shows the result of preprocessing.</li>
            <li>Use the menus (<b>Preprocessing</b>, <b>Quantify</b>) to build an analysis pipeline. Operations are applied sequentially.</li>
            <li>The <b>Operation History</b> terminal shows the list of applied steps and the result of any quantification.</li>
            <li>Once a pipeline is defined, save it as a preset or use the <b>Parameters</b> menu to apply it to a whole folder of images and export the results.</li>
        </ol>

        <h3>Menu Options:</h3>
        <h4>Preprocessing Menu</h4>
        <ul>
            <li><b>Presets Submenu:</b> Manages reusable parameter and preprocessing presets.
                <ul>
                    <li><i>Save New Parameter Preset...:</i> Saves the current pipeline (preprocessing operations + quantification setting + channel configurations) into a reusable <code>.json</code> preset file.</li>
                    <li><i>Load Parameter Preset File...:</i> Browses and loads any custom <code>.json</code> preset file to instantly apply its steps and quantification settings.</li>
                    <li><i>Configure Presets Location...:</i> Customizes the active directory location where preset JSON files are stored and dynamically loaded into the Presets menu.</li>
                    <li><i>OFDM Preset:</i> Default built-in flow cytometry preprocessing preset consisting of Crop, Rescale down, Anisotropic Gaussian Blur, Rescale up, and Isotropic Gaussian Blur steps with cell count quantification.</li>
                </ul>
            </li>
            <li><b>Filter:</b> Noise reduction and smoothing operations (e.g., <i>Gaussian Filter</i> with configurable kernel size and sigma values).</li>
            <li><b>Manipulation:</b> Geometric image transformations:
                <ul>
                    <li><i>Crop:</i> Crops border pixels (Top, Bottom, Left, Right).</li>
                    <li><i>Rescale:</i> Scales X and Y dimensions using specified interpolation methods (Nearest, Linear, Area, Cubic, Lanczos4).</li>
                </ul>
            </li>
            <li><b>Segmentation:</b> Operations to isolate and partition objects of interest:
                <ul>
                    <li><i>Mask Otsu:</i> Creates a binary mask using Otsu's automatic thresholding.</li>
                    <li><i>Label Image:</i> Assigns a unique integer label to each disconnected object in a binary image.</li>
                    <li><i>Segment:</i> Uses a watershed algorithm based on Canny edge detection and distance transforms to separate touching objects.</li>
                </ul>
            </li>
            <li><b>Undo Last Operation (Ctrl+Z):</b> Removes the most recent preprocessing step from the history.</li>
            <li><b>Redo Operation (Ctrl+Y):</b> Re-applies the most recent undone preprocessing operation.</li>
            <li><b>Reset Preprocessing:</b> Clears all applied preprocessing steps for the current image.</li>
            <li><b>Save Single Image (.tiff):</b> Saves the preprocessed image for the currently selected channel to a <code>.tiff</code> file.</li>
            <li><b>Batch Process Folder:</b> Applies the active preprocessing pipeline to an entire directory of <code>.tiff</code> images and saves preprocessed TIFFs to an output subfolder.</li>
        </ul>

        <h4>Quantify Menu</h4>
        <p>Defines how the final processed image is converted into a single quantitative feature score. These options are mutually exclusive.</p>
        <ul>
            <li><b>Aggregation:</b>
                <ul>
                    <li><i>Count (unique):</i> Counts the number of unique non-zero labels in the image (useful after a 'Label' or 'Segment' step).</li>
                    <li><i>Mean (non-zero):</i> Calculates the mean intensity of all non-zero pixels.</li>
                    <li><i>Area (non-zero):</i> Counts the total number of non-zero pixels.</li>
                </ul>
            </li>
            <li><b>Geometry:</b>
                <ul>
                    <li><i>Solidity:</i> Measures the ratio of the object's area to the area of its convex hull. A perfect convex shape has a solidity of 1.</li>
                    <li><i>Colocalisation:</i> Measures the fraction of a 'Signal' channel's intensity that is within a 'Mask' channel.</li>
                    <li><i>Containment:</i> Measures the fraction of a 'Signal' channel's intensity that is inside the core of a 'Container' channel (excluding its shell).</li>
                    <li><i>Relative Skewness:</i> Measures the radial skewness of a 'Signal' relative to the centroid of a 'Reference' channel.</li>
                    <li><i>Angular Momentum:</i> Measures the angular asymmetry vector magnitude of a 'Signal' relative to the centroid of a 'Reference' channel.</li>
                    <li><i>Angular Entropy:</i> Measures the angular uniformity of a 'Signal' relative to the centroid of a 'Reference' channel (1 = perfectly uniform).</li>
                    <li><i>Spatial Correlation:</i> Calculates the Pearson correlation between two channels within a defined mask.</li>
                </ul>
            </li>
        </ul>

        <h4>Parameters Menu</h4>
        <p>Applies the complete defined pipeline (preprocessing + quantification) to a folder of images or merges tabular results.</p>
        <ul>
            <li><b>Export to FCS:</b> Applies the pipeline across all <code>.tiff</code> images in a folder and writes the calculated feature values into a new parameter column of a target <code>.fcs</code> file.</li>
            <li><b>Export to CSV:</b> Applies the pipeline across all <code>.tiff</code> images in a folder and exports calculated feature values and sample IDs to a <code>.csv</code> file.</li>
            <li><b>Export Terminal:</b> Saves all text in the Operation History terminal panel to a text file (<code>.txt</code>).</li>
            <li><b>Concatenate CSVs:</b> Merges or stacks multiple parameter CSV files (automatically merges on <code>sample_id</code> column if present, or validates dimensions for row-wise concatenation).</li>
            <li><b>Merge CSV into FCS:</b> Merges an N x P parameter matrix from a CSV file directly into an N-event base <code>.fcs</code> file template.</li>
        </ul>
        """


class RefinePreferencesDialog(QDialog):
    def __init__(self, parent=None, default_boots=1000, default_bootsize=200, dataset_size=None, default_conv_check=True, default_conv_threshold=1e-5):
        super().__init__(parent)
        self.setWindowTitle("Refine Preferences")

        layout = QGridLayout(self)

        self.boots_edit = QLineEdit(str(default_boots))
        self.bootsize_edit = QLineEdit(str(default_bootsize))

        layout.addWidget(QLabel("Bootstrap Iterations (BOOT):"), 0, 0)
        layout.addWidget(self.boots_edit, 0, 1)
        layout.addWidget(QLabel("Bootstrap Sample Size (BOOTSIZE):"), 1, 0)
        layout.addWidget(self.bootsize_edit, 1, 1)
        
        layout.addWidget(QLabel("Dataset Size (N):"), 2, 0)
        self.dataset_size_edit = QLineEdit()
        if dataset_size is not None:
            self.dataset_size_edit.setText(str(dataset_size))
        layout.addWidget(self.dataset_size_edit, 2, 1)

        self.conv_check_box = QCheckBox("Enable Convergence Check")
        self.conv_check_box.setChecked(default_conv_check)
        self.conv_check_box.stateChanged.connect(self.toggle_threshold_input)
        layout.addWidget(self.conv_check_box, 3, 0, 1, 2)

        layout.addWidget(QLabel("Convergence Threshold (Epsilon):"), 4, 0)
        self.threshold_edit = QLineEdit(str(default_conv_threshold))
        layout.addWidget(self.threshold_edit, 4, 1)
        self.toggle_threshold_input()

        self.calc_coverage_btn = QPushButton("Calculate Expected Coverage")
        self.calc_coverage_btn.clicked.connect(self.calculate_coverage)
        layout.addWidget(self.calc_coverage_btn, 5, 0, 1, 2)

        self.coverage_label = QLabel("Expected Coverage: -")
        self.coverage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.coverage_label, 6, 0, 1, 2)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 7, 0, 1, 2)

    def toggle_threshold_input(self):
        self.threshold_edit.setEnabled(self.conv_check_box.isChecked())

    def calculate_coverage(self):
        try:
            boots = int(self.boots_edit.text())
            bootsize = int(self.bootsize_edit.text())
            N = int(self.dataset_size_edit.text())
            notinboot = (1-1./N)**bootsize
            coverage = 1-notinboot**boots
            
            self.coverage_label.setText(f"Expected Coverage: {coverage:.4f}")
        except ValueError:
            self.coverage_label.setText("Error: Invalid input(s)")

    def get_values(self):
        try:
            boots = int(self.boots_edit.text())
            bootsize = int(self.bootsize_edit.text())
            conv_check = self.conv_check_box.isChecked()
            threshold = float(self.threshold_edit.text())
            return boots, bootsize, conv_check, threshold
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for all values.")
            return None, None, None, None

    def accept(self):
        boots, bootsize, conv_check, threshold = self.get_values()
        if boots is not None and bootsize is not None:
            if boots > 0 and bootsize > 0:
                if conv_check and not (0 <= threshold <= 1):
                    QMessageBox.warning(self, "Invalid Input", "Convergence threshold must be between 0 and 1.")
                    return
                super().accept()
            else:
                QMessageBox.warning(self, "Invalid Input", "Values must be positive integers.")



# ==============================================================================
# MAIN APPLICATION WINDOW (MainWindow)
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FlowFI: Flow cytometry Feature Importance")
        logo_path = resource_path("logo.png")
        if os.path.exists(logo_path):
            print(f"Path to icon exists: {logo_path}") # Confirms path resolution and file existence
            icon = QIcon(logo_path)
            if not icon.isNull():
                print("Successfully loaded icon!")
                self.setWindowIcon(icon)
            else:
                print(f"QIcon failed to load image despite path existence. Potential issue with file format ({logo_name}) or file corrupt/unreadable by Qt.")
        else:
            print(f"Icon path not found: {logo_path}") # Confirms path issue (likely post-packaging typo or function logic)


        self.setGeometry(100, 100, 800, 600)
        self.boots_param = BOOT
        self.bootsize_param = BOOTSIZE
        self.ci_alpha = alpha
        self.ci_boots = BOOTSTAT
        self.convergence_check = True
        self.convergence_threshold = THRESHOLD

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        
        self.tabs.addTab(self.tab2, " Design ")
        self.tabs.addTab(self.tab1, " Refine ")
        self.tab1.layout = QVBoxLayout(self.tab1)
        self.tab2.layout = QVBoxLayout(self.tab2)
        self.layout.addWidget(self.tabs)

        # TAB LAYOUT: ANALYSIS


        # Input field for filepath
        self.filepath_input = QLineEdit()
        self.filepath_input.setPlaceholderText("Enter file path here")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)

        self.input_layout = QHBoxLayout()
        self.input_layout.addWidget(self.filepath_input)
        self.input_layout.addWidget(self.browse_button)


        # Button to execute the function
        self.execute_button = QPushButton("Execute")
        self.execute_button.clicked.connect(self.execute_function)

        self.checkbox_layout = QHBoxLayout()
        self.ftypes = ['UV','V','B','YG','R','ImgB','Imaging','Misc']
        self.colors = ['green','darkviolet','blue','darkgoldenrod','darkred','saddlebrown','teal','black']
        self.clustercolors = ['lightcoral','palegoldenrod','palegreen','lightblue','aquamarine','dimgray','peru','darkseagreen','white','cornflowerblue','green','darkviolet','blue','darkgoldenrod','darkred','saddlebrown','teal','black']
        self.selected_feature_types = self.ftypes
        self.feature_checkboxes = {}
        for i,feature_type in enumerate(self.ftypes):
            checkbox = QCheckBox(feature_type)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_display)
            checkbox.setStyleSheet("color: " + self.colors[i])
            self.feature_checkboxes[feature_type] = checkbox
            self.checkbox_layout.addWidget(checkbox)
        
        centrality_checkbox = QCheckBox('CEN ONLY')
        centrality_checkbox.setChecked(False)
        centrality_checkbox.stateChanged.connect(self.update_display)
        self.centrality_checkbox = centrality_checkbox
        self.checkbox_layout.addWidget(self.centrality_checkbox)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Output display panel
        self.output_panel = QScrollArea()
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()

        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)
        self.output_panel.setWidgetResizable(True)

        # Sorting dropdown box
        self.sort_dropdown = QComboBox()
        self.sort_dropdown.addItem("Sort by: Importance (features that are important to the data structure)")
        self.sort_dropdown.addItem("Sort by: Type (UV, V, etc.)")
        self.sort_dropdown.addItem("Sort by: Cluster (similar features)")
        self.sort_dropdown.addItem("Sort by: Centrality (featuress typical of a cluster)")
        self.sort_dropdown.addItem("Sort by: Change from Previous Importance (contrast scores against previous run)")
        # self.sort_dropdown.setItemData(4, False, Qt.ItemIsEnabled)

        self.sort_dropdown.currentIndexChanged.connect(self.attempt_sort)
        
        self.tab1.layout.addLayout(self.checkbox_layout)
        self.tab1.layout.addLayout(self.input_layout)
        self.tab1.layout.addWidget(self.execute_button)
        self.tab1.layout.addWidget(self.progress_bar)
        self.tab1.layout.addWidget(self.sort_dropdown)
        self.tab1.layout.addWidget(QLabel("Feature/Importance:"))
        self.tab1.layout.addWidget(self.output_panel)
        self.finalcluster = False

        self.tab1.setLayout(self.tab1.layout)
        self.central_widget.setLayout(self.layout)

        #TAB-2 DESIGN LAYOUT



        self.operation_history = []
        self.redo_history = []
        self.operations_performed = 0
        self.current_channel = None
        self.current_image_array = None
        self.processed_image = None
        self.agg_operation = 'count'
        self.agg_channels = None
        self.previous_agg_operation = None
        self.previous_agg_channels = None

        # Define which aggregation operations are multi-channel
        self.multi_channel_ops = {'scorr', 'coloc', 'containment', 'relativeskew', 'angular_momentum', 'angular_entropy'}

        # Root directory input
        root_path_layout = QHBoxLayout()
        self.root_path_input = QLineEdit(QDir.homePath())
        self.root_path_input.returnPressed.connect(self.set_tree_root)

        self.change_root_button = QPushButton("Change Root")
        self.change_root_button.clicked.connect(self.browse_for_root)

        root_path_layout.addWidget(self.root_path_input)
        root_path_layout.addWidget(self.change_root_button)
        self.tab2.layout.addLayout(root_path_layout)

        # File system tree
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.homePath())
        self.model.setNameFilters(["*.tiff", "*.tif"])
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(QDir.homePath()))

        # Image display (left)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setScaledContents(True)

        # Processed image display (right)
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFrameShape(QFrame.StyledPanel)
        self.processed_image_label.setText("Processed Image") # Initial text
        self.processed_image_label.setScaledContents(True)
    
        # Channel slider
        self.channel_label = QLabel("Channel: ")
        self.channel_label.setAlignment(Qt.AlignCenter)
        self.channel_slider = QSlider(Qt.Vertical)
        self.channel_slider.valueChanged.connect(self.update_displayed_channel)
        self.channel_slider.setEnabled(False) # Disable initially

        # Create a vertical layout for each side of the split
        left_image_panel = QWidget()
        left_layout = QVBoxLayout(left_image_panel)
        left_layout.addWidget(self.image_label)
        
        right_image_panel = QWidget()
        right_layout = QVBoxLayout(right_image_panel)
        right_layout.addWidget(self.processed_image_label)

        # Create a vertical layout for each side of the split
        channel_panel = QWidget()
        channel_layout = QVBoxLayout(channel_panel)
        channel_layout.addWidget(self.channel_label)
        channel_layout.addWidget(self.channel_slider)

        # Create a horizontal splitter for the image panels
        self.image_splitter = QSplitter(Qt.Horizontal)
        self.image_splitter.addWidget(left_image_panel)
        self.image_splitter.addWidget(right_image_panel)
        self.image_splitter.addWidget(channel_panel)
        self.image_splitter.setStretchFactor(0, 1) # Image 1 expands
        self.image_splitter.setStretchFactor(1, 1) # Image 2 expands equally
        self.image_splitter.setStretchFactor(2, 0) # Slider is fixed size
        self.image_splitter.setSizes([300, 300, 10]) # Set initial proportions

        self.terminal = OperationHistory()


        self.reset_operations_button = QPushButton("Reset")
        self.reset_operations_button.clicked.connect(self.reset_operations)
        self.undo_operations_button = QPushButton("Undo")
        self.undo_operations_button.clicked.connect(self.undo_last_operation)

        # Container for top part of right panel
        top_right_container = QWidget()
        top_right_layout = QVBoxLayout(top_right_container)
        top_right_layout.addWidget(self.image_splitter)
        top_right_layout.setContentsMargins(0,0,0,0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel) # This will hold the vertical splitter
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 400]) # Revert to original horizontal split

        self.tab2.layout.addWidget(splitter) # Add the splitter to the tab's layout

        # Vertical splitter for the right panel
        right_v_splitter = QSplitter(Qt.Vertical)
        right_v_splitter.addWidget(top_right_container)
        right_v_splitter.addWidget(self.terminal)
        right_v_splitter.setSizes([400, 150]) # Revert to original vertical split

        # Bottom bar for info label and reset button
        bottom_bar_layout = QHBoxLayout()
        self.info_label = QLabel("No Image Loaded")
        bottom_bar_layout.addWidget(self.info_label)
        bottom_bar_layout.addStretch()
        bottom_bar_layout.addWidget(self.undo_operations_button)
        bottom_bar_layout.addWidget(self.reset_operations_button)

        right_layout.addWidget(right_v_splitter)
        right_layout.addLayout(bottom_bar_layout)

       # Connect the tree view's double-click signal to the image loading function
        self.tree.doubleClicked.connect(self.load_tiff_image)
        # Connect terminal's info update to the new label
        self.terminal.info_updated.connect(self.info_label.setText)

        # Borrow font size from tree view for other elements
        font = self.tree.font()
        fsize = font.pointSize()
        if fsize > 0:
            fs_str = f"{fsize}pt"
        elif font.pixelSize() > 0:
            fs_str = f"{font.pixelSize()}px"
        else:
            fs_str = "10pt"
            font.setPointSize(10)
        self.tabs.setFont(font)
        self.setStyleSheet(f"QLabel, QTextEdit, QLineEdit, QCheckBox, QComboBox, QProgressBar, QPushButton {{ font-size: {fs_str}; }} QTabBar::tab {{ font-size: {fs_str}; padding: 10px 30px; }}")

        # Menu bar
        self.create_menus()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)

    
    # --------------------------------------------------------------------------
    # Image Processing & Tree Navigation (Design Tab)
    # --------------------------------------------------------------------------
    def browse_for_root(self):
        directory = QFileDialog.getExistingDirectory(self, "Select New Root Directory",
                                                   self.root_path_input.text(),
                                                   QFileDialog.ShowDirsOnly)
        if directory:
            self.root_path_input.setText(directory)
            self.set_tree_root()

    def set_tree_root(self):
        root_path = self.root_path_input.text()
        if QDir(root_path).exists():
            self.model.setRootPath(root_path)
            self.tree.setRootIndex(self.model.index(root_path))
        else:
            print(f"Error: Root path '{root_path}' does not exist.")

    def load_tiff_image(self, index):
        self.processed_image = None
        self.operations_performed = 0
        file_path = self.model.filePath(index)
        self.tree.scrollTo(index)
        self.tree.setCurrentIndex(index)
        if file_path.lower().endswith(('.tiff', '.tif')):
            try:
                tif_image = tifffile.imread(file_path)
                self.current_image_array = np.array(tif_image)

                if self.current_image_array.ndim >= 3:
                    # Assuming channels are the first or last dimension
                    # You might need to adjust this based on your TIFF structure
                    if self.current_image_array.shape[0] > 1:
                        self.num_channels = self.current_image_array.shape[0]
                    elif self.current_image_array.shape[-1] > 1:
                        self.num_channels = self.current_image_array.shape[-1]
                    else:
                        self.num_channels = 1
                        self.current_image_array = np.expand_dims(self.current_image_array, axis=0) # Add a channel dimension

                    self.channel_slider.setMinimum(0)
                    self.channel_slider.setMaximum(self.num_channels - 1)
                    self.channel_slider.setEnabled(True)
                    if self.current_channel is None:
                        self.channel_slider.setValue(0)
                        self.update_displayed_channel(0) # Display the first channel
                    else:
                        if self.num_channels>self.current_channel>=0:#if channel does not exist for new image
                            self.channel_slider.setValue(self.current_channel)
                            self.update_displayed_channel(self.current_channel)
                        else:
                            self.channel_slider.setValue(0)
                            self.update_displayed_channel(0) # Display the first channel
                    self.terminal.add_operation('Image Set: ' + os.path.basename(file_path))
                    self.terminal.update_info(f"Array Info: Shape={self.current_image_array.shape}, Dtype={self.current_image_array.dtype}")

                elif self.current_image_array.ndim == 2:
                    self.current_image_array = np.expand_dims(self.current_image_array, axis=0) # Treat as single channel
                    self.num_channels = 1
                    self.channel_slider.setEnabled(False)
                    self.update_displayed_channel(0)
                    self.terminal.add_operation('Image Set: ' + os.path.basename(file_path))
                    self.terminal.update_info(f"Array Info: Shape={self.current_image_array.shape}, Dtype={self.current_image_array.dtype}")
                    
                else:
                    self.terminal.add_operation("Not a suitable image format for channel viewing.")
                    self.channel_slider.setEnabled(False)
                    self.terminal.update_info("No Image Loaded")
                    self.current_image_array = None
                    self.num_channels = 0
                if self.current_image_array is not None:
                    self.preprocessing_menu.setEnabled(True)
                    self.quantify_menu.setEnabled(True)
                    self.export_to_fcs.setEnabled(True)
                    self.export_to_csv.setEnabled(True)

            except ImportError:
                self.image_label.setText("Error: Required library not found.")
            except Exception as e:
                self.image_label.setText(f"Error loading TIFF file: {e}")
                self.channel_slider.setEnabled(False)
                self.current_image_array = None
                self.num_channels = 0
        else:
            self.image_label.clear()
            self.channel_slider.setEnabled(False)
            self.current_image_array = None
            self.num_channels = 0

    def update_displayed_channel(self, channel_index):
        self.operations_performed = 0
        if self.current_image_array is not None and 0 <= channel_index < self.num_channels:
            self.terminal.add_operation(f"Channel Set to:  {channel_index+1}")
            self.channel_label.setText(f"Channel: {channel_index + 1}/{self.num_channels}")
            self.current_channel = channel_index
            self.processed_image = self.current_image_array[self.current_channel]

            # Normalize and convert to 8-bit grayscale for display
            normalized_array = self.norm(self.current_image_array[channel_index])
            height, width = normalized_array.shape
            self.current_q_image = QImage(normalized_array.data, width, height, width, QImage.Format_Grayscale8)

            self.update_left_image_label()
            self.process_image()

    def update_left_image_label(self):
        if self.current_q_image is not None:
            pixmap = QPixmap.fromImage(self.current_q_image)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()

    def reset_operations(self):
        self.operation_history = []
        self.redo_history = []
        self.operations_performed = 0
        if self.current_image_array is not None and self.current_channel is not None:
            self.processed_image = self.current_image_array[self.current_channel].copy()
        self.terminal.add_operation('Reset Preprocessing Operations')
        self.process_image()

    def norm(self,array,eightbit=True):
        array -= np.min(array)
        max_val = np.max(array)
        if max_val > 0:
            array /= max_val
        array *= 255
        if eightbit:
            array = np.round(array).astype('uint8')
        return array

    def enable_aggregation(self,action):
        # Store the previous state in case the user cancels a dialog
        self.previous_agg_operation = self.agg_operation
        self.previous_agg_channels = self.agg_channels

        if action == self.count_action:
            self.enable_count()
        elif action == self.mean_action:
            self.enable_mean()
        elif action == self.area_action:
            self.enable_area()
        elif action == self.solidity_action:
            self.enable_solidity()
        elif action == self.scorr_action:
            self.open_multi_channel_dialog('scorr', ['Mask (Optional)', 'Channel 1', 'Channel 2'], disable_snr_checks=True)
        elif action == self.coloc_action:
            self.open_multi_channel_dialog('coloc', ['Signal', 'Mask'])
        elif action == self.containment_action:
            self.open_multi_channel_dialog('containment', ['Signal', 'Container', 'Global Mask (Optional)'])
        elif action == self.relativeskew_action:
            self.open_multi_channel_dialog('relativeskew', ['Signal', 'Reference', 'Global Mask (Optional)'])
        elif action == self.angular_momentum_action:
            self.open_multi_channel_dialog('angular_momentum', ['Signal', 'Reference', 'Global Mask (Optional)'])
        elif action == self.angular_entropy_action:
            self.open_multi_channel_dialog('angular_entropy', ['Signal', 'Reference', 'Global Mask (Optional)'])

    def enable_area(self):
        self.agg_operation = 'area'
        self.terminal.add_operation('Feature set to: Area')
        self.process_image()

    def enable_mean(self):
        self.agg_operation = 'mean'
        self.terminal.add_operation('Feature set to: Mean')
        self.process_image()

    def enable_count(self):
        self.agg_operation = 'count'
        self.terminal.add_operation('Feature set to: Count')
        self.process_image()

    def revert_to_previous_aggregation(self):
        """Reverts the aggregation operation to the previously selected one."""
        self.agg_operation = self.previous_agg_operation
        self.agg_channels = self.previous_agg_channels

        # Find and re-check the action corresponding to the previous operation
        if self.agg_operation:
            previous_action = self.findChild(QAction, f"{self.agg_operation}_action")
            if previous_action:
                previous_action.setChecked(True)
        else: # If there was no previous operation, default to count
            self.count_action.setChecked(True)
            self.enable_count()

    
    # --------------------------------------------------------------------------
    # Quantification & Feature Aggregation
    # --------------------------------------------------------------------------
    def do_aggregation(self):
        uniq = np.unique(self.processed_image)
        luniq = len(uniq)
        if luniq>1:
            if self.agg_operation == 'area':
                if 0 in uniq:
                    area = self.get_area(self.processed_image)
                    self.terminal.add_operation(f"Area is: {area}")
            if self.agg_operation == 'mean':
                mean = self.get_mean(self.processed_image)
                self.terminal.add_operation(f"Mean is: {mean}")
            if self.agg_operation == 'count':
                count = self.get_count(self.processed_image)
                self.terminal.add_operation(f"Count is: {count}")
            if self.agg_operation == 'scorr':
                mask_channel = self.agg_channels.get('Mask (Optional)')
                mask_img = self.process_image_for_channel(mask_channel) if mask_channel is not None else None
                ch1_img = self.process_image_for_channel(self.agg_channels['Channel 1'])
                ch2_img = self.process_image_for_channel(self.agg_channels['Channel 2'])
                scorr = self.get_spatial_correlation(ch1_img, ch2_img, mask_img=mask_img)
                self.terminal.add_operation(f"Spatial Correlation is: {scorr:.4f}")
            if self.agg_operation == 'solidity':
                solidity = self.get_solidity(self.processed_image)
                self.terminal.add_operation(f"Solidity is: {solidity:.4f}")
            if self.agg_operation == 'coloc':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'])
                mask_img = self.process_image_for_channel(self.agg_channels['Mask'])
                coloc = self.get_coloc(signal_img, mask_img)
                self.terminal.add_operation(f"Colocalisation is: {coloc:.4f}")
            if self.agg_operation == 'containment':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'])
                container_img = self.process_image_for_channel(self.agg_channels['Container'])
                global_mask_channel = self.agg_channels.get('Global Mask (Optional)')
                global_mask = self.process_image_for_channel(global_mask_channel) if global_mask_channel is not None else None
                containment = self.get_containment(signal_img, container_img, global_mask=global_mask)
                self.terminal.add_operation(f"Containment is: {containment:.4f}")
            if self.agg_operation == 'relativeskew':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'])
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'])
                global_mask_channel = self.agg_channels.get('Global Mask (Optional)')
                global_mask = self.process_image_for_channel(global_mask_channel) if global_mask_channel is not None else None
                relskew = self.get_relativeskew(signal_img, ref_img, global_mask=global_mask)
                self.terminal.add_operation(f"Relative Skewness is: {relskew:.4f}")
            if self.agg_operation == 'angular_momentum':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'])
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'])
                snr_checks = {'Signal': self.agg_channels['snr_checks']['Signal'], 'Reference': self.agg_channels['snr_checks']['Reference']}
                global_mask_channel = self.agg_channels.get('Global Mask (Optional)')
                global_mask = self.process_image_for_channel(global_mask_channel) if global_mask_channel is not None else None
                ang_mom = self.get_angular_momentum(signal_img, ref_img, global_mask=global_mask, snr_checks=snr_checks)
                self.terminal.add_operation(f"Angular Momentum is: {ang_mom:.4f}")
            if self.agg_operation == 'angular_entropy':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'])
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'])
                snr_checks = {'Signal': self.agg_channels['snr_checks']['Signal'], 'Reference': self.agg_channels['snr_checks']['Reference']}
                global_mask_channel = self.agg_channels.get('Global Mask (Optional)')
                global_mask = self.process_image_for_channel(global_mask_channel) if global_mask_channel is not None else None
                ang_ent = self.get_angular_entropy(signal_img, ref_img, global_mask=global_mask, snr_checks=snr_checks)
                self.terminal.add_operation(f"Angular Entropy is: {ang_ent:.4f}")

    def do_aggregation_silent(self,image):
        score = np.nan
        
        # Prepare optional global mask if it exists
        global_mask_channel = self.agg_channels.get('Global Mask (Optional)') if self.agg_channels is not None else None
        global_mask = None
        if global_mask_channel is not None:
            global_mask = self.process_image_for_channel(global_mask_channel, source_image_array=image)

        # Multi-channel operations handle their own channel extraction and processing
        if self.agg_operation in self.multi_channel_ops:
            if self.agg_operation == 'scorr':
                # For batch processing, 'image' is the full multi-channel image
                mask_channel = self.agg_channels.get('Mask (Optional)')
                mask_img = self.process_image_for_channel(mask_channel, source_image_array=image) if mask_channel is not None else None
                ch1_img = self.process_image_for_channel(self.agg_channels['Channel 1'], source_image_array=image)
                ch2_img = self.process_image_for_channel(self.agg_channels['Channel 2'], source_image_array=image)
                score = self.get_spatial_correlation(ch1_img, ch2_img, mask_img=mask_img)
            elif self.agg_operation == 'coloc':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'], source_image_array=image)
                mask_img = self.process_image_for_channel(self.agg_channels['Mask'], source_image_array=image)
                score = self.get_coloc(signal_img, mask_img)
            elif self.agg_operation == 'containment':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'], source_image_array=image)
                container_img = self.process_image_for_channel(self.agg_channels['Container'], source_image_array=image)
                score = self.get_containment(signal_img, container_img, global_mask=global_mask)
            elif self.agg_operation == 'relativeskew':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'], source_image_array=image)
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'], source_image_array=image)
                snr_checks = {'Signal': self.agg_channels['snr_checks']['Signal'], 'Reference': self.agg_channels['snr_checks']['Reference']}
                score = self.get_relativeskew(signal_img, ref_img, global_mask=global_mask, snr_checks=snr_checks)
            elif self.agg_operation == 'angular_momentum':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'], source_image_array=image)
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'], source_image_array=image)
                snr_checks = {'Signal': self.agg_channels['snr_checks']['Signal'], 'Reference': self.agg_channels['snr_checks']['Reference']}
                score = self.get_angular_momentum(signal_img, ref_img, global_mask=global_mask, snr_checks=snr_checks)
            elif self.agg_operation == 'angular_entropy':
                signal_img = self.process_image_for_channel(self.agg_channels['Signal'], source_image_array=image)
                ref_img = self.process_image_for_channel(self.agg_channels['Reference'], source_image_array=image)
                snr_checks = {'Signal': self.agg_channels['snr_checks']['Signal'], 'Reference': self.agg_channels['snr_checks']['Reference']}
                score = self.get_angular_entropy(signal_img, ref_img, global_mask=global_mask, snr_checks=snr_checks)

        # Single-channel operations work on a pre-processed image
        elif self.agg_operation == 'solidity':
            score = self.get_solidity(image)
        # Single-channel operations work on a pre-processed image
        else:
            uniq = np.unique(image)
            luniq = len(uniq)
            if luniq > 1:
                if self.agg_operation == 'area':
                    if 0 in uniq:
                        score = self.get_area(image)
                elif self.agg_operation == 'mean':
                    score = self.get_mean(image)
                elif self.agg_operation == 'count':
                    score = self.get_count(image)
                else: #default to count
                    score = self.get_count(image)
            else:
                score = np.nan

        if np.isnan(score):
            return 0
        else:
            return score

    def process_image(self):
        self.perform_operations()
        height, width = self.processed_image.shape
        pimage = self.norm(self.processed_image).data
        self.processed_q_image = QImage(pimage, width, height, width, QImage.Format_Grayscale8)
        self.update_right_image_label()

    def update_right_image_label(self):
        if self.processed_q_image is not None:
            pixmap = QPixmap.fromImage(self.processed_q_image)
            self.processed_image_label.setPixmap(pixmap)
        else:
            self.processed_image_label.clear()

    def perform_operations(self):
        nops = len(self.operation_history)
        for i in range(self.operations_performed,nops):
            self.do_operation(i)
        self.operations_performed = nops
        if self.agg_operation is not None:
            self.do_aggregation()

    def do_operation(self,opindex):
        operation = self.operation_history[opindex]

        if operation[0]=='gauss':
            self.processed_image = self.gaussblur(self.processed_image, operation[1])
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'{self.get_operation_description(operation)} Channel: {self.current_channel+1}')
        elif operation[0]=='mask':
            self.processed_image = self.get_mask(self.processed_image.astype(float)).astype(float)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Mask Channel: {self.current_channel+1}')
        elif operation[0]=='label':
            self.processed_image = self.get_label(self.processed_image.astype(int)).astype(float)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Label Channel: {self.current_channel+1}')
        elif operation[0]=='segment':
            self.processed_image = self.get_segment(self.processed_image.astype(float)).astype(float)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Segment Channel: {self.current_channel+1}')
        elif operation[0]=='preset1':
            # preset1_preprocess returns the processed image and a threshold mask. We only need the image here.
            self.processed_image, _ = self.preset1_preprocess(self.processed_image)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Preset 1 Preprocess Channel: {self.current_channel+1}')
        elif operation[0] == 'crop':
            top, bottom, left, right = operation[1]
            self.processed_image = self.crop_image(self.processed_image, top, bottom, left, right)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Crop: T={top}, B={bottom}, L={left}, R={right} on Channel: {self.current_channel+1}')
        elif operation[0] == 'rescale':
            scale_x, scale_y, interpolation_method = operation[1]
            self.processed_image = self.rescale_image(self.processed_image, scale_x, scale_y, interpolation_method)
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f'Rescale: X={scale_x}, Y={scale_y} on Channel: {self.current_channel+1}')

    def crop_image(self, image, top, bottom, left, right):
        h, w = image.shape
        return image[top:h-bottom, left:w-right]

    def do_operation_silent(self,index,image):     
        operation = self.operation_history[index]

        if operation[0]=='gauss':
            return self.gaussblur(image, operation[1])
        elif operation[0]=='mask':
            return self.get_mask(image.astype(float), clopen=True).astype(float)
        elif operation[0]=='label':
            return self.get_label(image.astype(int)).astype(float)
        elif operation[0]=='segment':
            return self.get_segment(image.astype(float)).astype(float)
        elif operation[0]=='preset1':
            image, _ = self.preset1_preprocess(image)
            return image
        elif operation[0] == 'crop':
            top, bottom, left, right = operation[1]
            return self.crop_image(image, top, bottom, left, right)
        elif operation[0] == 'rescale':
            scale_x, scale_y, interpolation_method = operation[1]
            return self.rescale_image(image, scale_x, scale_y, interpolation_method)

        return image # Return original image if operation is not found

    def process_image_for_channel(self, channel_index, source_image_array=None):
        """Applies the current operation history to a specific channel."""
        if source_image_array is None:
            source_image_array = self.current_image_array

        image = source_image_array[channel_index].copy().astype(np.float32)
        for i in range(len(self.operation_history)):
            image = self.do_operation_silent(i, image)
        return image

    # def ofdm_historical(self, img):
    #     """Historical implementation of hardcoded OFDM preprocessing."""
    #     img = img[TOPCROP:img.shape[0]-BOTTOMCROP,LEFTCROP:img.shape[1]-RIGHTCROP]
    #     width = int(img.shape[1] * 1)
    #     height = int(img.shape[0] * .25)
    #     dsize = (width, height)
    #     imgdown = cv2.resize(img,dsize,interpolation=cv2.INTER_AREA)
    #     imgdown = cv2.GaussianBlur(imgdown, AKERNEL, sigmaX=SIG_X,sigmaY=SIG_Y)
    #     current_height, current_width = imgdown.shape
    #     new_height =  int(current_height/PAR)
    #     dsize = (current_width, new_height)
    #     imgup = cv2.resize(imgdown, dsize, interpolation=cv2.INTER_LANCZOS4)
    #     imgblur = cv2.GaussianBlur(imgup, KERNEL, sigmaX=SIG2, sigmaY=SIG2,borderType=cv2.BORDER_CONSTANT)
    #     hmask = imgup>NOISETHRESHOLD
    #     sumsig = np.sum(hmask)
    #     if sumsig>=SMALL:   
    #         th = threshold_otsu(imgblur[hmask])
    #         imgth = imgblur>=th
    #     else:
    #         imgth = np.zeros(imgup.shape,dtype=bool)
    #     imgth = binary_closing(binary_opening(imgth,footprint=FOOTPRINT),footprint=SQUARE)
    #     imgth = remove_small_objects(imgth,SMALL,connectivity=2)
    #     imgth = binary_fill_holes(imgth,structure=np.ones((3,3)))
    #     boundary_pixels = (np.sum(imgth[0, :]) + np.sum(imgth[-1, :]) +
    #                     np.sum(imgth[:, 0]) + np.sum(imgth[:, -1]) -
    #                     imgth[0, 0] - imgth[0, -1] - imgth[-1, 0] - imgth[-1, -1])
    #     boundary_fraction = boundary_pixels/(2*(current_width+new_height-2))
    #     if np.sum(imgth)>(imgth.shape[0]*imgth.shape[1])/4 and boundary_fraction>0.05:
    #         imgth *= False
    #     return imgup,imgth

    def open_gauss(self):
        dialog = GaussDialog(self)  # Pass self as parent
        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_values()
            self.add_new_operation(['gauss', params])
            self.process_image()
        else:
            print("Dialog cancelled.")

    def open_crop_dialog(self):
        dialog = CropDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            crop_values = dialog.get_values()
            self.add_new_operation(['crop', crop_values])
            self.process_image()

    def open_rescale_dialog(self):
        dialog = RescaleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            scale_x, scale_y, interpolation_method = dialog.get_values()
            self.add_new_operation(['rescale', (scale_x, scale_y, interpolation_method)])
            self.process_image()

    def rescale_image(self, image, scale_x, scale_y, interpolation_method):
        h, w = image.shape
        new_w = max(1, int(w * scale_x))
        new_h = max(1, int(h * scale_y))
        dsize = (new_w, new_h)
        if isinstance(interpolation_method, int):
            interp = interpolation_method
        else:
            inter_map = {'Nearest': cv2.INTER_NEAREST, 'Linear': cv2.INTER_LINEAR, 'Area': cv2.INTER_AREA, 'Cubic': cv2.INTER_CUBIC, 'Lanczos4': cv2.INTER_LANCZOS4}
            interp = inter_map.get(str(interpolation_method), cv2.INTER_LINEAR)
        return cv2.resize(image, dsize, interpolation=interp)

    def do_mask(self):
        self.add_new_operation(['mask'])
        self.process_image()

    def do_segment(self):
        self.add_new_operation(['segment'])
        self.process_image()

    def do_label(self):
        self.add_new_operation(['label'])
        self.process_image()

    # def do_preset1(self):
    #     self.operation_history = [['preset1']]
    #     self.redo_history = []
    #     self.operations_performed = 0
    #     if self.current_image_array is not None and self.current_channel is not None:
    #         self.processed_image = self.current_image_array[self.current_channel].copy()
    #     if not getattr(self, 'suppress_terminal_logging', False):
    #         self.terminal.add_operation("Applied Hardcoded Preset: OFDM")
    #         self.terminal.add_operation(self.get_pipeline_summary())
    #     self.process_image()

    
    # --------------------------------------------------------------------------
    # Preset Directory & File Management
    # --------------------------------------------------------------------------
    def get_presets_dir(self):
        """Retrieves the active presets directory path from session memory, config file, or defaults to user data dir / <app_dir>/Presets."""
        if getattr(self, 'session_presets_dir', None) and os.path.exists(self.session_presets_dir):
            return self.session_presets_dir

        config = load_flowfi_config()
        presets_dir = config.get('presets_dir')
        if not presets_dir or not os.path.exists(presets_dir):
            if getattr(sys, 'frozen', False):
                presets_dir = os.path.join(get_user_data_dir(), 'Presets')
            else:
                presets_dir = os.path.join(get_app_dir(), 'Presets')
        return presets_dir

    def get_default_ofdm_preset_data(self):
        """Returns the default OFDM Preset payload dictionary."""
        return {
            "name": "OFDM Preset",
            "description": "Standard OFDM flow cytometry preprocessing preset composed of standard crop, rescale, anisotropic gaussian blur, rescale, and isotropic gaussian blur steps",
            "preprocessing_steps": [
                {
                    "type": "crop",
                    "args": [int(TOPCROP), int(BOTTOMCROP), int(LEFTCROP), int(RIGHTCROP)]
                },
                {
                    "type": "rescale",
                    "args": [1.0, 0.25, "Area"]
                },
                {
                    "type": "gauss",
                    "args": {
                        "kernel": list(AKERNEL),
                        "sigmaX": float(SIG_X),
                        "sigmaY": float(SIG_Y)
                    }
                },
                {
                    "type": "rescale",
                    "args": [1.0, float(1.0 / PAR), "Lanczos4"]
                },
                {
                    "type": "gauss",
                    "args": {
                        "kernel": list(KERNEL),
                        "sigmaX": float(SIG2),
                        "sigmaY": float(SIG2)
                    }
                }
            ],
            "quantification": {
                "operation": "count",
                "channels": None
            }
        }

    def create_default_ofdm_preset(self, filepath):
        """Creates the default OFDM Preset JSON file with decomposed individual preprocessing steps and quantification configuration."""
        ofdm_data = self.get_default_ofdm_preset_data()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ofdm_data, f, indent=4)
        except Exception as e:
            print(f"Error creating default OFDM preset JSON: {e}")

    def ensure_presets_dir_exists(self, presets_dir=None):
        """Ensures the presets directory exists, creating the default OFDM Preset JSON if missing.
        If the directory is not accessible and cannot be created, prompts the user to select one."""
        if presets_dir is None:
            presets_dir = self.get_presets_dir()

        dir_accessible = False
        try:
            if not os.path.exists(presets_dir):
                os.makedirs(presets_dir, exist_ok=True)
            # Verify write accessibility
            test_file = os.path.join(presets_dir, '.writable_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            dir_accessible = True
        except Exception:
            dir_accessible = False

        if not dir_accessible:
            try:
                fallback_dir = os.path.join(get_user_data_dir(), 'Presets')
                os.makedirs(fallback_dir, exist_ok=True)
                presets_dir = fallback_dir
                dir_accessible = True
            except Exception:
                dir_accessible = False

        if not dir_accessible:
            app_inst = QApplication.instance()
            if app_inst:
                chosen_dir = QFileDialog.getExistingDirectory(self if hasattr(self, 'isVisible') else None, 
                                                             "Presets Folder Inaccessible - Select Presets Location", 
                                                             os.path.expanduser('~'))
                if chosen_dir and os.path.exists(chosen_dir):
                    presets_dir = chosen_dir
                    config = load_flowfi_config()
                    config['presets_dir'] = presets_dir
                    save_flowfi_config(config)
                    dir_accessible = True

        if dir_accessible and os.path.exists(presets_dir):
            # Copy bundled default presets if in frozen/PyInstaller mode and bundled presets exist
            bundled_presets = resource_path('Presets')
            if getattr(sys, 'frozen', False) and os.path.exists(bundled_presets) and os.path.abspath(bundled_presets) != os.path.abspath(presets_dir):
                for fname in os.listdir(bundled_presets):
                    if fname.lower().endswith('.json'):
                        src_path = os.path.join(bundled_presets, fname)
                        dst_path = os.path.join(presets_dir, fname)
                        if not os.path.exists(dst_path):
                            import shutil
                            shutil.copy2(src_path, dst_path)

            # Check if any existing JSON file in presets_dir is already an OFDM preset
            ofdm_exists = False
            for fname in os.listdir(presets_dir):
                if fname.lower().endswith('.json'):
                    fpath = os.path.join(presets_dir, fname)
                    if fname.lower() in ('ofdm preset.json', 'ofdm_preset.json', 'ofdm.json'):
                        ofdm_exists = True
                        break
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            d = json.load(f)
                            if isinstance(d, dict) and d.get('name', '').lower().strip() in ('ofdm preset', 'ofdm'):
                                ofdm_exists = True
                                break
                    except Exception:
                        pass

            if not ofdm_exists:
                ofdm_json_path = os.path.join(presets_dir, 'OFDM_Preset.json')
                self.create_default_ofdm_preset(ofdm_json_path)

        return presets_dir

    def refresh_presets_menu(self):
        """Dynamically populates the Presets submenu with available preset JSON files or in-memory OFDM fallback."""
        if not hasattr(self, 'presets_submenu'):
            return
        
        self.presets_submenu.clear()

        save_action = QAction('&Save New Parameter Preset...', self)
        save_action.triggered.connect(self.save_new_preset)

        load_file_action = QAction('&Load Parameter Preset File...', self)
        load_file_action.triggered.connect(self.load_preset_from_file)

        config_dir_action = QAction('&Configure Presets Location...', self)
        config_dir_action.triggered.connect(self.configure_presets_location)

        self.presets_submenu.addAction(save_action)
        self.presets_submenu.addAction(load_file_action)
        self.presets_submenu.addAction(config_dir_action)

        presets_dir = self.ensure_presets_dir_exists()
        json_files = []
        dir_accessible = False

        if presets_dir and os.path.exists(presets_dir):
            try:
                for fname in os.listdir(presets_dir):
                    if fname.lower().endswith('.json'):
                        json_files.append(fname)
                dir_accessible = True
            except Exception:
                dir_accessible = False
        
        json_files.sort()

        if dir_accessible and json_files:
            self.presets_submenu.addSeparator()
            for fname in json_files:
                file_path = os.path.join(presets_dir, fname)
                preset_name = os.path.splitext(fname)[0]
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'name' in data and data['name']:
                            preset_name = data['name']
                except Exception:
                    pass
                
                action = QAction(f'&Preset - {preset_name}', self)
                action.triggered.connect(lambda checked, p=file_path: self.load_preset(p))
                self.presets_submenu.addAction(action)
        else:
            # If Presets directory is inaccessible or empty, populate with in-memory OFDM Preset
            self.presets_submenu.addSeparator()
            ofdm_data = self.get_default_ofdm_preset_data()
            action = QAction(f"&Preset - {ofdm_data['name']}", self)
            action.triggered.connect(lambda checked, d=ofdm_data: self.load_preset(d))
            self.presets_submenu.addAction(action)

    def save_new_preset(self):
        """Prompts user for a preset name and saves preprocessing steps and quantification settings to JSON."""
        name, ok = QInputDialog.getText(self, "Save New Parameter Preset", "Enter preset name:")
        if not ok or not name.strip():
            return
        
        preset_name = name.strip()
        safe_filename = re.sub(r'[^\w\s-]', '', preset_name).strip()
        if not safe_filename:
            safe_filename = "preset"
        safe_filename += ".json"

        steps_data = []
        for op in self.operation_history:
            op_type = op[0]
            if op_type in ('gauss',):
                steps_data.append({"type": op_type, "args": float(op[1])})
            elif op_type in ('crop',):
                top, bottom, left, right = op[1]
                steps_data.append({"type": op_type, "args": [int(top), int(bottom), int(left), int(right)]})
            elif op_type in ('rescale',):
                scale_x, scale_y, interp = op[1]
                steps_data.append({"type": op_type, "args": [float(scale_x), float(scale_y), int(interp)]})
            else:
                steps_data.append({"type": op_type})

        quant_data = {
            "operation": self.agg_operation,
            "channels": self.agg_channels
        }

        preset_payload = {
            "name": preset_name,
            "preprocessing_steps": steps_data,
            "quantification": quant_data
        }

        presets_dir = self.ensure_presets_dir_exists()
        filepath = os.path.join(presets_dir, safe_filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(preset_payload, f, indent=4)
            
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f"Saved Preset '{preset_name}' to {filepath}")
            
            self.refresh_presets_menu()
            if self.isVisible():
                QMessageBox.information(self, "Preset Saved", f"Preset '{preset_name}' successfully saved.")
        except Exception as e:
            if self.isVisible():
                QMessageBox.critical(self, "Error Saving Preset", f"Could not save preset file:\n{e}")

    def load_preset_from_file(self):
        """Opens a file dialog to select and load a preset JSON file."""
        presets_dir = self.ensure_presets_dir_exists()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Parameter Preset", presets_dir, "JSON Files (*.json)")
        if filepath:
            self.load_preset(filepath)

    def update_quantification_menu_checks(self):
        """Updates the checked state of actions in the Quantify menu based on self.agg_operation."""
        op_action_map = {
            'count': getattr(self, 'count_action', None),
            'mean': getattr(self, 'mean_action', None),
            'area': getattr(self, 'area_action', None),
            'solidity': getattr(self, 'solidity_action', None),
            'coloc': getattr(self, 'coloc_action', None),
            'containment': getattr(self, 'containment_action', None),
            'relativeskew': getattr(self, 'relativeskew_action', None),
            'angular_momentum': getattr(self, 'angular_momentum_action', None),
            'angular_entropy': getattr(self, 'angular_entropy_action', None),
            'scorr': getattr(self, 'scorr_action', None)
        }
        target_action = op_action_map.get(self.agg_operation)
        if target_action:
            target_action.setChecked(True)

    def get_pipeline_summary(self):
        """Returns a string summary of the current pipeline: [method] -> [method] => quantification."""
        ops_summary = " -> ".join([f"[{op[0]}]" for op in getattr(self, 'operation_history', [])])
        quant_summary = getattr(self, 'agg_operation', None) or "count"
        return f"{ops_summary} => {quant_summary}" if ops_summary else f"=> {quant_summary}"

    def load_preset(self, target):
        """Parses a preset JSON file path or dictionary object and applies its preprocessing steps and quantification settings.
        Fails gracefully and rejects malformed presets with a warning to the user."""
        data = None
        preset_name = "Preset"
        source_desc = target if isinstance(target, str) else "in-memory payload"
        
        if isinstance(target, str):
            if not os.path.exists(target):
                err_msg = f"Preset file does not exist:\n{target}"
                if not getattr(self, 'suppress_terminal_logging', False):
                    self.terminal.add_operation(f"Error: {err_msg}")
                if self.isVisible():
                    QMessageBox.warning(self, "File Not Found", err_msg)
                return False
            try:
                with open(target, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                preset_name = os.path.splitext(os.path.basename(target))[0]
            except Exception as e:
                err_msg = f"Failed to read or parse JSON file '{target}':\n{e}"
                if not getattr(self, 'suppress_terminal_logging', False):
                    self.terminal.add_operation(f"Rejected Preset: {err_msg}")
                if self.isVisible():
                    QMessageBox.warning(self, "Invalid Preset File", f"Preset rejected due to JSON parsing error:\n\n{e}")
                return False
        elif isinstance(target, dict):
            data = target
        else:
            if self.isVisible():
                QMessageBox.warning(self, "Invalid Preset Target", "Target preset must be a valid file path or dictionary.")
            return False

        # Validate top-level data structure
        if not isinstance(data, dict):
            err_msg = f"Invalid preset format in '{source_desc}': Root object must be a JSON dictionary."
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f"Rejected Preset: {err_msg}")
            if self.isVisible():
                QMessageBox.warning(self, "Invalid Preset Format", err_msg)
            return False

        if "name" in data and isinstance(data["name"], str) and data["name"].strip():
            preset_name = data["name"].strip()

        steps_input = data.get("preprocessing_steps", [])
        if not isinstance(steps_input, list):
            err_msg = f"Invalid preset format in '{preset_name}': 'preprocessing_steps' must be a list."
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f"Rejected Preset: {err_msg}")
            if self.isVisible():
                QMessageBox.warning(self, "Invalid Preset Format", err_msg)
            return False

        # Parse and validate individual steps into new_ops
        new_ops = []
        try:
            for idx, step in enumerate(steps_input):
                if isinstance(step, dict):
                    op_type = step.get("type")
                    args = step.get("args")
                    if not op_type:
                        raise ValueError(f"Step #{idx+1} is missing a 'type' field.")

                    if op_type == 'gauss':
                        if isinstance(args, (dict, list, tuple)):
                            new_ops.append(['gauss', args])
                        elif args is not None:
                            try:
                                new_ops.append(['gauss', float(args)])
                            except (ValueError, TypeError):
                                new_ops.append(['gauss', args])
                        else:
                            raise ValueError(f"Gauss step #{idx+1} is missing arguments.")

                    elif op_type == 'crop':
                        if not isinstance(args, (list, tuple)) or len(args) != 4:
                            raise ValueError(f"Crop step #{idx+1} args must be a 4-element list/tuple [top, bottom, left, right].")
                        top, bottom, left, right = args
                        new_ops.append(['crop', (int(top), int(bottom), int(left), int(right))])

                    elif op_type == 'rescale':
                        if not isinstance(args, (list, tuple)) or len(args) < 3:
                            raise ValueError(f"Rescale step #{idx+1} args must be [scale_x, scale_y, interpolation].")
                        sx, sy, interp = args[0], args[1], args[2]
                        try:
                            interp_val = int(interp)
                        except (ValueError, TypeError):
                            interp_val = str(interp)
                        new_ops.append(['rescale', (float(sx), float(sy), interp_val)])

                    elif op_type in ('mask', 'label', 'segment'):
                        new_ops.append([op_type])
                    else:
                        raise ValueError(f"Unknown or unsupported operation type '{op_type}' in step #{idx+1}.")

                elif isinstance(step, list) and len(step) > 0:
                    op_type = step[0]
                    if op_type == 'gauss':
                        if len(step) < 2:
                            raise ValueError(f"Gauss step #{idx+1} missing argument.")
                        arg = step[1]
                        if isinstance(arg, (dict, list, tuple)):
                            new_ops.append(['gauss', arg])
                        else:
                            try:
                                new_ops.append(['gauss', float(arg)])
                            except (ValueError, TypeError):
                                new_ops.append(['gauss', arg])

                    elif op_type == 'crop':
                        if len(step) < 2 or not isinstance(step[1], (list, tuple)) or len(step[1]) != 4:
                            raise ValueError(f"Crop step #{idx+1} requires a 4-tuple bounds.")
                        new_ops.append(['crop', tuple(step[1])])

                    elif op_type == 'rescale':
                        if len(step) < 2 or not isinstance(step[1], (list, tuple)) or len(step[1]) < 3:
                            raise ValueError(f"Rescale step #{idx+1} requires (scale_x, scale_y, interp).")
                        sx, sy, interp = step[1][0], step[1][1], step[1][2]
                        try:
                            interp_val = int(interp)
                        except (ValueError, TypeError):
                            interp_val = str(interp)
                        new_ops.append(['rescale', (float(sx), float(sy), interp_val)])

                    elif op_type in ('mask', 'label', 'segment'):
                        new_ops.append([op_type])
                    else:
                        raise ValueError(f"Unknown operation type '{op_type}' in step #{idx+1}.")
                else:
                    raise ValueError(f"Step #{idx+1} is improperly formatted.")

        except Exception as parse_err:
            err_msg = f"Preset '{preset_name}' rejected due to formatting error:\n{parse_err}"
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f"Rejected Preset: {err_msg}")
            if self.isVisible():
                QMessageBox.warning(self, "Invalid Preset Format", err_msg)
            return False

        # If validation succeeds, apply the preset to the state
        self.operation_history = new_ops
        self.redo_history = []

        quant_data = data.get("quantification", {})
        if isinstance(quant_data, dict):
            self.agg_operation = quant_data.get("operation", "count")
            self.agg_channels = quant_data.get("channels", None)
            self.update_quantification_menu_checks()

        if not getattr(self, 'suppress_terminal_logging', False):
            self.terminal.add_operation(f"Loaded Preset: {preset_name}")
            self.terminal.add_operation(self.get_pipeline_summary())

        self.suppress_terminal_logging = True
        try:
            self.operations_performed = 0
            if self.current_image_array is not None and self.current_channel is not None:
                self.processed_image = self.current_image_array[self.current_channel].copy()
            self.process_image()
        finally:
            self.suppress_terminal_logging = False

        if self.isVisible():
            QMessageBox.information(self, "Preset Loaded", f"Loaded preset '{preset_name}' successfully.")
        return True

    def configure_presets_location(self):
        """Allows user to select a new Presets directory and saves the choice to flowfi_config.json.
        If config saving fails, the chosen directory is still applied for the current session."""
        current_dir = self.ensure_presets_dir_exists()
        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Presets Directory", current_dir)
        if chosen_dir and os.path.exists(chosen_dir):
            self.session_presets_dir = chosen_dir
            config = load_flowfi_config()
            config['presets_dir'] = chosen_dir
            saved = save_flowfi_config(config)
            
            self.refresh_presets_menu()
            if not getattr(self, 'suppress_terminal_logging', False):
                self.terminal.add_operation(f"Presets location updated to: {chosen_dir}")

            if self.isVisible():
                if saved:
                    QMessageBox.information(self, "Presets Location Updated", f"Presets directory updated to:\n{chosen_dir}")
                else:
                    QMessageBox.warning(self, "Config File Inaccessible", f"Presets directory applied for current session, but preferences could not be saved to config file. Presets in:\n\n{chosen_dir}")

    def add_new_operation(self, op):
        self.operation_history.append(op)
        self.redo_history = []  # Clear redo history on new action

    def get_operation_description(self, operation):
        op_type = operation[0]
        if op_type == 'gauss':
            arg = operation[1]
            if isinstance(arg, dict):
                k = arg.get('kernel', (5, 3))
                sx = arg.get('sigmaX', 2.5)
                sy = arg.get('sigmaY', 0.8)
                return f"Gaussian Blur: K={k}, σX={sx}, σY={sy}"
            elif isinstance(arg, (list, tuple)):
                return f"Gaussian Blur: {arg}"
            try:
                return f"Gaussian Blur: {np.round(float(arg), 2)}"
            except Exception:
                return f"Gaussian Blur: {arg}"
        elif op_type == 'mask':
            return "Mask"
        elif op_type == 'label':
            return "Label"
        elif op_type == 'segment':
            return "Segment"
        elif op_type == 'preset1':
            return "Preset 1 Preprocess"
        elif op_type == 'crop':
            top, bottom, left, right = operation[1]
            return f"Crop: T={top}, B={bottom}, L={left}, R={right}"
        elif op_type == 'rescale':
            scale_x, scale_y, interpolation_method = operation[1]
            return f"Rescale: X={scale_x}, Y={scale_y}"
        return str(op_type)

    def undo_last_operation(self):
        """Removes the last operation from the history, adds it to redo history, and re-processes the image."""
        if not self.operation_history:
            self.terminal.add_operation("No operations to undo.")
            return

        last_op = self.operation_history.pop()
        self.redo_history.append(last_op)
        op_desc = self.get_operation_description(last_op)
        self.terminal.add_operation(f"Undo: {op_desc}")

        # Reset and re-process from the original image for the current channel
        self.suppress_terminal_logging = True
        try:
            self.operations_performed = 0
            if self.current_image_array is not None and self.current_channel is not None:
                self.processed_image = self.current_image_array[self.current_channel].copy()
            self.process_image()
        finally:
            self.suppress_terminal_logging = False

    def redo_operation(self):
        """Pops the last undone operation from redo history and re-applies it."""
        if not self.redo_history:
            self.terminal.add_operation("No operations to redo.")
            return

        op = self.redo_history.pop()
        self.operation_history.append(op)
        op_desc = self.get_operation_description(op)
        self.terminal.add_operation(f"Redo: {op_desc}")

        self.suppress_terminal_logging = True
        try:
            self.process_image()
        finally:
            self.suppress_terminal_logging = False

    def export_terminal_history(self):
        """Saves all text in the terminal to a txt file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Terminal History",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                text = self.terminal.history_text_edit.toPlainText()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.terminal.add_operation(f"Exported terminal history to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save terminal history: {e}")

    
    # --------------------------------------------------------------------------
    # FCS & CSV Merging & Output I/O
    # --------------------------------------------------------------------------
    def do_concat_csvs(self):
        """Concatenates multiple CSV files and alerts on dimensionality mismatch."""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files to Concatenate",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if not file_paths:
            return
            
        if len(file_paths) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least two CSV files to concatenate.")
            return

        dfs = []
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                dfs.append((path, df))
            except Exception as e:
                QMessageBox.critical(self, "Error Reading File", f"Failed to read file:\n{os.path.basename(path)}\n\nError: {e}")
                return

        # Check if sample_id is available in all loaded files
        all_have_sample_id = all('sample_id' in df.columns for _, df in dfs)

        if all_have_sample_id:
            first_path, first_df = dfs[0]
            first_sample_ids = list(first_df['sample_id'])
            
            # Identify any file with mismatched sample IDs
            mismatch_file = None
            for path, df in dfs[1:]:
                if list(df['sample_id']) != first_sample_ids:
                    mismatch_file = os.path.basename(path)
                    break

            if mismatch_file:
                QMessageBox.critical(
                    self,
                    "Sample ID Mismatch",
                    f"The sample IDs in '{mismatch_file}' do not match '{os.path.basename(first_path)}'. Aborting concatenation."
                )
                return

            # Sample IDs match across all files: perform column merge on sample_id
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Concatenated CSV",
                "",
                "CSV Files (*.csv);;All Files (*)",
                options=options
            )
            if not save_path:
                return

            try:
                concatenated_df = dfs[0][1]
                for _, df in dfs[1:]:
                    concatenated_df = pd.merge(concatenated_df, df, on='sample_id', how='outer')
                concatenated_df.to_csv(save_path, index=False)
                
                param_cols = [c for c in concatenated_df.columns if c != 'sample_id']
                num_params = len(param_cols)
                num_files = len(file_paths)

                self.terminal.add_operation(
                    f"Merged {num_files} CSV files ({num_params} parameter column(s)) on sample_id into: {os.path.normpath(save_path)}"
                )
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully merged {num_files} CSV files containing {num_params} parameter column(s) on sample_id.\n\n"
                    f"Saved to: {os.path.basename(save_path)}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", f"Failed to save merged CSV:\n{e}")
            return

        # Fallback: Row-wise concatenation as before
        first_path, first_df = dfs[0]
        first_cols = list(first_df.columns)
        first_num_cols = len(first_cols)

        for path, df in dfs[1:]:
            # Check empty DataFrame or 0 columns
            if df.empty or len(df.columns) == 0:
                QMessageBox.warning(self, "Empty CSV File", f"The file '{os.path.basename(path)}' is empty or has no columns.")
                return

            if df.shape[1] != first_df.shape[1]:
                QMessageBox.critical(
                    self,
                    "Dimensionality Mismatch",
                    f"The selected CSV files are not a match (dimensionality mismatch).\n\n"
                    f"'{os.path.basename(first_path)}' has {first_num_cols} columns.\n"
                    f"'{os.path.basename(path)}' has {df.shape[1]} columns."
                )
                return

            if list(df.columns) != first_cols:
                QMessageBox.critical(
                    self,
                    "Dimensionality Mismatch",
                    f"The selected CSV files are not a match (column names or order mismatch).\n\n"
                    f"'{os.path.basename(first_path)}' and '{os.path.basename(path)}' have different column headers."
                )
                return

        # Prompt for output file
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Concatenated CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if not save_path:
            return

        try:
            concatenated_df = pd.concat([df for _, df in dfs], ignore_index=True)
            concatenated_df.to_csv(save_path, index=False)

            param_cols = [c for c in concatenated_df.columns if str(c).lower() not in ['sample_id', 'sample_ids', 'sample', 'id']]
            num_params = len(param_cols)
            num_files = len(file_paths)

            self.terminal.add_operation(
                f"Concatenated {num_files} CSV files ({num_params} parameter column(s)) into: {os.path.normpath(save_path)}"
            )
            QMessageBox.information(
                self,
                "Success",
                f"Successfully concatenated {num_files} CSV files containing {num_params} parameter column(s).\n\n"
                f"Saved to: {os.path.basename(save_path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Saving File", f"Failed to save concatenated CSV:\n{e}")

    def do_merge_csv_to_fcs(self):
        import flowio
        import flowkit as fk
        """
        Merges an N x P parameter matrix from a CSV file into an FCS file with N events.
        Strips sample ID columns, validates event vector lengths, and aborts if channel name conflicts exist.
        """
        options = QFileDialog.Options()
        fcs_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Base FCS Template File",
            "",
            "FCS Files (*.fcs);;All Files (*)",
            options=options
        )
        if not fcs_file:
            return

        csv_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Parameter CSV File to Merge into FCS",
            os.path.dirname(fcs_file),
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if not csv_file:
            return

        # Load FCS template data
        try:
            fcdata, metadata = self.load_fcs(fcs_file)
            num_fcs_channels = fcdata.channel_count
            fcs_events_flat = fcdata.events
            n_fcs_events = len(fcs_events_flat) // num_fcs_channels
            fcs_channel_names = [fcdata.channels[k]['PnN'] for k in fcdata.channels.keys()]
        except Exception as e:
            QMessageBox.critical(self, "Error Loading FCS File", f"Failed to load FCS template file:\n{e}")
            return

        # Read CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            QMessageBox.critical(self, "Error Reading CSV File", f"Failed to read CSV file:\n{os.path.basename(csv_file)}\n\nError: {e}")
            return

        # Remove sample ID / identifier columns
        id_cols_to_drop = [col for col in df.columns if str(col).lower() in ['sample_id', 'sample_ids', 'sample', 'id', 'filename', 'index', 'unnamed: 0']]
        parameter_df = df.drop(columns=id_cols_to_drop)

        if parameter_df.empty or parameter_df.shape[1] == 0:
            QMessageBox.critical(
                self,
                "No Parameter Columns",
                "No parameter data columns were found in the selected CSV file after removing sample IDs."
            )
            return

        # Check dimension matching against FCS event count
        n_csv_rows = parameter_df.shape[0]
        if n_csv_rows != n_fcs_events:
            QMessageBox.critical(
                self,
                "Dimensionality Mismatch",
                f"The parameter data has {n_csv_rows} rows, but the FCS template file has {n_fcs_events} events.\n\n"
                f"Cannot merge parameter channels due to event count mismatch."
            )
            return

        # Check for channel name conflicts with existing FCS channels
        new_pnames = [str(col) for col in parameter_df.columns]
        conflicting_channels = [p for p in new_pnames if p in fcs_channel_names]
        if conflicting_channels:
            QMessageBox.critical(
                self,
                "Channel Name Conflict",
                f"The following parameter channel name(s) already exist in the FCS file:\n"
                f"{', '.join(conflicting_channels)}\n\n"
                f"Merging aborted to prevent overwriting existing channels."
            )
            return

        # Prompt for Log Transformation (10^x)
        reply = QMessageBox.question(
            self,
            'Log Transformation',
            "Apply 10^x transformation to the new parameter(s) for loglog visualisation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        transform_log = (reply == QMessageBox.Yes)

        # Convert parameter_df to numeric values, filling missing/NaN values with 0
        vals = parameter_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).values
        if transform_log:
            vals = 10 ** vals

        # Prompt for output FCS file save location
        base, ext = os.path.splitext(fcs_file)
        default_out_fcs = base + "_merged" + ext
        out_fcs, _ = QFileDialog.getSaveFileName(
            self,
            "Save Merged FCS File",
            default_out_fcs,
            "FCS Files (*.fcs);;All Files (*)",
            options=options
        )
        if not out_fcs:
            return

        try:
            self.add_params(fcdata, out_fcs, metadata, vals, pnames=new_pnames)
            self.terminal.add_operation(f"Merged {len(new_pnames)} parameter(s) ({', '.join(new_pnames)}) into FCS file: {os.path.normpath(out_fcs)}")
            QMessageBox.information(
                self,
                "Success",
                f"Successfully merged {len(new_pnames)} parameter(s) into FCS file:\n{os.path.basename(out_fcs)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Saving FCS File", f"Failed to save merged FCS file:\n{e}")

    def get_indexed_parameter_filename(self, folder_path, method_name):
        """
        Generates default parameter filename following convention: [method]_parameter.csv
        or [method]_parameter_[index].csv / [method]_parameter[index].csv if file already exists.
        e.g., if solidity_parameter.csv exists in folder_path, proposes solidity_parameter2.csv.
        """
        method_name = method_name if method_name else 'new'
        base_name = f"{method_name}_parameter.csv"
        target_path = os.path.join(folder_path, base_name)
        if not os.path.exists(target_path):
            return base_name
        
        idx = 2
        while True:
            cand1 = f"{method_name}_parameter{idx}.csv"
            cand2 = f"{method_name}_parameter_{idx}.csv"
            if not os.path.exists(os.path.join(folder_path, cand1)) and not os.path.exists(os.path.join(folder_path, cand2)):
                return cand1
            idx += 1

    def enable_solidity(self):
        self.agg_operation = 'solidity'
        self.terminal.add_operation('Feature set to: Solidity')
        self.process_image()

    def open_multi_channel_dialog(self, op_name, channel_roles, disable_snr_checks=False):
        if self.current_image_array is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            # Find the action and uncheck it
            action = self.findChild(QAction, f"{op_name}_action")
            if action:
                action.setChecked(False)
            return

        dialog = MultiChannelDialog(channel_roles, self.num_channels, self, disable_snr_checks=disable_snr_checks)
        if dialog.exec_() == QDialog.Accepted:
            self.agg_channels = dialog.get_channels()
            self.agg_operation = op_name
            self.terminal.add_operation(f'Feature set to: {op_name.capitalize()} with channels {self.agg_channels}')
            self.process_image()
        else: # Dialog was cancelled
            self.terminal.add_operation(f"{op_name.capitalize()} selection cancelled.")
            # Uncheck the action that was just clicked
            current_action = self.findChild(QAction, f"{op_name}_action")
            if current_action:
                current_action.setChecked(False)
            
            # Revert to the previous state
            self.revert_to_previous_aggregation()

    def parse_channel_string(self, text):
        channels = set()
        parts = text.split(',')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    subparts = part.split('-')
                    if len(subparts) != 2:
                         raise ValueError(f"Invalid range format: '{part}'")
                    start, end = map(int, subparts)
                    if start > end:
                        start, end = end, start
                    # Adjust for 1-based indexing
                    channels.update(range(start - 1, end))
                except ValueError:
                    raise ValueError(f"Invalid range format: '{part}'")
            else:
                try:
                    # Adjust for 1-based indexing
                    channels.add(int(part) - 1)
                except ValueError:
                    raise ValueError(f"Invalid number format: '{part}'")
        
        # Filter valid channels
        valid_channels = sorted([c for c in channels if 0 <= c < self.num_channels])
        
        if not valid_channels:
            raise ValueError(f"No valid channels selected (Range: 1-{self.num_channels})")
            
        return valid_channels

    def save_image(self):
        if self.current_image_array is None:
            QMessageBox.warning(self, "Warning", "No image loaded to save.")
            return

        # Ask for channels
        default_range = f"1-{self.num_channels}"
        text, ok = QInputDialog.getText(self, "Select Channels", 
                                        f"Enter channels to save (e.g. 1, 3-5). Max {self.num_channels}:", 
                                        QLineEdit.Normal, default_range)
        
        if not ok:
            return

        try:
            channels_to_save = self.parse_channel_string(text)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "TIFF Files (*.tiff *.tif);;All Files (*)", options=options)

        if file_path:
            try:
                # Process selected channels
                processed_channels = []
                for i in channels_to_save:
                    processed_channels.append(self.process_image_for_channel(i))
                
                # Stack channels if multi-channel
                if len(processed_channels) > 1:
                    final_image = np.stack(processed_channels, axis=0)
                else:
                    final_image = processed_channels[0]

                # Save using tifffile
                tifffile.imwrite(file_path, final_image)
                self.terminal.add_operation(f"Image saved to: {os.path.normpath(file_path)}")
                self.terminal.add_operation(f"Saved channels: {', '.join([str(c+1) for c in channels_to_save])}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                print(traceback.format_exc())

    def do_batch_process_images(self):
        self.do_process_images(mode='image')

    def do_export_csv(self):
        self.do_process_images(mode='csv')

    def do_export_fcs(self):
        self.do_process_images(mode='fcs')

    def do_process_images(self,mode='image'):
        if mode == False:#Edge case
            mode = 'image'
        if self.current_image_array is None:
            QMessageBox.warning(self, "Warning", "No image is currently displayed. Please open or display an image first.")
            return
        if self.current_channel is None:
            QMessageBox.warning(self, "Warning", "Could not determine the number of channels in the currently displayed image.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if not folder_path:
            return  # User cancelled the folder selection
        
        folder_path = os.path.normpath(folder_path)
        
        method_desc = self.agg_operation
        if not method_desc:
            method_desc = "None"
        else:
            mapping = {
                'count': 'Count',
                'mean': 'Mean',
                'area': 'Area',
                'solidity': 'Solidity',
                'scorr': 'Spatial Correlation',
                'coloc': 'Colocalisation',
                'containment': 'Containment',
                'relativeskew': 'Relative Skewness',
                'angular_momentum': 'Angular Momentum',
                'angular_entropy': 'Angular Entropy'
            }
            method_desc = mapping.get(self.agg_operation, self.agg_operation.capitalize())
        self.terminal.add_operation(f"Selected Folder to Process: {folder_path} (Quantification Method: {method_desc})")

        ppath = None
        abs_ppath = None
        transform_log = False
        vals = []
        sample_ids = []
        pname = None
        vfile = None
        old_fcsfile = None
        new_fcsfile = None
        csv_column_name = None

        if mode == 'image':
            ppath = os.path.join(folder_path, 'processed')
            abs_ppath = os.path.abspath(ppath)
            os.makedirs(ppath, exist_ok=True)
        elif mode == 'csv' or mode == 'fcs':
            if mode == 'csv':
                method_name = self.agg_operation if self.agg_operation else 'new'
                default_name = self.get_indexed_parameter_filename(folder_path, method_name)
                default_path = os.path.join(folder_path, default_name)
                
                vfile, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Save Parameter CSV File", 
                    default_path, 
                    "CSV Files (*.csv);;All Files (*)"
                )
                if not vfile:
                    return  # User cancelled save file selection
                
                vfile = os.path.normpath(vfile)
                if not vfile.lower().endswith('.csv'):
                    vfile += '.csv'
                
                csv_column_name = os.path.splitext(os.path.basename(vfile))[0]
            elif mode == 'fcs':
                try:
                    method_name = self.agg_operation if self.agg_operation else 'new'
                    default_param_name = f"{method_name}_parameter"
                    pname, ok = QInputDialog.getText(
                        self, 
                        "FCS Parameter Name", 
                        "Enter parameter name to add in FCS file:", 
                        QLineEdit.Normal, 
                        default_param_name
                    )
                    if not ok or not pname.strip():
                        return
                    pname = pname.strip()

                    old_fcsfile = self.get_fcs_files(folder_path)[0]
                    base, ext = os.path.splitext(old_fcsfile)
                    new_fcsfile = base + '_' + method_name + ext
                    reply = QMessageBox.question(self, 'Log Transformation', 
                                                 "Apply 10^x transformation to the new parameter for loglog visualisation?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    transform_log = (reply == QMessageBox.Yes)
                except Exception as e:
                    QMessageBox.critical(self, "Error", "No suitable fcs file found in directory")
                    return

        if mode == 'csv' or mode == 'fcs':
            self.terminal.add_operation(self.get_pipeline_summary())

        # First, count the total number of TIFF files to process for the progress bar
        total_files = 0
        for subdir, dirs, files in os.walk(folder_path):
            if abs_ppath and 'processed' in dirs and os.path.abspath(os.path.join(subdir, 'processed')) == abs_ppath:
                dirs.remove('processed')
            for file in files:
                if file.lower().endswith((".tif", ".tiff")):
                    total_files += 1

        if total_files == 0:
            QMessageBox.information(self, "No Files Found", "No TIFF files were found in the selected folder.")
            return

        # Create and configure the progress dialog after all file prompts are complete
        progress_dialog = QProgressDialog("Processing images...", "Cancel", 0, total_files, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("Bulk Processing")

        processed_count = 0
        progress_dialog.setValue(processed_count)
        
        # Calculate 10% increment
        ten_percent_increment = max(1, int(total_files * 0.1))

        for subdir, dirs, files in os.walk(folder_path):
            if abs_ppath and 'processed' in dirs and os.path.abspath(os.path.join(subdir, 'processed')) == abs_ppath:
                dirs.remove('processed')
            for file in files:
                if progress_dialog.wasCanceled():
                    break

                filename = os.path.join(subdir,file)
                if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
                    filepath = filename
                    try:
                        # Use Pillow to open the TIFF image and get its number of bands (channels)
                        img = np.array(tifffile.imread(filepath))
                        
                        if img.ndim == 2:
                            img = np.expand_dims(img, axis=0)

                        tiff_channels = img.shape[0]

                        if mode == 'image':
                            processed_channels = []
                            for i in range(tiff_channels):
                                processed_channels.append(self.process_image_for_channel(i, source_image_array=img))
                            
                            if len(processed_channels) > 1:
                                final_image = np.stack(processed_channels, axis=0)
                            else:
                                final_image = processed_channels[0]
                            
                            save_path = os.path.join(ppath, file)
                            tifffile.imwrite(save_path, final_image)

                        elif mode == 'csv' or mode == 'fcs':
                            # For multi-channel operations, pass the whole image.
                            # For single-channel, extract and process the channel first.
                            if self.agg_operation in self.multi_channel_ops:
                                image_to_process = img
                            elif tiff_channels > self.current_channel:
                                single_channel_img = img[self.current_channel, :, :]
                                image_to_process = self.process_image_export(single_channel_img)
                            else:
                                print(f"Skipping: {filename} (insufficient channels for single-channel operation)")
                                continue

                            vals.append(self.do_aggregation_silent(image_to_process))
                            sample_ids.append(self.derive_sample_id(file))
                        processed_count += 1
                        progress_dialog.setValue(processed_count)
                        
                        if processed_count % ten_percent_increment == 0:
                            self.terminal.add_operation(f"Processed {processed_count}/{total_files} images ({int(processed_count/total_files*100)}%)")
                            
                        QApplication.processEvents() # Keep GUI responsive

                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error processing {filename}: {e}")
            if progress_dialog.wasCanceled():
                break
        
        progress_dialog.setValue(total_files) # Ensure it shows 100%

        if progress_dialog.wasCanceled():
            self.terminal.add_operation(f"Processing cancelled by user. {processed_count} files were processed.")
            return

        if mode == 'image':
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} TIFF files into: {ppath}")
        elif mode == 'csv':
            self.param_to_csv(vals, vfile, sample_ids=sample_ids, column_name=csv_column_name)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {vfile}")
        elif mode == 'fcs':
            self.param_to_fcs(vals, old_fcsfile, new_fcsfile, pname=pname, transform=transform_log)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {new_fcsfile}")

    def process_image_export(self,image):
        for i,op in enumerate(self.operation_history):
            image = self.do_operation_silent(i,image)
        return image

    def derive_sample_id(self, filename):
        """Derives sample ID from file name:
        1. Split the file name by _ and remove extension.
        2. Take the final element of the split if it can be parsed as a number.
        3. Otherwise use the whole file name minus the extension as the sample_id.
        """
        basename = os.path.basename(filename)
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('_')
        if parts:
            last_part = parts[-1]
            try:
                int(last_part)
                return last_part
            except ValueError:
                pass
        return name_without_ext

    def param_to_csv(self, vals, vfile, sample_ids=None, column_name=None):
        if not column_name:
            column_name = getattr(self, 'agg_operation', None) or os.path.splitext(os.path.basename(vfile))[0]
        if sample_ids is not None and len(sample_ids) == len(vals):
            df = pd.DataFrame({
                'sample_id': sample_ids,
                column_name: vals
            })
            df.to_csv(vfile, index=False)
        else:
            df = pd.DataFrame({column_name: vals})
            df.to_csv(vfile, index=False)

    def param_to_fcs(self,vals,ofcs,nfcs, pname='new_param', transform=True):
        vals = np.array(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1,1)
        if transform:
            vals = 10**vals
        fcs,metadata = self.load_fcs(ofcs)
        self.add_params(fcs,nfcs,metadata,vals,pnames=pname)

    def load_fcs(self,fcsfile):
        fcdata = flowio.FlowData(fcsfile)
        fcsample = fk.Sample(fcsfile)
        metadata = fcsample.metadata
        return fcdata,metadata

    def add_param(self,fcdata,nfcs,metadata,vals,pname='new_param'):
        self.add_params(fcdata,nfcs,metadata,vals,pnames=pname)

    def add_params(self,fcdata,nfcs,metadata,vals,pnames='new_param'):
        if isinstance(pnames, str):
            pnames = [pnames]
        vals = np.array(vals)
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        
        numc = fcdata.channel_count
        events = np.reshape(fcdata.events, (-1, numc))
        
        channels = [fcdata.channels[k]['PnN'] for k in fcdata.channels.keys()]
        for name in pnames:
            channels.append(str(name))
            
        events = np.hstack([events, vals])
        events = events.flatten()
        flowio.create_fcs(open(nfcs,'wb'), events, channels, opt_channel_names=channels, metadata_dict=metadata)

    def get_fcs_files(self,directory):
        """
        Returns a list of all files in the given directory that have the suffix .fcs.

        Args:
            directory (str): The path to the directory to search.

        Returns:
            list: A list of the full paths to the .fcs files found.
                Returns an empty list if no .fcs files are found or if the
                directory does not exist.
        """
        fcs_files = []
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.lower().endswith(".fcs"):
                    full_path = os.path.join(directory, filename)
                    if os.path.isfile(full_path):  # Ensure it's a file, not a subdirectory
                        fcs_files.append(full_path)
        return fcs_files

    def get_peaks(self,image,mind=10):
        coordinates = peak_local_max(
            image, 
            min_distance=mind,  # Controls separation between peaks
            threshold_abs=0.01,# Ignores low-intensity peaks
            exclude_border=False
        )
        peakimage = np.zeros(image.shape)
        peakimage[coordinates[:,0],coordinates[:,1]]=image[coordinates[:,0],coordinates[:,1]]
        
        return peakimage

    def get_segment(self,image):
        labmask = label(image)
        if np.max(labmask)!=1:
            segmented = labmask
        else:
            edges = canny(labmask.astype('float'),sigma=1)
            distance = distance_transform_edt(edges)  # Compute distance from edges
            markers = label(self.get_peaks(distance,10)*labmask) 
            segmented =  watershed(-distance,markers=markers,mask=labmask)
            if np.sum(segmented>0)==0:
                segmented = labmask
            # segmented = distance
        return segmented

    def get_spatial_correlation(self, ch1, ch2, mask_img=None):
        """Calculates spatial correlation between two channels within a mask."""
        if mask_img is not None:
            valid_mask = mask_img > 0
        else:
            valid_mask = np.ones_like(ch1, dtype=bool)
        if np.sum(valid_mask) < 2:  # Need at least 2 points to correlate
            return 0.0
        ch1_masked = ch1[valid_mask]
        ch2_masked = ch2[valid_mask]
        return np.corrcoef(ch1_masked, ch2_masked)[0, 1]

    def get_mask(self,image,clopen=True):
        mask = (image>=threshold_otsu(image))
        if clopen:
            # This sequence is from preset1_preprocess
            mask = binary_closing(binary_opening(mask,footprint=FOOTPRINT),footprint=SQUARE)
            mask = remove_small_objects(mask,SMALL,connectivity=2)
            mask = binary_fill_holes(mask,structure=np.ones((3,3)))
        return mask

    def get_area(self,image):
        return np.sum(image!=0)

    def get_coloc(self, image, mask_image):
        mask = self.get_mask(mask_image)
        image[image<=0] = 0
        total = np.sum(image)
        coloc = np.sum(image[mask>0])
        if total>0:
            return coloc/total
        else:
            return 0.

    def get_solidity(self, image):
        mask = self.get_mask(image)
        labmask = label(mask)
        if np.max(labmask)!=1:
            solid = 0
        else:
            props = regionprops(labmask)
            if props:
                solid = props[0].solidity
            else:
                solid = 0.
        return solid

    def get_angular_momentum(self, img1, img2, sectors_power_of_2=2, weighted=True, global_mask=None, snr_checks=None):
        """
        Calculates the angular skewness of a signal (img1) normalized by the intensity
        distribution of another signal (img2), robust to cell shape asymmetry.

        The parameter measures the "center of mass" of the intensity distribution in angular
        space, and is normalized by sector area to remove the influence of cell shape.

        Args:
            img1 (np.array): Image channel for which to calculate the angular skewness.
            img2 (np.array): Image channel for the intensity-weighted centroid and sector definition.
            sectors_power_of_2 (int): The number of sectors is 2^n.
                                    1=2 sectors, 2=4 (quadrants), etc.

        Returns:
            float: The magnitude of the resultant vector, a measure of angular skewness.
        """
        mask1 = self.get_mask(img1)
        mask2 = self.get_mask(img2)
        if global_mask is None:
            mask = mask1 | mask2
        else:
            mask = self.get_mask(global_mask)
        
        if not np.any(mask):
            return 0.
        
        # Optional SNR checks for individual channels
        if snr_checks and snr_checks.get('Signal') and not np.any(mask1):
            return 0.
        if snr_checks and snr_checks.get('Reference') and not np.any(mask2):
            return 0.
        if weighted:

            # --- Calculate the Centroid for the Polar Coordinate System ---
            y_coords_2, x_coords_2 = np.where(mask2)
            img2_intensities = img2[y_coords_2, x_coords_2]
            
            reference_centroid_y = np.average(y_coords_2, weights=img2_intensities)
            reference_centroid_x = np.average(x_coords_2, weights=img2_intensities)
        else:
            y_coords, x_coords = np.where(mask)
            reference_centroid_y = np.average(y_coords)
            reference_centroid_x = np.average(x_coords)

        # --- Define Sectors and Collect Data (Vectorized) ---
        num_sectors = 2 ** sectors_power_of_2
        
        y_all, x_all = np.where(mask)
        
        # Calculate angles for all pixels at once
        angles = np.arctan2(y_all - reference_centroid_y, x_all - reference_centroid_x)
        
        # Convert angles to sector indices
        sector_indices = np.floor(((angles + np.pi) / (2 * np.pi)) * num_sectors).astype(int)
        
        # Clamp indices to the valid range (0 to num_sectors-1)
        sector_indices[sector_indices >= num_sectors] = num_sectors - 1

        # Get the intensity values for img1 at the overall mask locations
        img1_intensities_in_mask = img1[mask]
        
        # Efficiently accumulate intensities per sector using bincount
        sector_intensities = np.bincount(
            sector_indices, 
            weights=img1_intensities_in_mask, 
            minlength=num_sectors
        )
        
        # Efficiently count pixels (area) per sector
        sector_areas = np.bincount(sector_indices, minlength=num_sectors)
        
        total_img1_intensity_in_mask = np.sum(sector_intensities)
        total_pixels_in_mask = np.sum(sector_areas)

        if total_pixels_in_mask == 0 or total_img1_intensity_in_mask == 0:
            return 0.
            
        avg_intensity_total = total_img1_intensity_in_mask / total_pixels_in_mask

        # --- Normalize and Create Vectors (Vectorized) ---
        # Handle sectors with no pixels to avoid division by zero.
        avg_intensity_per_sector = np.divide(sector_intensities, sector_areas, out=np.zeros_like(sector_intensities, dtype=np.float64), where=sector_areas != 0)
        
        # Calculate the magnitude for each vector based on normalized intensity difference.
        magnitudes = avg_intensity_per_sector - avg_intensity_total
        
        # Angles for each sector's vector direction.
        sector_angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)

        # Calculate resultant vector components in a single operation.
        resultant_vector_x = np.sum(magnitudes * np.cos(sector_angles))
        resultant_vector_y = np.sum(magnitudes * np.sin(sector_angles))

        # --- Calculate the Final Parameter (Magnitude of Resultant Vector) ---
        angular_skewness = np.sqrt(resultant_vector_x**2 + resultant_vector_y**2)
        
        return angular_skewness

    def get_angular_entropy(self, img1, img2, sectors_power_of_2=2, weighted=True, global_mask=None, snr_checks=None):
        """
        Calculates the entropy of the angular distribution of a signal (img1)
        relative to a centroid defined by another signal (img2).

        This parameter measures the uniformity or randomness of the signal distribution
        in angular space. A low entropy value indicates a highly non-uniform
        distribution (e.g., concentrated in one sector), while a high entropy value
        indicates a uniform distribution across all sectors.

        Args:
            img1 (np.array): Image channel for which to calculate the angular entropy.
            img2 (np.array): Image channel for the intensity-weighted centroid and sector definition.
            sectors_power_of_2 (int): The number of sectors is 2^n.
                                    1=2 sectors, 2=4 (quadrants), etc.

        Returns:
            float: The entropy value of the normalized intensity distribution, scaled
                between 0 (non-uniform) and 1 (perfectly uniform).
        """
        mask1 = self.get_mask(img1)
        mask2 = self.get_mask(img2)
        if global_mask is None:
            mask = mask1 | mask2
        else:
            mask = self.get_mask(global_mask)

        if not np.any(mask):
            return 0.

        # Optional SNR checks for individual channels
        if snr_checks and snr_checks.get('Signal') and not np.any(mask1):
            return 0.
        if snr_checks and snr_checks.get('Reference') and not np.any(mask2):
            return 0.
        if weighted:

            # --- Calculate the Centroid for the Polar Coordinate System ---
            y_coords_2, x_coords_2 = np.where(mask2)
            img2_intensities = img2[y_coords_2, x_coords_2]
            
            reference_centroid_y = np.average(y_coords_2, weights=img2_intensities)
            reference_centroid_x = np.average(x_coords_2, weights=img2_intensities)
        else:
            y_coords, x_coords = np.where(mask)
            reference_centroid_y = np.average(y_coords)
            reference_centroid_x = np.average(x_coords)

        # --- Define Sectors and Collect Data (Vectorized) ---
        num_sectors = 2 ** sectors_power_of_2
        y_all, x_all = np.where(mask)
        angles = np.arctan2(y_all - reference_centroid_y, x_all - reference_centroid_x)
        sector_indices = np.floor(((angles + np.pi) / (2 * np.pi)) * num_sectors).astype(int)
        sector_indices[sector_indices >= num_sectors] = num_sectors - 1
        
        # Get the intensity values for img1 at the overall mask locations
        img1_intensities_in_mask = img1[mask]
        
        # Accumulate intensities per sector using bincount
        sector_intensities = np.bincount(
            sector_indices, 
            weights=img1_intensities_in_mask, 
            minlength=num_sectors
        )

        # --- Calculate the Probability Distribution ---
        total_intensity = np.sum(sector_intensities)
        if total_intensity == 0:
            return 0.
            
        probabilities = sector_intensities / total_intensity

        # --- Calculate Shannon Entropy ---
        # The term will be 0 when probability is 0, so we can ignore those entries.
        probabilities = probabilities[probabilities > 0]
        
        # Shannon entropy formula: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # --- Normalize Entropy to the number of bins ---
        # Maximum possible entropy for a given number of sectors is log2(num_sectors).
        if num_sectors <= 1:
            normalized_entropy = 0.
        else:
            max_entropy = np.log2(num_sectors)
            normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def get_shell(self, mask,thickness=2):
        core = mask>0
        inflated = mask>0
        for i in range(thickness):
            core = binary_erosion(core)
        return inflated & ~(core)

    def get_containment(self, signal_img, container_img, global_mask=None):
        signal_img[signal_img < 0] = 0.
        container_mask = self.get_mask(container_img)
        signal_mask = self.get_mask(signal_img)
        
        if global_mask is not None:
            analysis_mask = self.get_mask(global_mask)
        else:
            analysis_mask = container_mask

        tot = np.sum(signal_img[analysis_mask])
        if tot > 0 and np.any(signal_mask[analysis_mask]):
            shell = self.get_shell(container_mask)
            return 1 - np.sum(signal_img[shell & signal_mask & analysis_mask]) / tot
        return 0.

    def bskew(self, profile):
        """
        Calculates Bowley's skewness for a given distribution.
        The result is a bounded value between -1 and +1.
        """
        q1 = np.percentile(profile, 25)
        q2 = np.percentile(profile, 50)
        q3 = np.percentile(profile, 75)

        # Avoid division by zero for flat profiles
        if (q3 - q1) == 0:
            return 0.

        return (q1 + q3 - 2 * q2) / (q3 - q1)

    def get_relativeskew(self, img1, img2, type=0, weighted=True, global_mask=None, snr_checks=None):
        """
        Calculates the third conditional radial moment of the DNA signal relative to the
        intensity-weighted centroid of the membrane/cytoplasm region.

        Args:
            img1 (np.array): Pixel intensity image for channel 1.
            img2 (np.array): Pixel intensity image for the channel 2.

        Returns:
            float: The third conditional moment (skewness), or None if calculation fails.
        """
        mask1 = self.get_mask(img1)
        mask2 = self.get_mask(img2)
        if global_mask is None:
            mask = mask1 | mask2
        else:
            mask = self.get_mask(global_mask)
        
        if not np.any(mask):
            return 0.
        
        # Optional SNR checks for individual channels
        if snr_checks and snr_checks.get('Signal') and not np.any(mask1):
            return 0.
        if snr_checks and snr_checks.get('Reference') and not np.any(mask2):
            return 0.

        if weighted:
            # --- Calculate the Centroid for the Polar Coordinate System ---
            y_coords_2, x_coords_2 = np.where(mask2)
            img2_intensities = img2[y_coords_2, x_coords_2]
            
            reference_centroid_y = np.average(y_coords_2, weights=img2_intensities)
            reference_centroid_x = np.average(x_coords_2, weights=img2_intensities)
        else:
            if not np.any(mask) or not np.any(mask1):
                return 0.     
            y_coords, x_coords = np.where(mask)

            reference_centroid_y = np.average(y_coords)
            reference_centroid_x = np.average(x_coords)

        # --- Create the Radial Intensity Profile ---
        # Use vectorized numpy operations to calculate distances for all pixels at once.
        y_full, x_full = np.where(mask)
        distances_from_centroid = np.sqrt(
            (y_full - reference_centroid_y)**2 + (x_full - reference_centroid_x)**2
        )

        # Get the DNA intensity values for the pixels inside the mask
        img1_in_mask = img1[mask]

        # Use a histogram-like approach to bin distances and average intensities
        num_bins = int(np.max(distances_from_centroid)) + 1
        
        # Use binned_statistic for efficiency.
        profile, _, _ = binned_statistic(
            distances_from_centroid, 
            img1_in_mask, 
            statistic='mean', 
            bins=num_bins
        )

        # Clean up the profile by removing NaN values
        profile = profile[~np.isnan(profile)]

        # --- Calculate the Third Moment (Skewness) of the Profile ---
        if len(profile) < 4:
            # print("Warning: Not enough data points to calculate a meaningful skewness.")
            return 0
        
        if type:
            third_moment = self.bskew(profile)
        else:
            third_moment = skew(profile)

        return third_moment

    def get_mean(self,image):
        return np.mean(image!=0)

    def get_count(self,image):
        uniq = np.unique(image)
        luniq = len(uniq)
        if 0 in uniq:
            luniq -= 1
        return luniq

    def get_label(self,image):
        return label(image)

    def gaussblur(self,image,sigma=2):
        if isinstance(sigma, (dict, list, tuple)):
            if isinstance(sigma, dict):
                ksize = tuple(sigma.get('kernel', (5, 3)))
                sigX = float(sigma.get('sigmaX', 2.5))
                sigY = float(sigma.get('sigmaY', 0.8))
            elif isinstance(sigma, (list, tuple)):
                if len(sigma) >= 4:
                    ksize = (int(sigma[0]), int(sigma[1]))
                    sigX = float(sigma[2])
                    sigY = float(sigma[3])
                else:
                    ksize = (5, 3)
                    sigX = float(sigma[0]) if len(sigma) > 0 else 2.5
                    sigY = float(sigma[1]) if len(sigma) > 1 else 0.8
            return cv2.GaussianBlur(image.astype(np.float32), ksize, sigmaX=sigX, sigmaY=sigY)
        return gaussian(image, float(sigma), mode='wrap')

    
    # --------------------------------------------------------------------------
    # GUI Menu Bar & Options Actions
    # --------------------------------------------------------------------------
    def create_menus(self):
        menu_bar = self.menuBar()

        # --- Preprocessing Submenu ---
        self.preprocessing_menu = menu_bar.addMenu('&Preprocessing')

        filters_submenu = QMenu('&Filter', self)
        gauss = QAction('&Gaussian Filter',self)
        gauss.triggered.connect(self.open_gauss)
        filters_submenu.addAction(gauss)
        breg_action = QAction('&Bregman Denoising',self)
        breg_action.setEnabled(False)
        filters_submenu.addAction(breg_action)

        # New manipulation actions
        manipulation_submenu = self.preprocessing_menu.addMenu('&Manipulation')
        crop_action = QAction('&Crop', self)
        crop_action.triggered.connect(self.open_crop_dialog)
        rescale_action = QAction('&Rescale', self)
        rescale_action.triggered.connect(self.open_rescale_dialog)
        manipulation_submenu.addAction(crop_action)
        manipulation_submenu.addAction(rescale_action)


        segmentation_submenu = QMenu('&Segmentation', self)
        mask = QAction('&Mask Otsu',self)
        mask.triggered.connect(self.do_mask)
        mlabel = QAction('&Label Image',self)
        mlabel.triggered.connect(self.do_label)
        segment = QAction('&Segment',self)
        segment.triggered.connect(self.do_segment)
        segmentation_submenu.addAction(mask)
        segmentation_submenu.addAction(segment)
        segmentation_submenu.addAction(mlabel)

        self.presets_submenu = QMenu('&Presets', self)
        self.refresh_presets_menu()

        # Add Undo action
        undo_action = QAction('&Undo Last Operation', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_last_operation)

        # Add Redo action
        redo_action = QAction('&Redo Operation', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.triggered.connect(self.redo_operation)

        reset_action = QAction('&Reset Preprocessing', self)
        reset_action.triggered.connect(self.reset_operations)

        save_image_action = QAction('&Save Single Image (.tiff)', self)
        save_image_action.triggered.connect(self.save_image)

        batch_process_action = QAction('&Batch Process Folder', self)
        batch_process_action.triggered.connect(self.do_batch_process_images)


        self.preprocessing_menu.addMenu(self.presets_submenu)
        self.preprocessing_menu.addMenu(filters_submenu)
        self.preprocessing_menu.addMenu(manipulation_submenu)
        self.preprocessing_menu.addMenu(segmentation_submenu)

        # --- Quantify Submenu ---
        self.quantify_menu = menu_bar.addMenu('&Quantify')
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(undo_action)
        self.preprocessing_menu.addAction(redo_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(reset_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(save_image_action)
        self.preprocessing_menu.addAction(batch_process_action)

        aggregation_submenu = QMenu('&Aggregation', self)
        geometry_submenu = QMenu('&Geometry', self)
        self.quantify_menu.addMenu(aggregation_submenu)
        self.quantify_menu.addMenu(geometry_submenu)

        # Aggregation Actions
        self.count_action = QAction("Count (unique)", self, checkable=True)
        self.count_action.setChecked(True)
        self.mean_action = QAction("Mean (non-zero)", self, checkable=True)
        self.area_action = QAction("Area (non-zero)", self, checkable=True)
        
        # Geometry Actions
        self.solidity_action = QAction("Solidity", self, checkable=True)
        self.solidity_action.setObjectName("solidity_action")
        self.coloc_action = QAction("Colocalisation", self, checkable=True)
        self.coloc_action.setObjectName("coloc_action")
        self.containment_action = QAction("Containment (Signal, Container, Optional Mask)", self, checkable=True)
        self.containment_action.setObjectName("containment_action")
        self.relativeskew_action = QAction("Relative Skewness (Signal, Reference, Optional Mask)", self, checkable=True)
        self.relativeskew_action.setObjectName("relativeskew_action")
        self.angular_momentum_action = QAction("Angular Momentum (Signal, Reference, Optional Mask)", self, checkable=True)
        self.angular_momentum_action.setObjectName("angular_momentum_action")
        self.angular_entropy_action = QAction("Angular Entropy (Signal, Reference, Optional Mask)", self, checkable=True)
        self.angular_entropy_action.setObjectName("angular_entropy_action")
        self.scorr_action = QAction("Spatial Correlation (Optional Mask, Chan1, Chan2)", self, checkable=True)
        self.scorr_action.setObjectName("scorr_action")

        aggregation_group = QActionGroup(self)
        aggregation_group.triggered.connect(self.enable_aggregation)
        for action in [self.count_action, self.mean_action, self.area_action, self.solidity_action,
                       self.coloc_action, self.containment_action, self.relativeskew_action,
                       self.angular_momentum_action, self.angular_entropy_action, self.scorr_action]:
            aggregation_group.addAction(action)

        aggregation_submenu.addAction(self.count_action)
        aggregation_submenu.addAction(self.mean_action)
        
        geometry_submenu.addActions([self.solidity_action, self.coloc_action, self.containment_action, self.relativeskew_action, self.angular_momentum_action, self.angular_entropy_action, self.scorr_action])
        geometry_submenu.addAction(self.area_action)

        # --- Parameters Submenu ---
        self.parameters_menu = menu_bar.addMenu('&Parameters')

        self.export_to_fcs = QAction('Export to FCS',self)
        self.export_to_fcs.triggered.connect(self.do_export_fcs)
        self.export_to_csv = QAction('Export to CSV',self)
        self.export_to_csv.triggered.connect(self.do_export_csv)
        self.export_terminal = QAction('Export Terminal', self)
        self.export_terminal.triggered.connect(self.export_terminal_history)
        self.concat_csv = QAction('Concatenate CSVs', self)
        self.concat_csv.triggered.connect(self.do_concat_csvs)
        self.merge_csv_to_fcs = QAction('Merge CSV into FCS', self)
        self.merge_csv_to_fcs.triggered.connect(self.do_merge_csv_to_fcs)
        self.parameters_menu.addActions([self.export_to_fcs, self.export_to_csv, self.export_terminal, self.concat_csv, self.merge_csv_to_fcs])

        self.preprocessing_menu.setEnabled(False)
        self.quantify_menu.setEnabled(False)
        self.parameters_menu.setEnabled(True)
        self.export_to_fcs.setEnabled(False)
        self.export_to_csv.setEnabled(False)

        # Refine Menu (formerly File)
        refine_menu = menu_bar.addMenu('&Refine')

        save_action = QAction('Save Output as CSV', self)
        save_action.triggered.connect(self.save_output)
        refine_menu.addAction(save_action)

        load_action = QAction('Load Output CSV for Comparison', self)
        load_action.triggered.connect(self.compare_output)
        refine_menu.addAction(load_action)

        self.calc_ci_action = QAction('Calculate Importance CIs', self, checkable=True)
        self.calc_ci_action.setChecked(False)
        self.calc_ci_action.triggered.connect(self.configure_ci_alpha)
        refine_menu.addAction(self.calc_ci_action)

        refine_menu.addSeparator()
        
        self.ri_group = QActionGroup(self)
        self.lsri_action = QAction('Use lsRI (Laplacian Score)', self, checkable=True)
        self.pri_action = QAction('Use pRI (PCA-based)', self, checkable=True)
        self.sri_action = QAction('Use sRI (SOM-based)', self, checkable=True)
        self.miri_action = QAction('Use miRI (Mutual Information)', self, checkable=True)
        self.lsri_action.setChecked(True)
        
        self.ri_group.addAction(self.lsri_action)
        self.ri_group.addAction(self.pri_action)
        self.ri_group.addAction(self.sri_action)
        self.ri_group.addAction(self.miri_action)
        
        ri_menu = refine_menu.addMenu('RI Metric')
        ri_menu.addAction(self.lsri_action)
        ri_menu.addAction(self.pri_action)
        ri_menu.addAction(self.sri_action)
        ri_menu.addAction(self.miri_action)

        refine_menu.addSeparator()
        
        preferences_action = QAction('Preferences...', self)
        preferences_action.triggered.connect(self.open_refine_preferences)
        refine_menu.addAction(preferences_action)

        # Help menu (last)
        help_menu = menu_bar.addMenu('&Help')
        readme_action = QAction('README / User Guide', self)
        readme_action.triggered.connect(self.show_readme)
        about_action = QAction('&About FlowFI (v1.6.0)...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(readme_action)
        help_menu.addAction(about_action)

    def configure_ci_alpha(self):
        if self.calc_ci_action.isChecked():
            dialog = AlphaDialog(self, default_alpha=self.ci_alpha, default_boots=self.ci_boots)
            if dialog.exec_() == QDialog.Accepted:
                self.ci_alpha = dialog.get_alpha()
                self.ci_boots = dialog.get_boots()
                self.calculate_cis()
            else:
                self.calc_ci_action.setChecked(False)

    def compare_output(self):
        if hasattr(self, 'result'):
            self.load_output()
        else:
            QMessageBox.information(self, "Error", "No complete results to compare to yet.")

    def save_output(self):
        if not self.result:
            QMessageBox.warning(self, "Warning", "There is no output to save.")
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "CSV Files (*.csv)", options=options)
        if filepath:
            metric_name = getattr(self.worker, 'metric_name', 'ls') if hasattr(self, 'worker') else 'ls'
            try:
                with open(filepath, 'w', newline='') as csvfile:
                    if not hasattr(self,"loaded_result"):
                        fieldnames = ['feature','ri', metric_name,'membership','centrality']
                        if 'LowCI' in self.result:
                            fieldnames += ['LowCI', 'UpperCI']
                    else:
                        fieldnames = ['feature','ri', metric_name,'membership','centrality',"comparison"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    result = self.result['raw_score']
                    impresult = self.result['Relative Importance']
                    columns = self.columns
                    memb = self.result['Membership']
                    centrality = self.result['Centrality']
                    if not hasattr(self,"loaded_result"):
                        for i in range(len(result)):
                            row_data = {'feature': columns[i], 'ri': impresult[i], metric_name: result[i],'membership':memb[i],'centrality': centrality[i]}
                            if 'LowCI' in self.result:
                                row_data['LowCI'] = self.result['LowCI'][i]
                                row_data['UpperCI'] = self.result['UpperCI'][i]
                            writer.writerow(row_data)
                    else:
                        comparison = self.result['Comparison']
                        for i in range(len(result)):
                            writer.writerow({'feature': columns[i], 'ri': impresult[i], metric_name: result[i],'membership':memb[i],'centrality': centrality[i],'comparison': comparison[i]})
                QMessageBox.information(self, "Success", "Output successfully saved to CSV file.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save output to CSV file: {e}")

    def show_readme(self):
        dialog = HelpDialog(self)
        dialog.exec_()

    def show_about(self):
        if self.isVisible():
            QMessageBox.about(
                self,
                "About FlowFI",
                "<h3>FlowFI</h3>"
                "<p><b>Version:</b> 1.6.0</p>"
                "<p>FlowFI is a flow cytometry analysis and bespoke parameter generation tool.</p>"
            )

    def open_refine_preferences(self):
        dataset_size = None
        if hasattr(self, 'data') and self.data is not None:
            dataset_size = self.data.shape[0]

        dialog = RefinePreferencesDialog(self, default_boots=self.boots_param, default_bootsize=self.bootsize_param, dataset_size=dataset_size,
                                         default_conv_check=self.convergence_check, default_conv_threshold=self.convergence_threshold)
        if dialog.exec_() == QDialog.Accepted:
            self.boots_param, self.bootsize_param, self.convergence_check, self.convergence_threshold = dialog.get_values()
            self.terminal.add_operation(f"Refine parameters updated: BOOT={self.boots_param}, BOOTSIZE={self.bootsize_param}, ConvCheck={self.convergence_check}, Threshold={self.convergence_threshold}")

    def attempt_sort(self,index):
        if index == 4:
            if not hasattr(self,"loaded_result"):
                self.compare_output()
                if self.finalcluster==False:
                    self.sort_dropdown.setCurrentIndex(0)
                    self.update_display()
            else:
                self.update_display()
        else:
            self.update_display()
    def load_output(self,index=0):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Output CSV", "", "CSV Files (*.csv)", options=options)
        if filepath:
            try:
                loaded_result = {}
                with open(filepath, 'r') as csvfile:
                    loaded_result = {}
                    reader = csv.DictReader(csvfile)
                    fieldnames = reader.fieldnames
                    for f in fieldnames:
                        loaded_result[f] = []
                    for line in reader:
                        for f in line.keys():
                            loaded_result[f] += [line[f]]
                # print(fieldnames)
                
                metric_keys = ['lsRI', 'pRI', 'sRI', 'miRI']
                loaded_metric_key = None
                for mk in metric_keys:
                    if mk in fieldnames:
                        loaded_metric_key = mk
                        break

                for f in fieldnames:
                    if f!='feature':
                        loaded_result[f] = np.array(loaded_result[f]).astype('float')
                    else:
                        loaded_result[f] = np.array(loaded_result[f])
                
                if loaded_metric_key and loaded_metric_key != 'raw_score':
                    loaded_result['raw_score'] = loaded_result[loaded_metric_key]
                        
                self.loaded_result = loaded_result
                self.update_display()
                # QMessageBox.information(self, "Success", "Output successfully loaded from CSV file.")
            except Exception as e:
                self.sort_dropdown.setCurrentIndex(index)
                self.update_display()

                QMessageBox.critical(self, "Error", f"Failed to load output from CSV file: {e}")
    def browse_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)", options=options)
        if filepath:
            self.filepath_input.setText(filepath)
    def execute_function(self):
        filepath = self.filepath_input.text()
        if not filepath:
            QMessageBox.warning(self, "Warning", "Please enter a valid file path.")
            return
        
        self.filepath = filepath
        self.load_features()
        if not hasattr(self, 'data'):
            QMessageBox.warning(self, "Warning", "No features found in the FCS file.")
            return

        sample_size = self.data.shape[0]
        min_algo_sample_requirement = 16

        if sample_size < min_algo_sample_requirement:
            QMessageBox.critical(
                self,
                "Insufficient Sample Size",
                f"The dataset contains only {sample_size} sample event(s).\n\n"
                f"FlowFI algorithms require a minimum of {min_algo_sample_requirement} sample events to function. Processing aborted."
            )
            return

        if sample_size < 200:
            QMessageBox.warning(
                self,
                "Small Sample Size Warning",

                f"The loaded dataset contains only {sample_size} sample event(s) (below the recommended minimum of 200 events).\n\n"
                f"Statistical metrics, feature importance rankings, and confidence intervals may have higher variance on small sample sizes."
            )

        num_variable = int(np.sum(self.is_variable))
        num_constant = len(self.is_variable) - num_variable

        if num_variable < 3:
            QMessageBox.warning(
                self,
                "Too Many Constant Parameters",
                f"Analysis requires at least 3 variable parameters, but found only {num_variable} variable parameter(s) "
                f"({num_constant} parameter(s) are constant/non-variable).\n\n"
                f"FlowFI algorithms require at least 3 variable parameters. "
                f"Please select additional feature categories or load data with more variable parameters."
            )
            return

        if num_constant > 0:
            self.terminal.add_operation(f"Identified {num_constant} constant/non-variable parameter(s). Rated as 0 Relative Importance (RI).")
        
        self.execute_button.setEnabled(False)
        self.start_time = time.time()
        self.output_layout.removeWidget(self.output_widget)
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)

        self.progress_bar.setValue(0)
        
        if self.sri_action.isChecked():
            metric_name = "sRI"
        elif self.miri_action.isChecked():
            metric_name = "miRI"
        elif self.pri_action.isChecked():
            metric_name = "pRI"
        else:
            metric_name = "lsRI"
            
        self.is_cost = metric_name == "lsRI"

        self.variable_data = self.data[:, self.is_variable]

        self.worker = WorkerThread(self.variable_data, boots=self.boots_param, bootsize=self.bootsize_param, 
                                   conv_check=self.convergence_check, conv_threshold=self.convergence_threshold,
                                   metric_name=metric_name)
        self.boots = self.worker.boots
        self.feature_averages = np.zeros((num_variable, self.boots))
        self.calculated = np.zeros((self.boots))
        self.medoids = np.zeros((num_variable, self.boots))
        self.memberships = np.zeros((num_variable, self.boots))
        self.finalcluster = False

        self.worker.intermediate_result.connect(self.add_result)
        self.worker.result_ready.connect(self.finalize_results)
        self.worker.start()

        self.update_timer.setInterval(10000)
        self.update_timer.start()
        QApplication.processEvents()
    def load_features(self):
        try:
            if self.filepath.lower().endswith('.csv'):
                df = pd.read_csv(self.filepath)
                id_cols = [col for col in df.columns if str(col).lower() in ['sample_id', 'sample_ids', 'sample', 'id', 'filename', 'index', 'unnamed: 0']]
                if id_cols:
                    df = df.drop(columns=id_cols)
                df = df.select_dtypes(include=[np.number])
                self.columns = np.array(df.columns)
                self.data = df.values
            else:
                fcdata = flowio.FlowData(self.filepath)
                self.columns = np.array([fcdata.channels[c]['PnN'] for c in fcdata.channels])
                self.data = np.reshape(fcdata.events,[-1,fcdata.channel_count])
            self.cleandata() 
        except Exception as e:
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load features from FCS file: {e}")
    def NormalizeData(self, data):
        if len(data) == 0:
            return data
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data, dtype=float)
        return (data - min_val) / (max_val - min_val)
    def cleandata(self, norm=True): 
        excludedcols_lower = [str(e).lower() for e in excludedcols]
        included = [i for i, c in enumerate(self.columns) if str(c).lower() not in excludedcols_lower]
        self.columns = self.columns[included]
        self.data = self.data[:, included]

        UVpattern = r'^UV\d+.*'
        Vpattern = r'^V\d+.*'
        Bpattern = r'^B\d+.*'
        YGpattern = r'^YG\d+.*'
        Rpattern = r'^R\d+.*'
        ImgBpattern = r'^ImgB\d+.*'
        Imagingpattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*'

        patterns = [UVpattern, Vpattern, Bpattern, YGpattern, Rpattern, ImgBpattern, Imagingpattern]

        self.patternmatches = np.ones(len(self.columns)) * len(patterns)
        self.patternmatches = self.patternmatches.astype(int)

        for k, p in enumerate(patterns):
            matches = [i for i, c in enumerate(self.columns) if re.match(p, c)]
            self.patternmatches[matches] = k
        sort = np.argsort(self.patternmatches)
        self.patternmatches = self.patternmatches[sort]
        self.columns = self.columns[sort]
        self.data = self.data[:, sort]

        self.fcolors = np.array([self.colors[c] for c in self.patternmatches])
        self.flabels = np.array([self.ftypes[c] for c in self.patternmatches])
        
        self.filter = [i for i, f in enumerate(self.flabels) if f in self.selected_feature_types]
        self.patternmatches = self.patternmatches[self.filter]
        self.columns = self.columns[self.filter]
        self.data = self.data[:, self.filter]
        self.flabels = self.flabels[self.filter]
        self.fcolors = self.fcolors[self.filter]

        # Determine non-variable parameters strictly by variance
        self.is_variable = np.var(self.data, axis=0) > 1e-8

        if norm:
            if np.any(self.is_variable):
                self.data[:, self.is_variable] = StandardScaler().fit_transform(self.data[:, self.is_variable])
    def add_result(self, result):
        value = result['value']
        i = result['i']
        self.medoids[list(result['medoids'].astype(int)), i] += 1
        self.memberships[:, i] = result['membership']
        self.feature_averages[:, i] = value
        self.calculated[i] = 1
        non0 = self.calculated > 0
        imp_calculated = self.feature_averages[:, non0]
        var_mean_value = np.mean(imp_calculated, axis=1).flatten()
        var_mdds = np.sum(self.medoids[:, non0], axis=1).flatten()

        total_num = len(self.is_variable)
        var_indices = np.where(self.is_variable)[0]
        const_indices = np.where(~self.is_variable)[0]

        full_raw_score = np.zeros(total_num)
        full_raw_score[var_indices] = var_mean_value
        if len(const_indices) > 0 and len(var_indices) > 0:
            if getattr(self, 'is_cost', True):
                full_raw_score[const_indices] = np.max(var_mean_value)
            else:
                full_raw_score[const_indices] = np.min(var_mean_value)

        full_mdds = np.zeros(total_num)
        full_mdds[var_indices] = var_mdds

        full_membership = np.zeros(total_num, dtype=int)
        full_membership[var_indices] = result['membership']

        full_ri = np.zeros(total_num)
        if len(var_indices) > 0:
            if getattr(self, 'is_cost', True):
                full_ri[var_indices] = 1 - self.NormalizeData(var_mean_value)
            else:
                full_ri[var_indices] = self.NormalizeData(var_mean_value)
        full_ri[const_indices] = 0.0

        self.result = {
            'raw_score': full_raw_score,
            'Relative Importance': full_ri,
            'i': i,
            'medoids': full_mdds,
            'membership': full_membership
        }
    def color_name_to_rgba(self, color_name):
        try:
            rgba = mcolors.to_rgba(color_name)
            return rgba
        except ValueError:
            return (0, 0, 0, 0)
    def update_display(self):
        self.selected_feature_types = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]

        if hasattr(self, 'result'):
            if self.centrality_checkbox.isChecked():
                central_features = [i for i, m in enumerate(self.result['medoids']) if m > 0]
                filter = [i for i, f in enumerate(self.flabels) if f in self.selected_feature_types and i in central_features]
            else:
                filter = [i for i, f in enumerate(self.flabels) if f in self.selected_feature_types]
            self.output_layout.removeWidget(self.output_widget)
            self.output_widget = QWidget()
            self.output_layout = QVBoxLayout()
            self.output_widget.setLayout(self.output_layout)
            self.output_panel.setWidget(self.output_widget)

            if 'Relative Importance' in self.result:
                mean_value = self.result['Relative Importance'][filter]
            elif getattr(self, 'is_cost', True):
                mean_value = 1 - self.NormalizeData(self.result['raw_score'])[filter]
            else:
                mean_value = self.NormalizeData(self.result['raw_score'])[filter]
            
            loaded_final = self.finalcluster and hasattr(self, "loaded_result")
            if loaded_final:
                ffeatures = self.columns[filter]
                loaded_ffeatures = self.loaded_result['feature']

                loaded_orderedimp = np.zeros(len(ffeatures))
                orderedimp = np.zeros(len(ffeatures))
                for i in range(len(ffeatures)):
                    if ffeatures[i] in loaded_ffeatures:
                        ind = int(np.where(ffeatures[i] == loaded_ffeatures)[0][0])
                        loaded_orderedimp[i] = self.loaded_result['raw_score'][ind]
                        orderedimp[i] = self.result['raw_score'][filter][i]
                    else:
                        orderedimp[i] = -1
                        loaded_orderedimp[i] = -1
                    
                orderedimp[orderedimp >= 0] = orderedimp[orderedimp >= 0].argsort().argsort()
                loaded_orderedimp[loaded_orderedimp >= 0] = loaded_orderedimp[loaded_orderedimp >= 0].argsort().argsort()
                rankdiffs = np.zeros(len(ffeatures))
                rankdiffs[orderedimp >= 0] = orderedimp[orderedimp >= 0] - loaded_orderedimp[loaded_orderedimp >= 0]
                rankdiffs[orderedimp == -1] = np.nan
                self.result['Comparison'] = -rankdiffs

            # Sort the results based on the dropdown selection
            sorting = True
            if "Sort by: Importance" in self.sort_dropdown.currentText():
                sort = np.argsort(-mean_value)
                sorting = False
            else:
                second = -mean_value
                if "Sort by: Type" in self.sort_dropdown.currentText():
                    first = self.flabels[filter]
                elif "Sort by: Centrality" in self.sort_dropdown.currentText():
                    first = -self.result['medoids'][filter]
                elif "Sort by: Cluster" in self.sort_dropdown.currentText() and self.finalcluster:
                    first = self.membership[filter]
                elif "Sort by: Change" in self.sort_dropdown.currentText() and loaded_final:
                    first = rankdiffs       
                else: # If nothing else works (i.e. clustering not ready) then sort by Importance
                    sort = np.argsort(second)
                    sorting = False
            if sorting:
                sort = np.lexsort([second, first])
                sorting = False

            colors = self.fcolors[filter][sort]
            mean_value = mean_value[sort]
            medoids = self.result['medoids'][filter][sort]
            topmeds = np.where(medoids > 0)[0]
            texts = self.columns[filter][sort]
            labels = self.flabels[filter][sort]

            if loaded_final:
                rankdiffs = rankdiffs[sort]

            if hasattr(self, 'worker') and self.worker.early:
                self.worker.progress = self.boots    
            prog = int(100 * self.worker.progress / self.boots) if hasattr(self, 'worker') else 100
            self.progress_bar.setValue(prog)
            if self.finalcluster:
                membership = self.membership[filter][sort]
                memcolors = [self.clustercolors[m] for m in membership]

            has_ci = 'LowCI' in self.result and 'UpperCI' in self.result
            if has_ci:
                low_ci = self.result['LowCI'][filter][sort]
                upper_ci = self.result['UpperCI'][filter][sort]

            for i in range(len(filter)):
                entry_layout = QHBoxLayout()
                text = texts[i]
                if loaded_final:
                    if -rankdiffs[i] > 0:
                        text += ' (+' + str(int(-rankdiffs[i])) + ')'
                    elif -rankdiffs[i] <= 0:
                        text += ' (' + str(int(-rankdiffs[i])) + ')'
                text_label = QLabel(text)
                if self.finalcluster:
                    if i in topmeds:
                        text_label.setStyleSheet(f"color: {colors[i]};font-weight: bold;border: 3px solid {memcolors[i]};text-decoration: underline")
                    else:
                        text_label.setStyleSheet(f"color: {colors[i]};border: 3px solid {memcolors[i]};")
                    entry_layout.addWidget(text_label)
                else:
                    if i in topmeds:
                        text_label.setStyleSheet(f"color: {colors[i]};font-weight: bold;text-decoration: underline")
                    else:
                        text_label.setStyleSheet(f"color: {colors[i]};")
                    entry_layout.addWidget(text_label)

                stroke = colors[i]
                if self.finalcluster:
                    stroke = memcolors[i]
                
                l_ci = low_ci[i] if has_ci else None
                u_ci = upper_ci[i] if has_ci else None
                bar = BarWidget(mean_value[i], colors[i], l_ci, u_ci, stroke_color=stroke)
                entry_layout.addWidget(bar)

                entry_widget = QWidget()
                entry_widget.setLayout(entry_layout)
                self.output_layout.addWidget(entry_widget)

            self.output_widget.adjustSize()
            QApplication.processEvents()
    def show_processing_time(self):
        text = "Processing time: " + str(self.total_time) + 's'
        QMessageBox.information(self, "Processing Time", text)
    def consensusclustering_final(self):
        var_membership = self.worker.getclust(self.memberships)
        total_num = len(self.is_variable)
        self.membership = np.zeros(total_num, dtype=int)
        self.membership[self.is_variable] = var_membership
        self.finalcluster = True
        if EVAL == True:
            self.end_time = time.time()
            self.total_time = np.round(self.end_time - self.start_time, 2)
            self.show_processing_time()
        self.execute_button.setEnabled(True)
    def finalize_results(self):
        if self.worker.early:
            self.memberships = self.memberships[:, self.calculated > 0]
            self.feature_averages = self.feature_averages[:, self.calculated > 0]
            self.medoids = self.medoids[:, self.calculated > 0]
        self.output_widget.adjustSize()

        self.consensusclustering_final()

        total_num = len(self.is_variable)
        var_indices = np.where(self.is_variable)[0]
        const_indices = np.where(~self.is_variable)[0]

        full_ri = np.zeros(total_num)
        if len(var_indices) > 0:
            var_raw = self.result['raw_score'][var_indices]
            if getattr(self, 'is_cost', True):
                full_ri[var_indices] = 1 - self.NormalizeData(var_raw)
            else:
                full_ri[var_indices] = self.NormalizeData(var_raw)
        full_ri[const_indices] = 0.0
        self.result['Relative Importance'] = full_ri

        self.update_display()
        self.calculate_cis()

        full_centrality = np.zeros(total_num)
        if len(var_indices) > 0:
            full_centrality[var_indices] = self.NormalizeData(self.result['medoids'][var_indices])
        full_centrality[const_indices] = 0.0

        self.result['Centrality'] = full_centrality
        self.result['Membership'] = self.membership
        QMessageBox.information(self, "Information", "Processing complete!")
        self.update_timer.stop()  # Stop the update timer
    def calculate_cis(self):
        if self.calc_ci_action.isChecked() and hasattr(self, 'result') and hasattr(self, 'feature_averages'):
            total_num = len(self.is_variable)
            var_indices = np.where(self.is_variable)[0]
            const_indices = np.where(~self.is_variable)[0]

            var_raw_mean = self.result['raw_score'][var_indices]
            s = self.feature_averages.shape[1]
            numf = self.feature_averages.shape[0]

            if numf > 0 and s > 0:
                sample_ris = np.zeros([self.ci_boots, numf])
                sample_raw = np.zeros([self.ci_boots, numf])
                for i in range(self.ci_boots):
                    temp_ls = self.feature_averages[:, np.random.choice(s, s, replace=True)]
                    temp_mean = np.mean(temp_ls, axis=1)

                    if getattr(self, 'is_cost', True):
                        sample_ris[i, :] = 1 - self.NormalizeData(temp_mean)
                    else:
                        sample_ris[i, :] = self.NormalizeData(temp_mean)
                    sample_raw[i, :] = temp_mean

                raw_lcis = np.array([np.percentile(sample_raw[:, m], self.ci_alpha / 2) for m in np.arange(numf)])
                raw_ucis = np.array([np.percentile(sample_raw[:, m], 100 - self.ci_alpha / 2) for m in np.arange(numf)])

                if getattr(self, 'is_cost', True):
                    obs_max = np.max(var_raw_mean)
                    obs_min = np.min(var_raw_mean)
                    lb_violation = [obs_max <= raw_ucis[m] for m in range(numf)]
                    ub_violation = [obs_min >= raw_lcis[m] for m in range(numf)]
                else:
                    obs_max = np.max(var_raw_mean)
                    obs_min = np.min(var_raw_mean)
                    lb_violation = [obs_min >= raw_lcis[m] for m in range(numf)]
                    ub_violation = [obs_max <= raw_ucis[m] for m in range(numf)]

                lcis = np.array([np.percentile(sample_ris[:, m], self.ci_alpha / 2) for m in np.arange(numf)])
                ucis = np.array([np.percentile(sample_ris[:, m], 100 - self.ci_alpha / 2) for m in np.arange(numf)])
                lcis[lb_violation] = 0.
                ucis[ub_violation] = 1.

                full_lcis = np.zeros(total_num)
                full_ucis = np.zeros(total_num)
                full_lcis[var_indices] = lcis
                full_ucis[var_indices] = ucis
                full_lcis[const_indices] = 0.0
                full_ucis[const_indices] = 0.0

                self.result['LowCI'] = full_lcis
                self.result['UpperCI'] = full_ucis
                self.update_display()



# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    if app is None:
        app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.showMaximized()

    if splash:
        splash.finish(main_window)

    sys.exit(app.exec_())

