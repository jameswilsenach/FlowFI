import sys
import os
import csv
import re
import time
import traceback

# Third-party imports for data and analysis
import flowio
import flowkit as fk
import numpy as np
import pandas as pd # Example: Though unused directly, good to have for clarity if debugging
from scipy.stats import kendalltau
from sklearn.metrics import pairwise_distances,adjusted_mutual_info_score
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import leidenalg as la
import igraph as ig

# Third-party imports for GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QCheckBox, QPushButton, QProgressBar, QLabel, QFileDialog, QScrollArea, QFrame,
                             QAction, QMessageBox,QComboBox,QTabWidget,QSplitter, QFileSystemModel,QTreeView,QSlider,QMenu,QTextEdit,QSizePolicy,QDialog,QActionGroup, QDialogButtonBox, QGridLayout, QProgressDialog, QSplashScreen)
from PyQt5.QtCore import QThread, pyqtSignal,  QTimer, Qt, QDir
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QBrush, QPen, QColor
import matplotlib
import matplotlib.colors as mcolors

# Third-party imports for image processing (Feature Design)
import tifffile
import cv2
from scipy.ndimage import binary_fill_holes, binary_erosion, distance_transform_edt
from skimage.filters import threshold_otsu,gaussian
from skimage.morphology import binary_opening,binary_closing,disk,remove_small_objects
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max,canny
from scipy.stats import skew, binned_statistic

EVAL = False
#SIM = False
BOOT = 200
CLUSTERS = 3
MEDS = CLUSTERS
BOOTSIZE = 1000
THRESHOLD = 1e-5
# KFRAC = 1./3
alpha = 5
# KFRAC = 1./10
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

# %matplotlib ipympl

matplotlib.use('Qt5Agg')
excludedcols = ['Saturated', 'Time', 'Sorted', 'Row', 'Column']
excludedcols += ['Protocol', 'EventLabel', 'Regions0', 'Regions1', 'Regions2',
       'Regions3', 'Gates', 'IndexSort', 'SaturatedChannels', 'PhaseOffset',
       'PlateLocationX', 'PlateLocationY', 'EventNumber0', 'EventNumber1',
       'DeltaTime0', 'DeltaTime1', 'DropId', 'SaturatedChannels1',
       'SaturatedChannels2', 'SpectralEventWidth', 'EventWidthInDrops',
       'SpectralUnmixingFlags', 'WaveformPresent']


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

# Path to the global CSV file containing feature names

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

    def __init__(self, data, boots=BOOT, bootsize=BOOTSIZE, conv_check=True, conv_threshold=THRESHOLD):
        super().__init__()
        self.data = data
        
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

    # Function to allow for possible multithread parrallelisation
    def process_part(self, i):
        ls,medoids,medlabels = self.get_ulscore_parralel()
        return {"value": ls,"i": i,"medoids": medoids,"membership":medlabels}
    
    # Leidenalg cluster function based on medoids memberships so far
    def getclust(self,mems):
        memlabels = np.unique(mems.flatten())
        D = np.zeros([mems.shape[0],mems.shape[0]])
        for m in memlabels:
            mem = (mems == m)*1.
            D += mem @ mem.T
        np.fill_diagonal(D,0)
        return np.array(la.find_partition(ig.Graph.Adjacency(D), la.ModularityVertexPartition).membership)
    
    # kMedoids for cluster finding on PWD matrix
    def kmedoids(self,X):
        if CLUSTERS*10>=self.data.shape[1]:
            clusters = CLUSTERS
        else:
            clusters = int(self.data.shape[1]/10)
        model = KMedoids(n_clusters=clusters,method='pam').fit(X)
        medoids = model.medoid_indices_
        medlabels = model.labels_
        return medoids,medlabels
    
    # Partially paralllelised feature scoring (TO DO: Vectorise feature by feature scoring for full max parr)
    def get_ulscore_parralel(self):
        n = self.n
        ones = np.ones((n,1))
        sample = np.random.choice(self.data.shape[0],n)
        Xsub = self.data[sample,:]
        Wsub = self.get_similaritymatrix(Xsub)
        Dsub = np.diagflat(np.sum(Wsub,axis=0))
        Lsub = Dsub - Wsub
        LSsub = np.zeros(Xsub.shape[1])
        for r in range(Xsub.shape[1]):#iterate over features
            fsubr = Xsub[:,r].reshape([-1,1])
            neighb_est = ((fsubr.T @ Dsub @ ones).item()/ (ones.T @ Dsub @ ones).item())*ones
            fsubr_est = (fsubr - neighb_est)#subtract nbh mean est of feature to centre feature vector
            d = (fsubr_est.T @ Dsub @ fsubr_est).item()
            num = (fsubr_est.T @ Lsub @ fsubr_est).item()
            if d > 0 and num>0:
                LSsub[r] = num/d
            elif num==0 and d>0:
                LSsub[r] = 0.
            else:
                LSsub[r] = 0.
        medoids,medlabels = self.kmedoids(Xsub.T)
        return LSsub,medoids,medlabels
    
    def get_similaritymatrix(self, X):
        """
        Optimized similarity matrix calculation for GUI classes.
        Uses np.partition for O(N) neighbor thresholding and 
        ensures graph symmetry.
        """
        t = self.t
        k = self.k
        n = X.shape[0]
        mode = self.mode

        # 1. Compute pairwise distances using the existing UI mode
        # Assuming self.getpwd is available in your class as defined previously
        D = self.getpwd(X, mode)
        
        # 2. Optimized k-nearest neighbor thresholding
        # k+1 because the first neighbor is always the point itself (dist=0)
        k_effective = min(k + 1, n - 1)
        
        # np.partition is faster than np.sort for finding the k-th element
        # Get the distance to the k-th neighbor for every row
        kn_dist = np.partition(D, k_effective, axis=1)[:, k_effective].reshape(-1, 1)

        # 3. Adjacency Logic
        # Compare each row's distances to its specific k-th neighbor threshold
        G = D <= kn_dist
        
        # Ensure Symmetry: If A is a neighbor of B OR B is a neighbor of A
        # This prevents 'one-way' edges which can lead to lead to non-physical LS results
        G = np.logical_or(G, G.T)
        
        # Remove self-loops (diagonal)
        np.fill_diagonal(G, 0)

        # 4. Weighting
        W = np.zeros([n, n])
        if mode == 'heat':
            W[G] = np.exp(-D[G]**2 / (2 * t**2))
        else:
            # Cosine/Standard mode: Similarity = 1 - Distance
            # Using abs(1-D) ensures positive weights even with slight float errors
            W[G] = np.abs(1 - D[G])
            
        return W
    
    def getpwd(self,X,mode='cosine'):
        if mode == 'heat':#heat kernel based pwd (euclidean)
            D = pairwise_distances(X)
        if mode == 'cosine':#cosine pwd
            D = pairwise_distances(X,metric='cosine')
        return D

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FlowFI: Flow cytometry Feature Importance")
        
        # Set window icon. 
        # To use your own logo, create a 'logo.png' file (64x64 pixels is a good size)
        # and place it in the same directory as this script.
        logo_path = 'logo.png'
        if os.path.exists(logo_path):
            self.setWindowIcon(QIcon(logo_path))

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

        # TAB-1 LAYOUT: ANALYSIS


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
        self.operations_performed = 0
        self.current_channel = None
        self.current_image_array = None
        self.processed_image = None
        self.agg_operation = None
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
        # self.update_timer.timeout.connect(self.update_progress)
    
    #TAB-2 Image Analysis
    
    # Select root folder for tiff images
    def browse_for_root(self):
        directory = QFileDialog.getExistingDirectory(self, "Select New Root Directory",
                                                   self.root_path_input.text(),
                                                   QFileDialog.ShowDirsOnly)
        if directory:
            self.root_path_input.setText(directory)
            self.set_tree_root()

    # Set file tree for exploration
    def set_tree_root(self):
        root_path = self.root_path_input.text()
        if QDir(root_path).exists():
            self.model.setRootPath(root_path)
            self.tree.setRootIndex(self.model.index(root_path))
        else:
            print(f"Error: Root path '{root_path}' does not exist.")
        
    # Loads image from suitable tiff file
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
                    self.parameters_menu.setEnabled(True)

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
    
    # General update code including preprocessing operations
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
    
    # Image Update Panel when new image selected
    def update_left_image_label(self):
        if self.current_q_image is not None:
            pixmap = QPixmap.fromImage(self.current_q_image)
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.clear()
    
    # Reset Preprocessing Operations
    def reset_operations(self):
        self.operation_history = []
        self.operations_performed = 0
        self.processed_image = self.current_image_array[self.current_channel]
        self.terminal.add_operation('Reset Preprocessing Operations')
        self.process_image()
    
    # Normalise to 8-bit range and/or 8-bit type 
    def norm(self,array,eightbit=True):
        array -= np.min(array)
        max_val = np.max(array)
        if max_val > 0:
            array /= max_val
        array *= 255
        if eightbit:
            array = np.round(array).astype('uint8')
        return array
    
    # Enable the selected aggregation action
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


    # Do aggregation for given image stack
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

    # Do silent aggregation over a give image (for largescale compiling)
    def do_aggregation_silent(self,image):
        score = np.nan
        
        # Prepare optional global mask if it exists
        global_mask_channel = self.agg_channels.get('Global Mask (Optional)')
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

    # Preprocess images according to selected operations
    def process_image(self):
        self.perform_operations()
        height, width = self.processed_image.shape
        pimage = self.norm(self.processed_image).data
        self.processed_q_image = QImage(pimage, width, height, width, QImage.Format_Grayscale8)
        self.update_right_image_label()

    # Show preprocessed image in imag panel right
    def update_right_image_label(self):
        if self.processed_q_image is not None:
            pixmap = QPixmap.fromImage(self.processed_q_image)
            self.processed_image_label.setPixmap(pixmap)
        else:
            self.processed_image_label.clear()
    
    # Perform operations and aggregation on current image
    def perform_operations(self):
        nops = len(self.operation_history)
        for i in range(self.operations_performed,nops):
            self.do_operation(i)
        self.operations_performed = nops
        if self.agg_operation is not None:
            self.do_aggregation()

    # Perform given preprocessing operation    
    def do_operation(self,opindex):
        operation = self.operation_history[opindex]

        if operation[0]=='gauss':
            self.processed_image = self.gaussblur(self.processed_image, float(operation[1]))  # Call self.gauss
            self.terminal.add_operation(f'Gaussian Blur: {np.round(float(operation[1]),2)} Channel: {self.current_channel+1}')
        elif operation[0]=='mask':
            self.processed_image = self.get_mask(self.processed_image.astype(float)).astype(float)
            self.terminal.add_operation(f'Mask Channel: {self.current_channel+1}')
        elif operation[0]=='label':
            self.processed_image = self.get_label(self.processed_image.astype(int)).astype(float)
            self.terminal.add_operation(f'Label Channel: {self.current_channel+1}')
        elif operation[0]=='segment':
            self.processed_image = self.get_segment(self.processed_image.astype(float)).astype(float)
            self.terminal.add_operation(f'Segment Channel: {self.current_channel+1}')
        elif operation[0]=='preset1':
            # preset1_preprocess returns the processed image and a threshold mask. We only need the image here.
            self.processed_image, _ = self.preset1_preprocess(self.processed_image)
            self.terminal.add_operation(f'Preset 1 Preprocess Channel: {self.current_channel+1}')
        elif operation[0] == 'crop':
            top, bottom, left, right = operation[1]
            self.processed_image = self.crop_image(self.processed_image, top, bottom, left, right)
            self.terminal.add_operation(f'Crop: T={top}, B={bottom}, L={left}, R={right} on Channel: {self.current_channel+1}')
        elif operation[0] == 'rescale':
            scale_x, scale_y, interpolation_method = operation[1]
            self.processed_image = self.rescale_image(self.processed_image, scale_x, scale_y, interpolation_method)
            self.terminal.add_operation(f'Rescale: X={scale_x}, Y={scale_y} on Channel: {self.current_channel+1}')

    def crop_image(self, image, top, bottom, left, right):
        h, w = image.shape
        return image[top:h-bottom, left:w-right]
    
    # Perform operation without terminal reporting for given image (for multi-image processing)
    def do_operation_silent(self,index,image):     
        operation = self.operation_history[index]

        if operation[0]=='gauss':
            return self.gaussblur(image, float(operation[1]))  # Call self.gauss
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




    # S8 image preprocessing presets
    def preset1_preprocess(self,img):
        img = img[TOPCROP:img.shape[0]-BOTTOMCROP,LEFTCROP:img.shape[1]-RIGHTCROP]

        width = int(img.shape[1] * 1)
        height = int(img.shape[0] * .25)
        dsize = (width, height)
        imgdown = cv2.resize(img,dsize,interpolation=cv2.INTER_AREA)

        imgdown = cv2.GaussianBlur(imgdown, AKERNEL, sigmaX=SIG_X,sigmaY=SIG_Y)


        current_height, current_width = imgdown.shape
        new_height =  int(current_height/PAR)
        dsize = (current_width, new_height)

        # Use Lanczos interpolation for high quality resizing
        imgup = cv2.resize(imgdown, dsize, interpolation=cv2.INTER_LANCZOS4)


        imgblur = cv2.GaussianBlur(imgup, KERNEL, sigmaX=SIG2, sigmaY=SIG2,borderType=cv2.BORDER_CONSTANT)

        # Preset specific checks
        hmask = imgup>NOISETHRESHOLD
        sumsig = np.sum(hmask)
        if sumsig>=SMALL:   
            th = threshold_otsu(imgblur[hmask])
            imgth = imgblur>=th
        else:
            imgth = np.zeros(imgup.shape,dtype=bool)
        

        imgth = binary_closing(binary_opening(imgth,footprint=FOOTPRINT),footprint=SQUARE)
        imgth = remove_small_objects(imgth,SMALL,connectivity=2)
        imgth = binary_fill_holes(imgth,structure=np.ones((3,3)))
        
        # More preset specific checks
        boundary_pixels = (np.sum(imgth[0, :]) + np.sum(imgth[-1, :]) +
                        np.sum(imgth[:, 0]) + np.sum(imgth[:, -1]) -
                        imgth[0, 0] - imgth[0, -1] - imgth[-1, 0] - imgth[-1, -1])
        boundary_fraction = boundary_pixels/(2*(current_width+new_height-2))
        if np.sum(imgth)>(imgth.shape[0]*imgth.shape[1])/4 and boundary_fraction>0.05:
            imgth *= False
        

        # imghigh = imgup.copy()
        # imghigh[~imgth] = 0 

        return imgup,imgth

            
    # def resizeEvent(self, event):
    #     if self.current_image_array is not None:
    #         self.update_left_image_label()
    #         # current_channel_index = self.channel_slider.value()
    #         # self.processed_image = self.current_image_array[current_channel_index]
    #         self.process_image()
    #     super().resizeEvent(event)
    
    
    # Gaussian operation dialogue
    def open_gauss(self):
        dialog = GaussDialog(self)  # Pass self as parent
        if dialog.exec_() == QDialog.Accepted:
            sigma = dialog.get_sigma()
            self.operation_history += [['gauss',sigma]]
            self.terminal.add_operation(f'Gaussian Blur: {np.round(sigma,2)} Channel: {self.current_channel+1}')
            self.process_image()
        else:
            print("Dialog cancelled.")

    def open_crop_dialog(self):
        dialog = CropDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            crop_values = dialog.get_values()
            self.operation_history.append(['crop', crop_values])
            self.process_image()

    def open_rescale_dialog(self):
        dialog = RescaleDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            scale_x, scale_y, interpolation_method = dialog.get_values()
            self.operation_history.append(['rescale', (scale_x, scale_y, interpolation_method)])
            self.process_image()

    def rescale_image(self, image, scale_x, scale_y, interpolation_method):
        h, w = image.shape
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        dsize = (new_w, new_h)
        inter_map = {'Nearest': cv2.INTER_NEAREST, 'Linear': cv2.INTER_LINEAR, 'Area': cv2.INTER_AREA, 'Cubic': cv2.INTER_CUBIC, 'Lanczos4': cv2.INTER_LANCZOS4}
        return cv2.resize(image, dsize, interpolation=inter_map.get(interpolation_method, cv2.INTER_LINEAR))

    # Mask operation applied to current image with reporting
    def do_mask(self):
        self.operation_history += [['mask']]
        self.terminal.add_operation(f'Mask Channel: {self.current_channel+1}')
        self.process_image()

    def do_segment(self):
        self.operation_history += [['segment']]
        self.terminal.add_operation(f'Segment Channel: {self.current_channel+1}')
        self.process_image()
    
    def do_label(self):
        self.operation_history += [['label']]
        self.terminal.add_operation(f'Label Channel: {self.current_channel+1}')
        self.process_image()
    
    # Preset operation
    def do_preset1(self):
        self.operation_history += [['preset1']]
        self.terminal.add_operation(f'Preset 1 Preprocess Channel: {self.current_channel+1}')
        self.process_image()

    def undo_last_operation(self):
        """Removes the last operation from the history and re-processes the image."""
        if not self.operation_history:
            self.terminal.add_operation("No operations to undo.")
            return

        last_op = self.operation_history.pop()
        self.terminal.add_operation(f"Undo: {last_op[0]}")

        # Reset and re-process from the original image for the current channel
        self.operations_performed = 0
        self.processed_image = self.current_image_array[self.current_channel].copy()
        self.process_image()

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

    # Wrapper: Export CSV from image set
    def do_export_csv(self):
        self.do_process_images(mode='csv')

    # Wrapper: fcs from image (and selected fcs)
    def do_export_fcs(self):
        self.do_process_images(mode='fcs')
    
    
    # Write csv/fcs from images
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
        ppath = None
        abs_ppath = None
        if mode == 'image':
            ppath = os.path.join(folder_path, 'processed')
            abs_ppath = os.path.abspath(ppath)

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

        # Create and configure the progress dialog
        progress_dialog = QProgressDialog("Processing images...", "Cancel", 0, total_files, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setWindowTitle("Bulk Processing")

        transform_log = False
        if mode == 'image':
            # ppath is already defined above
            os.makedirs(ppath,exist_ok=True)
        elif mode=='csv' or mode=='fcs':
            vals = []
            if mode == 'csv':
                vfile = os.path.join(folder_path,'new_parameter.csv')
            elif mode=='fcs':
                try:
                    old_fcsfile = self.get_fcs_files(folder_path)[0]
                    base, ext = os.path.splitext(old_fcsfile)
                    new_fcsfile = base + '_new' + ext
                    reply = QMessageBox.question(self, 'Log Transformation', 
                                                "Apply 10^x transformation to the new parameter for loglog visualisation?",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    transform_log = (reply == QMessageBox.Yes)
                except Exception as e:
                    QMessageBox.critical(self, "Error", "No suitable fcs file found in directory")
                    return

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

        if mode=='image':
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} TIFF files into: {ppath}")
        elif mode == 'csv':
            self.param_to_csv(vals,vfile)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {vfile}")
        elif mode == 'fcs':
            self.param_to_fcs(vals,old_fcsfile,new_fcsfile, transform=transform_log)
            self.terminal.add_operation("Processing Complete") 
            self.terminal.add_operation(f"Processed {processed_count} parameter values in: {new_fcsfile}")

    # Process image explicitly for export
    def process_image_export(self,image):
        for i,op in enumerate(self.operation_history):
            image = self.do_operation_silent(i,image)
        return image
    
    # Write new param to csv export
    def param_to_csv(self,vals,vfile):
        vals = np.array(vals).reshape(-1,1)
        with open (vfile,'w') as f:
            wtr = csv.writer(f)
            wtr.writerows(vals)
    
    # Process Param for common fcs formats
    def param_to_fcs(self,vals,ofcs,nfcs, transform=True):
        vals = np.array(vals).reshape(-1,1)
        if transform:
            vals = 10**vals
        fcs,metadata = self.load_fcs(ofcs)
        self.add_param(fcs,nfcs,metadata,vals)
    
    def load_fcs(self,fcsfile):
        fcdata = flowio.FlowData(fcsfile)
        fcsample = fk.Sample(fcsfile)
        metadata = fcsample.metadata
        return fcdata,metadata
    
    def add_param(self,fcdata,nfcs,metadata,vals,pname='new_param'):
        numc = fcdata.channel_count
        channels = [fcdata.channels[k]['PnN'] for k in fcdata.channels.keys()]
        events = np.reshape(fcdata.events,(-1,numc))
        channels.append(pname)
        events = np.hstack([events,vals])
        print(events.shape)
        events = events.flatten()
        flowio.create_fcs(open(nfcs,'wb'),events,channels,opt_channel_names=channels,metadata_dict=metadata)



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

    
    # Image functions

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
    
    # Quantification functions

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

            # --- Step 1: Calculate the Centroid for the Polar Coordinate System ---
            y_coords_2, x_coords_2 = np.where(mask2)
            img2_intensities = img2[y_coords_2, x_coords_2]
            
            reference_centroid_y = np.average(y_coords_2, weights=img2_intensities)
            reference_centroid_x = np.average(x_coords_2, weights=img2_intensities)
        else:
            y_coords, x_coords = np.where(mask)
            reference_centroid_y = np.average(y_coords)
            reference_centroid_x = np.average(x_coords)

        # --- Step 2: Define Sectors and Collect Data (Vectorized) ---
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

        # --- Step 3: Normalize and Create Vectors (Vectorized) ---
        # Handle sectors with no pixels to avoid division by zero.
        avg_intensity_per_sector = np.divide(sector_intensities, sector_areas, out=np.zeros_like(sector_intensities, dtype=np.float64), where=sector_areas != 0)
        
        # Calculate the magnitude for each vector based on normalized intensity difference.
        magnitudes = avg_intensity_per_sector - avg_intensity_total
        
        # Angles for each sector's vector direction.
        sector_angles = np.linspace(0, 2 * np.pi, num_sectors, endpoint=False)

        # Calculate resultant vector components in a single operation.
        resultant_vector_x = np.sum(magnitudes * np.cos(sector_angles))
        resultant_vector_y = np.sum(magnitudes * np.sin(sector_angles))

        # --- Step 4: Calculate the Final Parameter (Magnitude of Resultant Vector) ---
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

            # --- Step 1: Calculate the Centroid for the Polar Coordinate System ---
            y_coords_2, x_coords_2 = np.where(mask2)
            img2_intensities = img2[y_coords_2, x_coords_2]
            
            reference_centroid_y = np.average(y_coords_2, weights=img2_intensities)
            reference_centroid_x = np.average(x_coords_2, weights=img2_intensities)
        else:
            y_coords, x_coords = np.where(mask)
            reference_centroid_y = np.average(y_coords)
            reference_centroid_x = np.average(x_coords)

        # --- Step 2: Define Sectors and Collect Data (Vectorized) ---
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

        # --- Step 3: Calculate the Probability Distribution ---
        total_intensity = np.sum(sector_intensities)
        if total_intensity == 0:
            return 0.
            
        probabilities = sector_intensities / total_intensity

        # --- Step 4: Calculate Shannon Entropy ---
        # The term will be 0 when probability is 0, so we can ignore those entries.
        probabilities = probabilities[probabilities > 0]
        
        # Shannon entropy formula: -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # --- Step 5: Normalize Entropy to the number of bins ---
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
            # --- Step 1: Calculate the Centroid for the Polar Coordinate System ---
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

        # --- Step 3: Create the Radial Intensity Profile ---
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

        # --- Step 4: Calculate the Third Moment (Skewness) of the Profile ---
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
        return gaussian(image,sigma,mode='wrap')
    
    # MENU FUNCTIONS

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

        presets_submenu = QMenu('&Presets', self)
        preset1_action = QAction('&Preset - OFDM', self)
        preset1_action.triggered.connect(self.do_preset1)
        presets_submenu.addAction(preset1_action)

        # Add Undo action
        undo_action = QAction('&Undo Last Operation', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.triggered.connect(self.undo_last_operation)

        reset_action = QAction('&Reset Preprocessing', self)
        reset_action.triggered.connect(self.reset_operations)

        save_image_action = QAction('&Save Single Image (.tiff)', self)
        save_image_action.triggered.connect(self.save_image)

        batch_process_action = QAction('&Batch Process Folder', self)
        batch_process_action.triggered.connect(self.do_batch_process_images)


        self.preprocessing_menu.addMenu(presets_submenu)
        self.preprocessing_menu.addMenu(filters_submenu)
        self.preprocessing_menu.addMenu(manipulation_submenu)
        self.preprocessing_menu.addMenu(segmentation_submenu)

        # --- Quantify Submenu ---
        self.quantify_menu = menu_bar.addMenu('&Quantify')
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(reset_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(undo_action)
        self.preprocessing_menu.addSeparator()
        self.preprocessing_menu.addAction(save_image_action)
        self.preprocessing_menu.addAction(batch_process_action)

        aggregation_submenu = QMenu('&Aggregation', self)
        geometry_submenu = QMenu('&Geometry', self)
        self.quantify_menu.addMenu(aggregation_submenu)
        self.quantify_menu.addMenu(geometry_submenu)

        # Aggregation Actions
        self.count_action = QAction("Count (unique)", self, checkable=True)
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

        export_to_fcs = QAction('Export to FCS',self)
        export_to_fcs.triggered.connect(self.do_export_fcs)
        export_to_csv = QAction('Export to CSV',self)
        export_to_csv.triggered.connect(self.do_export_csv)
        self.parameters_menu.addActions([export_to_fcs,export_to_csv])

        self.preprocessing_menu.setEnabled(False)
        self.quantify_menu.setEnabled(False)
        self.parameters_menu.setEnabled(False)

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
        preferences_action = QAction('Preferences...', self)
        preferences_action.triggered.connect(self.open_refine_preferences)
        refine_menu.addAction(preferences_action)

        # Help menu (last)
        help_menu = menu_bar.addMenu('&Help')
        readme_action = QAction('README', self)
        readme_action.triggered.connect(self.show_readme)
        help_menu.addAction(readme_action)

    # TAB-1 FUNCTIONS

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
        if self.finalcluster:
            self.load_output()
        else:
            QMessageBox.information(self, "Error", "No complete results to compare to yet.")
    
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
                for f in fieldnames:
                    if f!='feature':
                        loaded_result[f] = np.array(loaded_result[f]).astype('float')
                    else:
                        loaded_result[f] = np.array(loaded_result[f])
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
        if not hasattr(self,'data'):
            QMessageBox.warning(self, "Warning", "No features found in the FCS file.")
            return
        
        self.execute_button.setEnabled(False)
        self.start_time = time.time()
        self.output_layout.removeWidget(self.output_widget)
        self.output_widget = QWidget()
        self.output_layout = QVBoxLayout()
        self.output_widget.setLayout(self.output_layout)
        self.output_panel.setWidget(self.output_widget)


        self.progress_bar.setValue(0)

        self.worker = WorkerThread(self.data, boots=self.boots_param, bootsize=self.bootsize_param, 
                                   conv_check=self.convergence_check, conv_threshold=self.convergence_threshold)
        self.boots = self.worker.boots
        self.feature_averages = np.zeros((self.data.shape[1],self.boots))
        self.calculated = np.zeros((self.boots))
        self.medoids = np.zeros((self.data.shape[1],self.boots))
        self.memberships = np.zeros((self.data.shape[1],self.boots))
        self.finalcluster = False

        
        # self.worker.progress_update.connect(self.update_progress)
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
                df = df.select_dtypes(include=[np.number])
                self.columns = np.array(df.columns)
                self.data = df.values
            else:
                fcdata = flowio.FlowData(self.filepath)
                self.columns = np.array([fcdata.channels[c]['PnN'] for c in fcdata.channels])
                self.data = np.reshape(fcdata.events,[-1,fcdata.channel_count])
            self.cleandata() 
            # if SIM:
            #     n_features = self.data.shape[1]
            #     ninf = np.max([int(n_features*.01),1])
            #     nred = np.max([int(n_features*.01),1])
            #     cshape = int(self.data.shape[0]*.6)
            #     self.data,_ = make_classification(self.data.shape[0], n_features,n_repeated=0,weights=[.1],n_informative=ninf,n_redundant=nred,n_clusters_per_class=1,shuffle=False)
            #     ncontam = n_features - ninf - nred
            #     contam,_ = make_classification(cshape, ncontam,n_repeated=0,weights=[.5],n_informative=ncontam,n_redundant=0,n_clusters_per_class=1,shuffle=False)
            #     self.data[:cshape,-ncontam:] += contam
            #     self.meaningful = np.zeros(self.data.shape[1])
            #     self.meaningful[:(ninf+nred)] = 1

            
        except Exception as e:
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load features from FCS file: {e}")

    # min-max normalization for values
    def NormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    # Prepare fcs data for analysis
    def cleandata(self,norm=True): 
        included = [i for i,c in enumerate(self.columns) if c not in excludedcols]
        self.columns = self.columns[included]
        self.data = self.data[:,included]
        
        # self.data,uind = np.unique(self.data,axis=1,return_index = True)
        # self.columns = self.columns[uind]
        # self.data = self.data[:,uind]

        included = np.var(self.data,axis=0)>1e-8
        nondiverse = [i for i in range(self.data.shape[1]) if len(np.unique(self.data[:,i]).flatten())<10]
        included[nondiverse] = 0
        self.data = self.data[:,included]
        self.columns = self.columns[included]

        UVpattern = r'^UV\d+.*'
        Vpattern = r'^V\d+.*'
        Bpattern = r'^B\d+.*'
        YGpattern = r'^YG\d+.*'
        Rpattern = r'^R\d+.*'
        ImgBpattern = r'^ImgB\d+.*'
        Imagingpattern = r'.*\(Imaging\).*|.*Axis.*|.*Mass.*|.*Intensity.*|.*Moment.*|.*Size.*|.*Diffusivity.*|.*Eccentricity.*'

        patterns = [UVpattern,Vpattern,Bpattern,YGpattern,Rpattern,ImgBpattern,Imagingpattern]
        self.patternmatches = np.ones(len(self.columns))*len(patterns)
        self.patternmatches = self.patternmatches.astype(int)

        for k,p in enumerate(patterns):
                matches = [i for i,c in enumerate(self.columns) if re.match(p,c)]
                self.patternmatches[matches] = k
        sort = np.argsort(self.patternmatches)
        self.patternmatches = self.patternmatches[sort]
        self.columns = self.columns[sort]
        self.data = self.data[:,sort]

        self.fcolors = np.array([self.colors[c] for c in self.patternmatches])
        self.flabels = np.array([self.ftypes[c] for c in self.patternmatches])
        
        self.filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types]
        self.patternmatches = self.patternmatches[self.filter]
        self.columns = self.columns[self.filter]
        self.data = self.data[:,self.filter]
        self.flabels = self.flabels[self.filter]
        self.fcolors = self.fcolors[self.filter]

        if norm:
            self.data = StandardScaler().fit_transform(self.data)
    
    def add_result(self, result):
        value = result['value']
        i =  result['i']
        self.medoids[list(result['medoids'].astype(int)),i] += 1
        self.memberships[:,i] = result['membership']
        self.feature_averages[:,i] = value
        self.calculated[i] = 1
        non0 = self.calculated>0
        imp_calculated = self.feature_averages[:,non0]
        mean_value = np.mean(imp_calculated,axis=1).flatten()
        mdds = np.sum(self.medoids[:,non0],axis=1).flatten()
        self.result = {'ls': mean_value,'i': i,'medoids': mdds,'membership':result['membership']}

    def color_name_to_rgba(self,color_name):
        try:
            rgba = mcolors.to_rgba(color_name)
            return rgba
        except ValueError:
            return (0,0,0,0)

    def update_display(self):
        self.selected_feature_types = [key for key, checkbox in self.feature_checkboxes.items() if checkbox.isChecked()]

        if hasattr(self, 'result'):
            if self.centrality_checkbox.isChecked():
                central_features = [i for i,m in enumerate(self.result['medoids']) if m>0]
                filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types and i in central_features]
            else:
                filter = [i for i,f in enumerate(self.flabels) if f in self.selected_feature_types]
            self.output_layout.removeWidget(self.output_widget)
            self.output_widget = QWidget()
            self.output_layout = QVBoxLayout()
            self.output_widget.setLayout(self.output_layout)
            self.output_panel.setWidget(self.output_widget)

            mean_value = 1-self.NormalizeData(self.result['ls'])[filter]
            
            loaded_final = self.finalcluster and hasattr(self,"loaded_result")
            if loaded_final:
                ffeatures = self.columns[filter]
                loaded_ffeatures = self.loaded_result['feature']

                loaded_orderedimp = np.zeros(len(ffeatures))
                orderedimp = np.zeros(len(ffeatures))
                for i in range(len(ffeatures)):
                    if ffeatures[i] in loaded_ffeatures:
                        ind = int(np.where(ffeatures[i]==loaded_ffeatures)[0][0])
                        loaded_orderedimp[i] = self.loaded_result['ls'][ind]
                        orderedimp[i] = self.result['ls'][filter][i]
                    else:
                        orderedimp[i] = -1
                        loaded_orderedimp[i] = -1
                    
                orderedimp[orderedimp>=0] = orderedimp[orderedimp>=0].argsort().argsort()
                loaded_orderedimp[loaded_orderedimp>=0] = loaded_orderedimp[loaded_orderedimp>=0].argsort().argsort()
                rankdiffs = np.zeros(len(ffeatures))
                rankdiffs[orderedimp>=0] = orderedimp[orderedimp>=0]-loaded_orderedimp[loaded_orderedimp>=0]
                rankdiffs[orderedimp==-1] = np.nan
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
                else:#If nothing else works (i.e. clustering not ready) then sort by Importance
                    sort = np.argsort(second)
                    sorting = False
            if sorting:
                sort = np.lexsort([second,first])
                sorting = False

            colors = self.fcolors[filter][sort]
            mean_value = mean_value[sort]
            medoids = self.result['medoids'][filter][sort]
            topmeds = np.where(medoids>0)[0]
            texts = self.columns[filter][sort]
            labels = self.flabels[filter][sort]

            if loaded_final:
                rankdiffs = rankdiffs[sort]

            if self.worker.early:
                self.worker.progress = self.boots    
            prog = int(100*self.worker.progress/self.boots)
            self.progress_bar.setValue(prog)
            if self.finalcluster:
                membership = self.membership[filter][sort]
                # membership = self.membership.astype(int)
                memcolors = [self.clustercolors[m] for m in membership]

            has_ci = 'LowCI' in self.result and 'UpperCI' in self.result
            if has_ci:
                low_ci = self.result['LowCI'][filter][sort]
                upper_ci = self.result['UpperCI'][filter][sort]

            for i in range(len(filter)):
                # Create a layout for each entry
                entry_layout = QHBoxLayout()
                text = texts[i]
                if loaded_final:
                    if -rankdiffs[i]>0:
                        text += ' (+' + str(int(-rankdiffs[i])) + ')'
                    elif -rankdiffs[i]<=0:
                        text += ' (' + str(int(-rankdiffs[i])) + ')'
                # Create and style the label for the colored text
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

                # Create and style the bar for the value
                stroke = colors[i]
                if self.finalcluster:
                    stroke = memcolors[i]
                
                l_ci = low_ci[i] if has_ci else None
                u_ci = upper_ci[i] if has_ci else None
                bar = BarWidget(mean_value[i], colors[i], l_ci, u_ci, stroke_color=stroke)
                entry_layout.addWidget(bar)

                # Create a container widget for the entry layout
                entry_widget = QWidget()
                entry_widget.setLayout(entry_layout)
                # Add the entry widget to the output layout
                self.output_layout.addWidget(entry_widget)

            self.output_widget.adjustSize()
            QApplication.processEvents()

    def show_processing_time(self):
        text = "Processing time: " + str(self.total_time) + 's'
        QMessageBox.information(self, "Processing Time", text)

    def consensusclustering_final(self):
        self.membership = self.worker.getclust(self.memberships)
        self.finalcluster = True
        if EVAL == True:
            self.end_time = time.time()
            self.total_time = np.round(self.end_time - self.start_time,2)
            self.show_processing_time()
        self.execute_button.setEnabled(True)

    # def logit_transform_ci(self,bootstrap_replicates, alpha=0.05, n_sample=None):
    #     """
    #     Calculates a confidence interval for a proportion/bounded estimate (0, 1)
    #     using the logit transformation and the bootstrap percentile method.

    #     Parameters:
    #     - bootstrap_replicates (array-like): The array of bootstrap estimates (0 <= theta* <= 1).
    #     - alpha (float): The significance level (e.g., 0.05 for a 95% CI).
    #     - n_sample (int, optional): The original sample size. Used for continuity 
    #     correction if bootstrap estimates hit 0 or 1. If None, a small fixed 
    #     epsilon is used.

    #     Returns:
    #     - tuple: (lower_bound, upper_bound) of the CI on the original scale.
    #     """
        
    #     # 1. Define the transformation functions
    #     def logit(p):
    #         """Logit: ln(p / (1-p))"""
    #         return np.log(p / (1 - p))

    #     def inv_logit(l):
    #         """Inverse Logit (Logistic function): 1 / (1 + e^(-l))"""
    #         return 1 / (1 + np.exp(-l))

    #     # Determine correction epsilon for 0 or 1 estimates in the bootstrap sample
    #     if n_sample is not None:
    #         # Standard continuity correction: use 1/(2n)
    #         epsilon = 1 / (2 * n_sample)
    #     else:
    #         # Fallback to a small fixed value if sample size is unknown
    #         epsilon = 1e-6

    #     # Convert to NumPy array
    #     theta_star = np.asarray(bootstrap_replicates)
        
    #     # 2. Apply Boundary Correction (Handling 0 and 1)
    #     # The correction pushes 0s and 1s slightly away from the boundary
    #     # so the logit can be calculated (e.g., 1 becomes 1-epsilon).
    #     theta_star_corrected = np.clip(theta_star, epsilon, 1 - epsilon)

    #     # 3. Transform to Logit Scale
    #     psi_star = logit(theta_star_corrected)

    #     # 4. Calculate Percentile CI on Transformed Scale
        
    #     # Calculate the desired quantiles for the CI (e.g., 2.5% and 97.5%)
    #     lower_quantile = alpha / 2
    #     upper_quantile = 1 - (alpha / 2)

    #     ci_psi_lower = np.quantile(psi_star, lower_quantile)
    #     ci_psi_upper = np.quantile(psi_star, upper_quantile)

    #     # 5. Inverse-Transform back to Original Scale
    #     ci_theta_lower = inv_logit(ci_psi_lower)
    #     ci_theta_upper = inv_logit(ci_psi_upper)

    #     return ci_theta_lower, ci_theta_upper

    def finalize_results(self):
        if self.worker.early:
            self.memberships = self.memberships[:,self.calculated>0]
            self.feature_averages = self.feature_averages[:,self.calculated>0]
            self.medoids = self.medoids[:,self.calculated>0]
        self.output_widget.adjustSize()
        self.consensusclustering_final()
        self.update_display()
        self.result['Relative Importance'] = 1 - self.NormalizeData(self.result['ls'])
        self.calculate_cis()

        self.result['Centrality'] = self.NormalizeData(self.result['medoids'])
        self.result['Membership'] = self.membership
        QMessageBox.information(self, "Information", "Processing complete!")
        self.update_timer.stop()  # Stop the update timer

    def calculate_cis(self):
        if self.calc_ci_action.isChecked() and hasattr(self, 'result') and hasattr(self, 'feature_averages'):
            negls = -self.result['ls']
            neglsmin = np.min(negls)
            neglsmax = np.max(negls)
            s = self.feature_averages.shape[1]
            numf = self.feature_averages.shape[0]
            
            sample_ris = np.zeros([self.ci_boots,numf])
            sample_negls = np.zeros([self.ci_boots,numf])
            for i in range(self.ci_boots):
                temp_ls = self.feature_averages[:,np.random.choice(s,s,replace=True)]
                temp_mean = np.mean(temp_ls,axis=1)
                sample_ris[i,:] = 1-self.NormalizeData(temp_mean)
                sample_negls[i,:] = -temp_mean
            
            neglcis = np.array([np.percentile(sample_negls[:,m],self.ci_alpha/2) for m in np.arange(numf)])
            negucis = np.array([np.percentile(sample_negls[:,m],100-self.ci_alpha/2) for m in np.arange(numf)])
            
            lb_violation = [neglsmin>=neglcis[m] for m in range(numf)]
            ub_violation = [neglsmax<=negucis[m] for m in range(numf)]

            lcis = np.array([np.percentile(sample_ris[:,m],self.ci_alpha/2) for m in np.arange(numf)])
            ucis = np.array([np.percentile(sample_ris[:,m],100-self.ci_alpha/2) for m in np.arange(numf)])

            lcis[lb_violation] = 0.
            ucis[ub_violation] = 1.

            # # lcis = np.zeros(numf)
            # # ucis = np.zeros(numf)
            # # print(s)
            # # for m in range(numf):
            # #     lcis[m],ucis[m] = self.logit_transform_ci(sample_ris[:,m],alpha=alpha*.01,n_sample = s)

            self.result['LowCI'] = lcis
            self.result['UpperCI'] = ucis
            self.update_display()

    def save_output(self):
        if not self.result:
            QMessageBox.warning(self, "Warning", "There is no output to save.")
            return

        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Output", "", "CSV Files (*.csv)", options=options)
        if filepath:
            try:
                with open(filepath, 'w', newline='') as csvfile:
                    if not hasattr(self,"loaded_result"):
                        fieldnames = ['feature','ri', 'ls','membership','centrality']
                        if 'LowCI' in self.result:
                            fieldnames += ['LowCI', 'UpperCI']
                    else:
                        fieldnames = ['feature','ri', 'ls','membership','centrality',"comparison"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    result = self.result['ls']
                    impresult = self.result['Relative Importance']
                    columns = self.columns
                    memb = self.result['Membership']
                    centrality = self.result['Centrality']
                    if not hasattr(self,"loaded_result"):
                        for i in range(len(result)):
                            row_data = {'feature': columns[i], 'ri': impresult[i], 'ls': result[i],'membership':memb[i],'centrality': centrality[i]}
                            if 'LowCI' in self.result:
                                row_data['LowCI'] = self.result['LowCI'][i]
                                row_data['UpperCI'] = self.result['UpperCI'][i]
                            writer.writerow(row_data)
                    else:
                        comparison = self.result['Comparison']
                        for i in range(len(result)):
                            writer.writerow({'feature': columns[i], 'ri': impresult[i], 'ls': result[i],'membership':memb[i],'centrality': centrality[i],'comparison': comparison[i]})
                QMessageBox.information(self, "Success", "Output successfully saved to CSV file.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save output to CSV file: {e}")

    def show_readme(self):
        dialog = HelpDialog(self)
        dialog.exec_()

    def open_refine_preferences(self):
        dataset_size = None
        if hasattr(self, 'data') and self.data is not None:
            dataset_size = self.data.shape[0]

        dialog = RefinePreferencesDialog(self, default_boots=self.boots_param, default_bootsize=self.bootsize_param, dataset_size=dataset_size,
                                         default_conv_check=self.convergence_check, default_conv_threshold=self.convergence_threshold)
        if dialog.exec_() == QDialog.Accepted:
            self.boots_param, self.bootsize_param, self.convergence_check, self.convergence_threshold = dialog.get_values()
            self.terminal.add_operation(f"Refine parameters updated: BOOT={self.boots_param}, BOOTSIZE={self.bootsize_param}, ConvCheck={self.convergence_check}, Threshold={self.convergence_threshold}")

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

class GaussDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Blur Sigma")  # More specific title
        self.sigma = 0.0

        layout = QVBoxLayout(self)

        # Label and Line Edit for Sigma
        sigma_layout = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        self.sigma_edit = QLineEdit("2.0")
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
        cancel_button.clicked.connect(self.reject)  # Connect Cancel

        self.setLayout(layout)

    def get_sigma(self):
        return self.sigma

    def accept(self):
        try:
            self.sigma = float(self.sigma_edit.text())
            if self.sigma <= 0:
                QMessageBox.warning(self, "Invalid Input", "Sigma must be greater than zero.", QMessageBox.Ok)
                return  # Stay in dialog if input is invalid
            super().accept()  # Close dialog and set result to Accepted
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for sigma.", QMessageBox.Ok)
            return  # Stay in dialog if input is not a number

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
        self.setWindowTitle("FlowFI Help")
        self.setGeometry(200, 200, 700, 550)

        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)

        # --- Create Tabs ---
        tabs.addTab(self.create_text_widget(self.get_workflow_text()), "Workflow")
        tabs.addTab(self.create_text_widget(self.get_design_text()), "Design Tab")
        tabs.addTab(self.create_text_widget(self.get_refine_text()), "Refine Tab")

        main_layout.addWidget(tabs)

    def create_text_widget(self, html_content):
        """Creates a read-only QTextEdit widget with the given HTML content."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(html_content)
        return text_edit

    def get_workflow_text(self):
        return """
        <h2>FlowFI General Workflow</h2>
        <p>FlowFI is a dual-purpose tool for flow and imaging cytometry analysis, combining feature ranking and feature engineering into one application.</p>
        
        <h3>Core Components:</h3>
        <ul>
            <li><b>Refine Tab:</b> Analyzes existing tabular flow cytometry data (<code>.fcs</code> files) to identify and rank the most important measurement channels (features) for describing the data's structure.</li>
            <li><b>Design Tab:</b> Provides an interactive environment to create <i>new</i> quantitative features from imaging cytometry data (<code>.tiff</code> files) by building custom image processing pipelines.</li>
        </ul>

        <h3>A Typical Use Case:</h3>
        <ol>
            <li>A researcher uses the <b>Design</b> tab to engineer a novel biological feature, such as "the symmetry of a protein signal relative to the cell's nucleus."</li>
            <li>They apply this new processing pipeline to a folder of cell images, exporting the results as a new parameter in their main <code>.fcs</code> dataset.</li>
            <li>They then switch to the <b>Refine</b> tab, load the newly augmented <code>.fcs</code> file, and run the analysis to see how important their custom-designed feature is compared to the standard, instrument-provided measurements.</li>
        </ol>
        <p>This workflow allows for a powerful cycle of hypothesis generation (Design) and validation (Refine).</p>
        """

    def get_refine_text(self):
        return """
        <h2>Refine Tab Guide</h2>
        <p>This tab is used to analyze a standard flow cytometry <code>.fcs</code> file to determine the importance of its features.</p>
        
        <h3>How to Use:</h3>
        <ol>
            <li>Enter the <code>.fcs</code> file path manually or click <b>Browse</b> to select a file.</li>
            <li>Use the checkboxes at the top to include or exclude broad categories of features from the analysis.</li>
            <li>Click <b>Execute</b> to start the analysis. The process involves bootstrapping and may take some time, with progress shown in the progress bar.</li>
            <li>Results will be displayed in the main panel, ranked by importance by default.</li>
        </ol>

        <h3>Interpreting the Results:</h3>
        <ul>
            <li><b>Feature Name:</b> The name of the channel from the <code>.fcs</code> file.</li>
            <li><b>Importance Bar:</b> The length of the colored bar indicates the relative importance of the feature. Longer bars are more important.</li>
            <li><b>Sorting:</b> Use the dropdown menu to sort features by different criteria:
                <ul>
                    <li><b>Importance:</b> (Default) Ranks features by their Laplacian Score, which measures how well a feature preserves the local data structure.</li>
                    <li><b>Type:</b> Groups features by their category (e.g., UV, V, B).</li>
                    <li><b>Cluster:</b> Groups features that are algorithmically determined to be similar to each other. The border color indicates cluster membership.</li>
                    <li><b>Centrality:</b> Ranks features by how representative they are of their assigned cluster. Central features are underlined.</li>
                    <li><b>Change from Previous:</b> Compares the current run's rankings to a previously loaded CSV file.</li>
                </ul>
            </li>
        </ul>
        
        <h3>Menu Options (Refine -> ...):</h3>
        <ul>
            <li><b>Save Output as CSV:</b> Saves the full results table, including raw scores and cluster memberships, to a CSV file.</li>
            <li><b>Load Output CSV for Comparison:</b> Loads a previously saved run to enable the "Sort by: Change from Previous" option.</li>
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
            <li>Once a pipeline is defined, use the <b>Parameters</b> menu to apply it to a whole folder of images and export the results.</li>
        </ol>

        <h3>Menu Options:</h3>
        <h4>Preprocessing Menu</h4>
        <ul>
            <li><b>Presets:</b> Pre-defined pipelines for common tasks.</li>
            <li><b>Filter:</b> Noise reduction and smoothing operations (e.g., <i>Gaussian Filter</i>).</li>
            <li><b>Manipulation:</b> Geometric transformations like <i>Crop</i> and <i>Rescale</i>.</li>
            <li><b>Segmentation:</b> Operations to isolate objects of interest.
                <ul>
                    <li><i>Mask Otsu:</i> Creates a binary mask using Otsu's thresholding.</li>
                    <li><i>Label Image:</i> Assigns a unique integer to each disconnected object in a binary image.</li>
                    <li><i>Segment:</i> Uses a watershed algorithm to separate touching objects.</li>
                </ul>
            </li>
            <li><b>Reset Preprocessing:</b> Clears all applied preprocessing steps for the current image.</li>
            <li><b>Undo Last Operation:</b> Removes the most recent preprocessing step.</li>
        </ul>

        <h4>Quantify Menu</h4>
        <p>Defines how the final processed image is converted into a single number. These are mutually exclusive.</p>
        <ul>
            <li><b>Aggregation:</b>
                <ul>
                    <li><i>Count (unique):</i> Counts the number of unique non-zero labels in the image (useful after a 'Label' or 'Segment' step).</li>
                    <li><i>Mean (non-zero):</i> Calculates the mean intensity of all non-zero pixels.</li>
                </ul>
            </li>
            <li><b>Geometry:</b>
                <ul>
                    <li><i>Area (non-zero):</i> Counts the number of non-zero pixels.</li>
                    <li><i>Solidity:</i> Measures the ratio of the object's area to the area of its convex hull. A perfect convex shape has a solidity of 1.</li>
                    <li><i>Colocalisation:</i> Measures the fraction of a 'Signal' channel's intensity that is within a 'Mask' channel.</li>
                    <li><i>Containment:</i> Measures the fraction of a 'Signal' channel's intensity that is inside the core of a 'Container' channel (excluding its shell).</li>
                    <li><i>Relative Skewness:</i> Measures the radial skewness of a 'Signal' relative to the centroid of a 'Reference'.</li>
                    <li><i>Angular Momentum:</i> Measures the angular asymmetry of a 'Signal' relative to the centroid of a 'Reference'.</li>
                    <li><i>Angular Entropy:</i> Measures the angular uniformity of a 'Signal' relative to the centroid of a 'Reference'. A value of 1 is perfectly uniform.</li>
                    <li><i>Spatial Correlation:</i> Calculates the Pearson correlation between two channels within a defined mask.</li>
                </ul>
            </li>
        </ul>

        <h4>Parameters Menu</h4>
        <p>Applies the complete, currently defined pipeline (preprocessing + quantification) to a folder of images.</p>
        <ul>
            <li><b>Export to FCS:</b> Creates a new <code>.fcs</code> file, adding the calculated feature as a new parameter. Requires an existing <code>.fcs</code> file in the folder to use as a template.</li>
            <li><b>Export to CSV:</b> Creates a <code>.csv</code> file containing the calculated feature value for each image.</li>
        </ul>
        """

class RefinePreferencesDialog(QDialog):
    def __init__(self, parent=None, default_boots=200, default_bootsize=1000, dataset_size=None, default_conv_check=True, default_conv_threshold=1e-5):
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

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create and show splash screen
    splash = None
    splash_path = 'flowfi_logo_white.png'
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.showMessage("Loading FlowFI...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        splash.show()
        app.processEvents() # Ensure splash screen is displayed

    main_window = MainWindow()
    main_window.showMaximized()

    if splash:
        splash.finish(main_window)

    sys.exit(app.exec_())
