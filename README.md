# FlowFI - Image Parameter Design & Feature Importance (v1.6.0)

FlowFI (Flow cytometry Feature Importance) is a Python-based, graphical tool for experimentalists, clinicians, and analysts to perform data-driven analysis and creation of cytometry data. FlowFI combines two key workflows into a single application:

- **Feature Design**: Interactively build, save, and automate image processing pipelines to engineer and quantify novel morphological or spatial features from imaging cytometry data (.tiff).
- **Feature Refinement**: Analyze tabular data (.fcs, .csv) to rank measurement channels (features) by their importance to the data's structure using manifold learning and statistical metrics.

The software was originally designed for data from instruments like the BD FACSDiscover™ S8 Cell Sorter but is compatible with generic .fcs and .tiff files. FlowFI does not perform or suggest a gating strategy, but instead ranks features by how much variance in the samples they account for using robust feature importance metrics (such as the Laplacian Score [1], PCA-based, SOM-based, or Mutual Information metrics).

This dual-tab approach allows users to cycle between hypothesis generation (Design) and validation (Refine). A researcher can engineer a new biological feature, export it as a parameter, and then use the Refine tab to evaluate how important their custom feature is compared to standard cytometry measurements.

---

## Installation

### Windows Application
For Windows machines, download the latest installer or executable package (`FlowFI_v1.6.0.msix` / `FlowFI.exe`). Install and run the program as you would any native Windows application.

### Run FlowFI from Source
To install FlowFI from source in a new Python 3.10 environment, clone or download the repository, navigate to the FlowFI directory, and run:

```bash
conda create -n flowfi python=3.10
conda activate flowfi
pip install flowkit opencv-python-headless "numpy<2"
conda install -c conda-forge "numpy<2" pandas scipy pyqt scikit-learn scikit-learn-extra matplotlib leidenalg tifffile scikit-image minisom
```

To launch FlowFI from the command line:

```bash
python main.py
```

### Build Executable
To build a FlowFI executable on your platform:

```bash
conda activate flowfi
pip install pyinstaller
pyinstaller --onefile --windowed main.py -n flowfi --collect-all flowkit
```
The resulting executable will be located in the `dist/` directory.

---

## Using FlowFI: The Design Tab

The Design Tab is a workbench for creating new, quantifiable features from multi-channel .tiff images.

![design_tab](https://github.com/jameswilsenach/FlowFI/blob/main/design.png?raw=true)

### Basic Workflow:
- Use the **File Tree** on the left to navigate folders and double-click a `.tiff` file to load it into the workspace.
- The top-left panel displays the original selected channel image. The top-right panel shows the processed image resulting from your active pipeline.
- Use the **Preprocessing** and **Quantify** menus to build an image analysis pipeline. Preprocessing operations are applied sequentially, with the selected quantify option producing the final numerical feature value.
- Use **Undo** (`Ctrl+Z`) and **Redo** (`Ctrl+Y`) from the Preprocessing menu or toolbar to modify your pipeline sequence step-by-step.
- The **Operation History** panel displays applied processing steps and quantification output.

### Presets & Pipeline Automation
- **Built-in Presets**: Access standard preset pipelines such as **OFDM (Optical Frequency Domain Multiplexing)** directly from `Preprocessing > Presets`.
- **Save / Load Presets**: Save custom image processing pipelines to reusable JSON files or load existing preset files.
- **Configurable Location**: Set a custom default folder path for saving and loading presets via `Preprocessing > Presets > Configure Presets Location`.

### Preprocessing Operations:
- **Filters**: Gaussian Blur (customizable kernel & sigma), Denoising.
- **Manipulation**: Image Crop, Image Rescale (scaling factors & interpolation selection).
- **Segmentation**: Mask Otsu, Label Image, Segment.

### Single-Channel Quantification Options:
- **Count**: Counts unique non-zero labels (for object counting).
- **Mean**: Calculates mean pixel intensity across non-zero regions.
- **Area**: Counts total number of non-zero pixels.
- **Solidity**: Measures object compactness relative to its convex hull.

### Multi-Channel Quantification Options:
- **Colocalisation**: Fraction of a Signal channel's intensity within a Mask channel.
- **Containment**: Fraction of Signal inside the core of a Container (excluding its shell).
- **Relative Skewness**: Radial skewness of Signal relative to a Reference centroid.
- **Angular Momentum**: Angular momentum of Signal around a Reference centroid.
- **Angular Entropy (Symmetry)**: Uniformity of Signal distribution around a Reference centroid.
- **Spatial Correlation**: Pearson correlation between two channels within a mask region.
- *Note: Multi-channel dialogs include an option to disable Signal-to-Noise Ratio (SNR) checks for low-signal analyses.*

### Parameters & Batch Processing:
- **Export to FCS**: Appends calculated feature parameters to standard `.fcs` files (requires a reference template `.fcs`).
- **Export to CSV**: Generates a `.csv` containing calculated parameters for all images in a target folder.
- **Batch Process Folder**: Runs the current pipeline across an entire directory of `.tiff` files.
- **Concatenate CSVs**: Merges multiple parameter `.csv` output files across folders into a consolidated dataset.
- **Merge CSV into FCS**: Injects parameter columns from a `.csv` directly into existing `.fcs` files.
- **Export Terminal**: Saves the full log of terminal execution and operations history to text.

---

## Using FlowFI: The Refine Tab

The Refine Tab evaluates standard flow cytometry `.fcs` or `.csv` files to rank feature importance.

![refine_tab](https://github.com/jameswilsenach/FlowFI/blob/main/refine.png?raw=true)

### How to Use:
1. Enter the data file path or click **Browse** to select an `.fcs` or `.csv` file.
2. Select category checkboxes at the top to include or exclude specific channel types from analysis.
3. Click **Execute** to calculate feature importance scores (uses bootstrapping, progress shown in progress bar).
4. View ranked results in the main interactive table and bar chart.

### Relative Importance (RI) Metrics:
Choose the ranking metric under `Refine > RI Metric`:
- **lsRI (Laplacian Score)**: Default spectral manifold learning metric ranking features by structure preservation.
- **pRI (PCA-based)**: Ranks features by variance contribution via Principal Component Analysis.
- **sRI (SOM-based)**: Ranks features using Self-Organizing Map topological representation.
- **miRI (Mutual Information)**: Measures feature importance via mutual information score.

### Preferences & Customization:
- **Refine Preferences**: Adjust bootstrap iterations, subsample size, convergence checks, and convergence thresholds via `Refine > Preferences...`.
- **Confidence Intervals**: Toggle calculation of confidence intervals (`Refine > Calculate Importance CIs`) with customizable alpha levels.
- **Small Sample Adaptation**: Automatically adapts bootstrap sub-sampling parameters when working with small cell populations or low-event files to ensure statistical stability.
- **Automatic Feature Filtering & Propagation**: Identifies zero-variance (constant) features and duplicate measurement columns. Duplicate features are filtered out during computationally intensive manifold calculations, and their computed importance metrics, centrality, and confidence intervals are automatically propagated back to the final output table and exported CSVs.

### Sorting, Export & Comparison:
- **Importance** (Default): Ranks features by chosen RI metric score.
- **Type**: Groups features by optical/channel category.
- **Cluster**: Groups features algorithmically into similarity clusters.
- **Centrality**: Identifies central representative features per cluster (underlined).
- **Change from Previous**: Compares current ranking against a reference CSV loaded via `Refine > Load Output CSV for Comparison`.
- **Export Table & Charts**: Save output feature ranking tables to CSV and export high-resolution bar charts and cluster visualizations in image formats.

### Built-in Help & User Guide:
- Access the interactive, offline-capable documentation and step-by-step feature guides directly inside the application via `Help > User Guide`.

---

## Remote HPC Execution (X11 Forwarding)

FlowFI can be run on a remote Linux-based High-Performance Computing (HPC) cluster while rendering the graphical interface locally over SSH using **X11 Forwarding**.

### Prerequisites
Ensure an X11 display server is running locally:
- **Windows**: [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [Xming](https://sourceforge.net/projects/xming/).
- **macOS**: [XQuartz](https://www.xquartz.org/).
- **Linux**: Standard X server.

### Launch Command
```bash
ssh -Y your_username@your_hpc_address
cd /path/to/FlowFI
conda activate flowfi
python main.py
```

---

## Release Notes

### Release Notes - v1.6.0
* **Version Incremented**: Version for consistency with Microsoft Store.
* **Preset Pipeline Management**: Added built-in OFDM presets alongside saving, loading, and custom folder configuration for image processing pipelines.
* **Expanded Operations & Quantification**: Introduced Image Crop, Rescale, Gaussian Blur, Mean intensity quantification, and SNR check options for multi-channel metrics.
* **Data Integration Utilities**: Added utilities to concatenate parameter CSV files and merge calculated parameter metrics directly into FCS files.
* **UI & History Enhancements**: Implemented an Operation Undo/Redo stack, cross-platform AppData config persistence, and an interactive embedded User Guide.
* **Duplicate Feature Handling**: Automatic detection, filtering, and metric propagation for duplicated feature columns during dataset refinement.

### Release Notes - v0.5.1
* Fixed an issue where single channel aggregation depended on multichannel aggregation to be run first.
* Fixed an issue with OpenCV causing the Windows application to enter an infinite loop.

---

## Developer Guide & Feature Roadmap

If you are a developer looking to get started or contribute to FlowFI, here are useful feature ideas and areas for enhancement:

- **Channel-Specific Preprocessing**: Support channel-specific transformations within image pipelines (e.g., channel-specific color inversion for masking in brightfield-like channels).
- **Custom Declaration of Feature/Parameter Types**: Support user-defined feature/parameter types (e.g., custom image parameters), expanding beyond the current reliance on BD FACSDiscover™ S8 Cell Sorter naming conventions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

[1] He, X., Cai, D., & Niyogi, P. (2005). Laplacian score for feature selection. Advances in Neural Information Processing Systems, 18.  
[2] Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9(1), 5233.  
[3] Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data. Machine Learning, 52, 91–118.  
[4] Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (Program PAM). In Finding Groups in Data: An Introduction to Cluster Analysis. John Wiley & Sons.  
[5] Kendall, M. G. (1938). A New Measure of Rank Correlation. Biometrika, 30(1/2), 81–93.  
[6] Vinh, N. X., Epps, J., & Bailey, J. (2010). Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance. Journal of Machine Learning Research, 11, 2837–2854.  
[7] Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions. Neural Computation, 16(6), 1299-1323.
