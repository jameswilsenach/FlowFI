# FlowFI - Image Parameter Design Update v0.4

FlowFI (Flow cytometry Feature Importance) is a Python-based, graphical tool for experimentalists, clinicians, and analysts to perform data-driven analysis and now creation of cytometry data. This new version combines two key workflows into a single application:

- **Feature Design**: Interactively build image processing pipelines to engineer and quantify novel morphological or spatial features from imaging cytometry data (.tiff).
- **Feature Refinement**: Analyze existing tabular data (.fcs, .csv) to rank measurement channels (features) by their importance to the data's structure.

The software was originally designed for data from instruments like the BD FACSDiscover™ S8 Cell Sorter but is compatible with generic .fcs and .tiff files. FlowFI does not perform or suggest a gating strategy, but instead ranks features by how much of the variance in the samples they account for using a robust spectral manifold learningmethod based on the Laplacian Score [1], with more importance measures to follow.

This dual-tab approach allows for users to cycle between hypothesis generation (Design) and validation (Refine). A researcher can engineer a new biological feature, export it as a parameter, and then use the Refine tab to see how important their custom feature is compared to other parameters or related measurements.

## Installation
For Windows machines, download the flowfi.zip file in the repository and unzip it to an available folder. Run the program as you would any other .exe file.

To install FlowFI from source in a new Python 3.10 environment, download the repository and navigate to the FlowFI directory. Then use the following command in your command line:

```
conda create -n flowfi python=3.10
conda activate flowfi
pip install flowkit
conda install -c conda-forge numpy=1.26 pandas scipy opencv pyqt scikit-learn scikit-learn-extra matplotlib leidenalg tifffile scikit-image
```

To build a flowfi executable on your platform follow these steps:

```
conda activate flowfi
pip install pyinstaller
pyinstaller --onefile --windowed main.py -n flowfi --collect-all flowkit
```
FlowFI depends on FlowKit (which has additional dependent non-python files that must be explicitly included in the build) for extraction and manipulation of .fcs files. The executable should be found in the .dist directory.
### Run FlowFI from Source
Alternatively, while in the FlowFI directory, run the following in the command line:

```
python flowfi_dev.py
```

FlowFI uses FlowIO and FlowKit to load .fcs files and PyQt5 to implement the Graphical User Interface (GUI).

## Using FlowFI: The Design Tab
This tab is a workbench for creating new, quantifiable features from multi-channel .tiff images.

![design_tab](https://github.com/jameswilsenach/FlowFI/blob/main/design.png?raw=true)

### Basic Workflow:
- Use the file tree on the left to navigate to and double-click a .tiff file to load it.
- The original image for the selected channel appears in the top-left panel. The top-right panel shows the result of the image processing pipeline.
- Use the menus (Preprocessing, Quantify) to build an analysis pipeline. Preprocessing operations are applied sequentially with the currently selected quantify option as the final operation that will produce a numerical value.
- The Operation History terminal shows the list of applied steps and the result of any quantification.
- Once a pipeline is defined, use the Parameters menu to apply it to a whole folder of images and export the results to a standard format (.fcs/.csv).


### Single-Channel Quantification Options:
- **Count**: Counts unique non-zero labels (for counting objects).
- **Area**: Counts the number of non-zero pixels.
- **Solidity**: Measures object compactness.

### Multi-Channel Quantification Options:
- **Colocalisation**: Fraction of a 'Signal' channel's intensity within a 'Mask' channel.
- **Containment**: Fraction of a 'Signal' inside the core of a 'Container' (excluding its shell).
- **Relative Skewness**: Radial skewness of a 'Signal' relative to a 'Reference' centroid.
- **Angular Momentum**: Angular momentum of a 'Signal' around a 'Reference' centroid.
- **Angular Entropy (Symmetry)**: Uniformity of a 'Signal' around a 'Reference' centroid.
- **Spatial Correlation**: Pearson correlation between two channels within a mask.

## Exporting Parameters
- **Export to FCS**: Creates a new .fcs file, adding the calculated feature as a new parameter. Requires a template .fcs file in the folder.
- **Export to CSV**: Creates a .csv file containing the calculated feature value for each image.


## Using FlowFI: The Refine Tab
This tab is used to analyze a standard flow cytometry .fcs or .csv file to determine the importance of its features.

![refine_tab](https://github.com/jameswilsenach/FlowFI/blob/main/refine.png?raw=true)

### How to Use:
- Enter the data file path manually or click Browse to select a file.  
- Use the checkboxes at the top to include or exclude broad categories of features from the analysis.  
- Click Execute to start the analysis. The process involves bootstrapping and may take some time, with progress shown in the progress bar.  
- Results will be displayed in the main panel, ranked by importance by default.



### Interpreting the Results:
**Feature Name:** The name of the channel from the source file.  
**Importance Bar:** The length of the colored bar indicates the relative importance of the feature. Longer bars are more important. The bar can also display confidence intervals (see menu options).  
**Sorting:** Use the dropdown menu to sort features by different criteria:  
*Importance* (Default): Ranks features by their Laplacian Score.  
*Type*: Groups features by their category (e.g., UV, V, B).  
*Cluster*: Groups features that are algorithmically determined to be similar (the colored border indicates cluster membership).  
*Centrality*: Ranks features by how representative they are of their assigned cluster. Central features are underlined.  
*Change from Previous*: Compares the current run's rankings to a previously loaded CSV file.

### Saving and Loading Refinement Outputs for Comparison/Analysis
**Save Output as CSV**: Saves the full results table, including raw scores, cluster memberships, and confidence intervals, to a CSV file.

**Load Output CSV for Comparison**: Loads a previously saved run to enable the "Sort by: Change from Previous" option.

**Calculate Importance CIs**: Toggles the calculation and display of confidence intervals on the importance bars, providing a measure of estimate stability.

FlowFI saves the analysis output in a CSV file with the following columns:
**feature**: Name of the corresponding feature in the original file.

**ri**: Relative Importance (normalized Laplacian Score from 0 to 1).

**ls**: Raw Laplacian Score.

**membership**: Numerical ID of the cluster this feature belongs to.

**centrality**: Score from 0-1 indicating how representative the feature is of its cluster.

**LowCI / UpperCI**: The lower and upper confidence interval bounds for the Relative Importance (if calculated).

**comparison**: The change in rank compared to a loaded reference file (if used).

## References
[1] He, X., Cai, D., & Niyogi, P. (2005). Laplacian score for feature selection. Advances in neural information processing systems, 18.  
[2] Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 5233.  
[3] Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus Clustering: A Resampling-Based Method for Class Discovery and Visualization of Gene Expression Microarray Data. Machine Learning, 52, 91–118.  
[4] Kaufman, L., & Rousseeuw, P. J. (1990). Partitioning around medoids (Program PAM). In Finding Groups in Data: An Introduction to Cluster Analysis. John Wiley & Sons.  
[5] Kendall, M. G. (1938). A New Measure of Rank Correlation. Biometrika, 30(1/2), 81–93.  
[6] Vinh, N. X., Epps, J., & Bailey, J. (2010). Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance. Journal of Machine Learning Research, 11, 2837–2854.  
[7] Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions. Neural Computation, 16(6), 1299-1323.
