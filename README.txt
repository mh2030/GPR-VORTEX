# GPR VORTEX

**Advanced Ground Penetrating Radar (GPR) Processing and Interpretation Software**

GPR VORTEX is a Python-based software platform developed for processing, visualization, enhancement, and interpretation of Ground Penetrating Radar (GPR) data. The software integrates advanced signal processing algorithms, interactive visualization tools, and interpretation utilities into a user-friendly environment for geophysical investigations.

---

## Key Features

### Data Import

* Reading IPRH header files
* Reading IPRB binary radar data
* Automatic data validation

### Signal Processing Tools

* Background Removal
* Gaussian Smoothing
* Sobel Filter
* Prewitt Filter
* Scharr Filter
* Laplace Filter
* Absolute Value Transformation
* Horizon Flattening
* Air-Wave Reflection Removal
* Binomial Filtering
* Moving Average Filtering
* Maximum Filter
* Minimum Filter
* Adaptive Thresholding
* Local Entropy Thresholding

### Visualization

* Interactive 2D Radargrams
* Adjustable Color Maps
* Gain Control
* Distance and Depth Scaling
* High-Resolution Export

### Interpretation Tools

* Interactive Grid Interpretation
* Marker-Based Anomaly Mapping
* Grid Save/Load Functions
* Interpretation Export

### Export Functions

* PNG Export
* JPG Export
* JSON Project Export
* Processed Radargram Saving

---

## Applications

GPR VORTEX can be applied in:

* Utility Detection
* Archaeological Investigations
* Environmental Assessments
* Geotechnical Studies
* Infrastructure Inspection
* Groundwater Exploration
* Mining and Exploration Projects
* Subsurface Mapping

---

## Supported Input Formats

| Format | Description        |
| ------ | ------------------ |
| IPRH   | Header Information |
| IPRB   | Binary Radar Data  |

---

## Software Requirements

### Operating System

* Windows 10
* Windows 11

### Python Version

* Python 3.10+

### Required Packages

```bash
pip install numpy scipy matplotlib pillow plotly opencv-python scikit-image scikit-learn
```

---

## Installation from Source

Clone the repository:

```bash
git clone https://github.com/USERNAME/GPR-VORTEX.git
```

Move to the project directory:

```bash
cd GPR-VORTEX
```

Run:

```bash
python main.py
```

---

## Executable Version

A standalone Windows executable version (**GPR_VORTEX.exe**) is available for users who prefer not to install Python or additional dependencies.

The executable can be downloaded directly from the **Releases** section of this repository and used immediately.

No installation is required.

---

## Source Code Availability

The complete source code is openly available to support:

* Scientific transparency
* Reproducible research
* Academic collaboration
* Community-driven development
* Software extension and customization

Researchers, students, and developers are encouraged to contribute to the project through issues, discussions, and pull requests.

---

## Downloads

### Windows Executable

Download the latest executable version from:

**GitHub → Releases → Latest Version**

### Source Code

The complete Python source code is available in this repository for research, educational, and collaborative development purposes.

---

## Contributing

Contributions are welcome.

Possible contribution areas include:

* New GPR processing algorithms
* Advanced visualization modules
* Performance optimization
* Cross-platform support
* Documentation improvements

To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## Citation

If you use GPR VORTEX in academic work, please cite:

> Soleimani, M.H. (2026). GPR VORTEX: Advanced Ground Penetrating Radar Processing and Interpretation Software. Zenodo. DOI: [To be assigned after publication].

---

## DOI and Versioning

Each official software release is archived through Zenodo and assigned a unique DOI to ensure long-term accessibility, reproducibility, and citation.

---

## Author

**Mohammad Hassan Soleimani**

M.Sc. Geophysics (Exploratory Seismology)

University of Tehran

---

## License

MIT License

Copyright (c) 2026 Mohammad Hassan Soleimani

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, modify, distribute, and publish the software subject to the conditions of the MIT License.
