# YOLO-PeakDetect

A peak detection tool based on the YOLO model, designed for fast identification of peak targets (such as specific objects, signal peaks, etc.) in images or signals. This project provides convenient detection functions and visualized results, supporting custom parameter configuration to adapt to different scenarios.



## Project Introduction

YOLO-PeakDetect combines the efficiency of the YOLO (You Only Look Once) object detection algorithm with peak detection logic, enabling fast localization and labeling of target peaks in both real-time and offline scenarios. It is suitable for the detection of electrophoresis profiles, chromatograms, and spectra.

## Installation

### Method 1: Install from Source Code

1.Clone the repository to your local machine:
git clone https://github.com/yourusername/python-YOLO-PeakDetect.git
cd python-YOLO-PeakDetect

2.Install dependencies:
pip install -r requirements.txt

3.Run the project:
python gui.py

### Method 2: Install via Executable (Recommended for Beginners)

The project provides a standalone executable installer EAP\_Setup.exe (not uploaded to the repository). It can be run directly without configuring a Python environment.
Obtaining the installer: Contact the project maintainer to get EAP\_Setup.exe.
Installation steps: Double-click the installer and follow the wizard. After installation, you can find the program shortcut in the Start Menu or on the desktop.

Usage Instructions

Select the electropherogram as prompted. Click "Automatic Lane Recognition" – manual adjustment is available. Click "Finish Lane Recognition" when completed. Then click "Automatic Region Selection" – manual adjustment is also supported. Click "Finish Region Selection" after adjustment. Finally, click "Peak Detection" to obtain the visualized results.

Directory Structure
python-YOLO-PeakDetect/
├── analysisi/   		# Analyze network performance during the training process
├── configs/         	# Configuration network parameters
├── dataset/ 		# simulated dataset
├── gui/ 	  		# User interface (UI)
├── loss/ 	  		# calculate the loss
├── Net/              		# YOLO-PeakDetect network architecture
├── predict/         		# UI invokes network prediction
├── predict2/       		# Perform network prediction independently
├── train/             		# Train the network(YOLO-PeakDetect)
├── utils/              		# Utility functions
├── requirements.txt   # Dependencies list
└── README.md         # Project description

### Notes

If running from source code, ensure you have Python 3.8 or higher installed.
EAP\_Setup.exe only supports the Windows operating system. For installers for other systems, please contact the maintainer.
For optimal performance, a CUDA-enabled GPU is recommended for model inference. Ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed if using GPU acceleration.

