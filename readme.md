# YOLO-PeakDetect

A peak detection tool based on the YOLO model, designed for fast identification of peak targets (such as specific objects, signal peaks, etc.) in images or signals. This project provides convenient detection functions and visualized results, supporting custom parameter configuration to adapt to different scenarios.



## Project Introduction

YOLO-PeakDetect combines the efficiency of the YOLO (You Only Look Once) object detection algorithm with peak detection logic, enabling fast localization and labeling of target peaks in both real-time and offline scenarios. It is suitable for the detection of electrophoresis profiles, chromatograms, and spectra.

## Installation

### Method : Install from Source Code

1.Clone the repository to your local machine:
git clone https://github.com/shuai-qi7/yolo-peakdetect.git
cd python-YOLO-PeakDetect

2.Install dependencies:
pip install -r requirements.txt

3.Run the project:
python gui.py

Usage Instructions

Select the electropherogram as prompted. Click "Automatic Lane Recognition" – manual adjustment is available. Click "Finish Lane Recognition" when completed. Then click "Automatic Region Selection" – manual adjustment is also supported. Click "Finish Region Selection" after adjustment. Finally, click "Peak Detection" to obtain the visualized results.

Directory Structure
python-YOLO-PeakDetect/
├── sample/ 	                       	       	# Experimental dataset (include sample_row_sums (1D sample) and multi_annot_records (annot data)) 
├── analysis/   		 		# Analyze network performance during the training process
├── comparision_on_experimental_dataset         	# comparison our network with traditional method on experimental dataset ✅ Executable 
├── configs/        		 	               	# Configuration network parameters
├── dataset/ 				# simulated dataset ✅ Executable 
├── gui/ 	  				# User interface (UI) (need 2D GE image like 1.jpg)✅ Executable 
├── label_experimential_gel/ 		# the tool of label the experimental dataset (single peak and overlapping peak) ✅ Executable 
├── loss/ 	  			# calculate the loss
├── Net/              				# YOLO-PeakDetect network architecture
├── predict/         				# UI invokes network prediction
├── predict2/       				# Perform network prediction independently (2D gel lane or 1D data) ✅ Executable 
├── traditional_method_gel/			# traditional method on gel experimental dataset  ✅ Executable 
├── traditional_method_sim/			# test performance of traditional method on simulated dataset ✅ Executable 
├── train/             				# Train the network(YOLO-PeakDetect) ✅ Executable 
├── utils/              				# Utility functions
├── requirements.txt   			# Dependencies list
└── README.md         			# Project description

### Notes

If running from source code, ensure you have Python 3.8 or higher installed.
For optimal performance, a CUDA-enabled GPU is recommended for model inference. Ensure you have the appropriate NVIDIA drivers and CUDA toolkit installed if using GPU acceleration.

