import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg
import math
from dataset import DataSet
from BaselineRemoval import BaselineRemoval
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import torch
import pywt
import warnings
import pandas as pd
from scipy.interpolate import interp1d
from predict2 import colors, seq_to_excel

warnings.filterwarnings('ignore')


def baseline_tools(input_array):
    # input_array = [10, 20, 1.5, 5, 2, 9, 99, 25, 47]
    polynomial_degree = 8  # only needed for Modpoly and IModPoly algorithm

    baseObj = BaselineRemoval(input_array)
    Modpoly_output = baseObj.ModPoly(polynomial_degree)
    Imodpoly_output = baseObj.IModPoly(polynomial_degree)
    Zhangfit_output = baseObj.ZhangFit()

    return Zhangfit_output


# ===== Chinese display configuration =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows: SimHei; replace with ['PingFang SC'] for macOS
plt.rcParams['axes.unicode_minus'] = False


def als_baseline_correction(x, lam, p, n_iter):
    '''Asymmetric Least-Squares baseline correction (remove baseline)'''
    L = len(x)
    D = scipy.sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)  # Initialize weights
    for i in range(n_iter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w * x)  # Solve for baseline z
        w = p * (x > z) + (1 - p) * (x < z)  # Update weights
    baseline = z
    corrected_signal = x - baseline
    return corrected_signal, baseline


def polynomial_baseline_correction(x, order=3, n_iter=100):
    '''Polynomial baseline correction (remove baseline)'''
    base = x.copy()
    cond = math.pow(abs(x).max(), 1. / order)
    y = np.linspace(0., cond, base.size)
    vander = np.vander(y, order)
    vander_pinv = scipy.linalg.pinv(vander)

    for _ in range(n_iter):
        coeffs = np.dot(vander_pinv, base)
        z = np.dot(vander, coeffs)  # Fit current baseline
        base = np.minimum(base, z)  # Iteratively optimize baseline
    baseline = z
    corrected_signal = x - baseline
    return corrected_signal, baseline


class LSF:
    def __init__(self, mode="detect", height=0.01, prominence=0.1, width=5):
        self.mode = mode
        self.height = height
        self.prominence = prominence
        self.width = width

    # Define single Gaussian function
    @staticmethod
    def gaussian(x, amplitude, mean, left_std, right_std):
        left_half = amplitude * np.exp(-(x - mean) ** 2 / (2 * left_std ** 2))
        right_half = amplitude * np.exp(-(x - mean) ** 2 / (2 * right_std ** 2))
        gaussian = np.where((np.arange(len(x))) < mean, left_half, right_half)
        return gaussian

    # Define multi-Gaussian function for fitting
    def multi_gaussian(self, x, *params):
        y_sum = np.zeros_like(x)
        for i in range(0, len(params), 4):
            amplitude = params[i]
            mean = params[i + 1]
            left_std = params[i + 2]
            right_std = params[i + 3]
            y_sum += self.gaussian(x, amplitude, mean, left_std, right_std)
        return y_sum

    def __call__(self, smoothed_sequence, width_max):
        # Apply SG filter to smooth waveform data and calculate second derivative
        peaks, _ = find_peaks(smoothed_sequence, height=self.height, prominence=self.prominence, width=self.width)
        # Prepare initial parameters (example, can be optimized based on actual conditions)
        initial_guess = []
        for p in peaks:
            initial_guess.extend([smoothed_sequence[int(p)], p, 10, 10])

        lower_bounds = []
        upper_bounds = []

        for p in peaks:
            # Amplitude lower bound is 0
            lower_bounds.extend([smoothed_sequence[int(p)] * 0.2, p - 50, 1.0, 1.0])
            # Assume mean upper bound is max value of x, std upper bound can be set to a larger value based on actual conditions
            upper_bounds.extend([min(smoothed_sequence[int(p)] * 2, 1.0), p + 50, 30.0, 30.0])
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        x = np.linspace(0, len(smoothed_sequence), len(smoothed_sequence))

        # Perform fitting using least squares method with bounds parameter
        try:
            popt, pcov = curve_fit(self.multi_gaussian, x, smoothed_sequence, p0=initial_guess, maxfev=1000,
                                   bounds=(lower_bounds, upper_bounds))
        except Exception as e:
            # print(f"Fitting error: {e}, returning empty results")
            outputs = np.array([])
            return outputs, outputs, outputs, outputs, outputs
        num_peaks = len(peaks)
        fitted_amplitudes = popt[0:num_peaks * 4:4]
        fitted_means = popt[1:num_peaks * 4:4]
        fitted_left_stds = popt[2:num_peaks * 4:4]
        fitted_right_stds = popt[3:num_peaks * 4:4]
        fitted_start = fitted_means - 3 * fitted_left_stds
        fitted_end = fitted_means + 3 * fitted_right_stds
        fitted_peaks = fitted_means
        fitted_height = fitted_amplitudes * width_max
        fitted_area = fitted_amplitudes * width_max * (fitted_left_stds + fitted_right_stds) * math.sqrt(
            2 * math.pi) / 2

        return fitted_start, fitted_end, fitted_peaks, fitted_height, fitted_area


if __name__ == "__main__":
    def read_xlsx_by_columns(sample_file_path, sheet_name: str | int = 0) -> dict:
        try:
            # Read xlsx file without setting header (header=None) to retain raw data
            df = pd.read_excel(sample_file_path, sheet_name=sheet_name, header=None)

            # Initialize result dictionary
            column_data = {}

            # Iterate through each column (df.columns are column indices: 0,1,2...)
            for col in df.columns:
                # Get all data of current column (convert to list)
                col_all_data = df[col].tolist()

                # 1. Take the first element of the column as key (convert to string + remove spaces to avoid type/space issues)
                key = str(col_all_data[0]).strip() if len(col_all_data) > 0 else ""
                # Skip empty keys (columns with empty first cell)
                if key == "" or key == "nan":
                    continue

                # 2. Take all data from the second element of the column as value list
                value = col_all_data[1:] if len(col_all_data) > 1 else []

                value = [item for item in value if not (isinstance(item, float) and np.isnan(item))]

                # 3. Store in dictionary
                column_data[key] = value

            return column_data

        except FileNotFoundError:
            # Fix original bug: change undefined file_path to sample_file_path
            print(f"Error: File {sample_file_path} not found, please check if the path is correct")
            return {}
        except KeyError:
            print(f"Error: Worksheet {sheet_name} does not exist, please check worksheet name/index")
            return {}
        except Exception as e:
            print(f"Failed to read file: {str(e)}")
            return {}


    def adjust_sequence(sequence, max_value=1.0):
        width_max = np.max(sequence)
        sequence = sequence / width_max
        return sequence, width_max


    def generate_gaussian_sequence(gaussian_params, sequence):
        # Parse parameters
        location, amplitude, left_std, right_std = gaussian_params
        # Generate sequence indices (assuming data length is DATA_LENGTH)
        indices = np.arange(len(sequence))
        output_sequences = []

        # Generate Gaussian sequence for each peak
        for i in range(len(location)):
            loc = location[i]
            amp = amplitude[i]
            left = left_std[i]
            right = right_std[i]

            # Left Gaussian distribution (use left standard deviation for left side of peak)
            left_half = np.exp(-0.5 * ((indices - loc) / left) ** 2)
            # Right Gaussian distribution (use right standard deviation for right side of peak)
            right_half = np.exp(-0.5 * ((indices - loc) / right) ** 2)
            # Merge left and right halves to form asymmetric Gaussian peak
            gaussian = np.where(indices < loc, left_half, right_half)
            gaussian *= amp  # Apply amplitude
            output_sequences.append(gaussian)

        return output_sequences


    sample_data = read_xlsx_by_columns("sample/sample_row_sums.xlsx", "data")
    for i in range(85, 102):
        print(i + 1)
        sequence, width_max = adjust_sequence(sample_data[f"sample{i + 1}"])
        inputs_without_baseline = baseline_tools(sequence)
        smoothed_sequence = savgol_filter(inputs_without_baseline, window_length=3, polyorder=2, deriv=0)
        lsf = LSF(height=0.01, prominence=0.01, width=5)
        output_left_edges, output_right_edges, output_locations, output_amplitudes, output_areas = lsf(
            smoothed_sequence, width_max)
        output_left_stds = (output_locations - output_left_edges) / 3
        output_right_stds = (output_right_edges - output_locations) / 3
        gaussian_params = output_locations, output_amplitudes, output_left_stds, output_right_stds
        output_sequences = generate_gaussian_sequence(gaussian_params, smoothed_sequence)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 6))

        # Plot each predicted Gaussian peak in ax3 (distinguished by different colors)
        for i in range(len(output_sequences)):
            ax3.plot(output_sequences[i], color=colors[i])
        ax4.plot(np.array(output_sequences).sum(axis=0),color="black")

        # seq_to_excel(np.array(sequence)*width_max,output_sequences,"sequence2.xlsx")

        # Plot original sequence in ax1 (black curve)
        ax1.plot(np.array(sequence) * width_max, color='black')
        ax2.plot(np.array(inputs_without_baseline) * width_max, color="black")

        # Set axis labels
        ax1.set_ylabel('Amplitude ')  # Y-axis label for ax1
        ax2.set_ylabel('Amplitude ')  # Y-axis label for ax2
        # Set axis ranges
        ax1.set_ylim(0, 1.2 * width_max)
        ax2.set_ylim(0, 1.2 * width_max)
        ax3.set_ylim(0, 1.2 * width_max)
        ax4.set_ylim(0, 1.2 * width_max)
        ax1.set_xlim(0, len(sequence))
        ax2.set_xlim(0, len(sequence))
        ax3.set_xlim(0, len(sequence))
        ax4.set_xlim(0, len(sequence))
        # Auto-adjust layout
        plt.tight_layout()
        plt.show()  # Display second set of figures