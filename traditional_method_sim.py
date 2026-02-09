import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.linalg
import math
from dataset import DataSet
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from BaselineRemoval import BaselineRemoval
import torch
import pywt
import warnings

warnings.filterwarnings('ignore')

# ===== Chinese display configuration =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows: SimHei; replace with ['PingFang SC'] for macOS
plt.rcParams['axes.unicode_minus'] = False


def baseline_tools(input_array):
    # input_array = [10, 20, 1.5, 5, 2, 9, 99, 25, 47]
    polynomial_degree = 8  # only needed for Modpoly and IModPoly algorithm

    baseObj = BaselineRemoval(input_array)
    Modpoly_output = baseObj.ModPoly(polynomial_degree)
    Imodpoly_output = baseObj.IModPoly(polynomial_degree)
    Zhangfit_output = baseObj.ZhangFit()

    return Zhangfit_output


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

    def __call__(self, smoothed_sequence):
        # Apply SG filter to smooth waveform data and calculate second derivative
        peaks, _ = find_peaks(smoothed_sequence, height=self.height, prominence=self.prominence)
        # Prepare initial parameters (example, can be optimized based on actual conditions)
        initial_guess = []
        for p in peaks:
            initial_guess.extend([smoothed_sequence[int(p)], p, 20, 20])

        lower_bounds = []
        upper_bounds = []

        num_peaks = len(peaks)

        for _ in range(num_peaks):
            # Amplitude lower bound is 0
            lower_bounds.extend([0.0, 10.0, 2.0, 2.0])
            # Assume mean upper bound is max value of x, std upper bound can be set to a larger value based on actual conditions
            upper_bounds.extend([1.0, 1024.0, 30.0, 30.0])
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        x = np.linspace(0, 1024, 1024)

        # Perform fitting using least squares method with bounds parameter
        try:
            popt, pcov = curve_fit(self.multi_gaussian, x, smoothed_sequence, p0=initial_guess, maxfev=500,
                                   bounds=(lower_bounds, upper_bounds))
        except Exception as e:
            # print(f"Fitting error: {e}, returning empty results")
            outputs = torch.zeros(16, 5)
            return outputs
        num_peaks = len(peaks)
        fitted_amplitudes = popt[0:num_peaks * 4:4]
        fitted_means = popt[1:num_peaks * 4:4]
        fitted_left_stds = popt[2:num_peaks * 4:4]
        fitted_right_stds = popt[3:num_peaks * 4:4]

        fitted_start = torch.tensor(fitted_means - 3 * fitted_left_stds).unsqueeze(1)
        fitted_end = torch.tensor(fitted_means + 3 * fitted_right_stds).unsqueeze(1)
        fitted_peaks = torch.tensor(fitted_means).unsqueeze(1)
        fitted_height = torch.tensor(fitted_amplitudes).unsqueeze(1)

        pred_output = torch.cat([torch.ones(num_peaks, 1), fitted_start, fitted_end,
                                 fitted_peaks, fitted_height], dim=1)

        # Make output shape [16,5]
        outputs = torch.zeros(16, pred_output.size(1))
        rows = min(len(pred_output), 16)
        outputs[0:rows] = pred_output[0:rows]
        # print(outputs)

        if self.mode == 'detect':
            peaks_areas = fitted_amplitudes * (fitted_left_stds + fitted_right_stds)
            print(peaks_areas, "area")

        else:
            return outputs


if __name__ == "__main__":

    class SingleSampleDataset:
        def __init__(self, single_sample):
            # single_sample is the tuple (inputs, target, inputs_without_noise) returned by data[i]
            self.single_sample = single_sample

        def __len__(self):
            # Only one sample
            return 1

        def __getitem__(self, idx):
            # Only index 0 is allowed, return the saved single sample
            if idx != 0:
                raise IndexError("SingleSampleDataset only has index 0")
            return self.single_sample


    @staticmethod
    def gaussian(x, mean, std1, std2, height):
        """
        Generate asymmetric Gaussian function values with different standard deviations on either side of the mean.
        Parameters:
            x: Input tensor, positions where Gaussian function values are calculated
            mean: Mean of Gaussian distribution (peak position)
            std1: Left standard deviation (used when x < mean)
            std2: Right standard deviation (used when x >= mean)
            height: Peak height of Gaussian function
        Returns:
            result: Asymmetric Gaussian function values at positions x
        """
        # Set left and right standard deviations separately
        left_std = std1
        right_std = std2

        # Calculate asymmetric Gaussian function values
        if std1 != 0 and std2 != 0:
            # Use left standard deviation when x < mean, otherwise use right standard deviation
            result = torch.where(
                x < mean,  # Condition judgment
                height * torch.exp(-((x - mean) ** 2) / (2 * left_std ** 2)),  # Gaussian value when x < mean
                height * torch.exp(-((x - mean) ** 2) / (2 * right_std ** 2))  # Gaussian value when x >= mean
            )
        else:
            # Return all-zero tensor if any standard deviation is 0 (avoid division by zero error)
            result = torch.zeros_like(x)

        return result


    def generate_peak_data(parameters, num_points=1024):
        """
        Generate synthetic data composed of multiple overlapping asymmetric Gaussian peaks.
        Parameters:
            parameters: Tensor containing parameters for each Gaussian peak, shape [num_peaks, 5]
                        Each row format: [?, left_std, right_std, mean, height]
            num_points: Number of points for generated data, default is global constant DATA_LENGTH
        Returns:
            x: Tensor of sampling positions
            total_data: Synthetic data tensor after overlapping all Gaussian peaks
        """
        # Create uniformly distributed sampling points
        x = torch.linspace(0, num_points, num_points)

        # Initialize total data as all-zero vector
        total_data = torch.zeros(num_points)

        # Iterate through parameters of each Gaussian peak
        for param in parameters:
            # Skip all-zero parameter cases (may indicate invalid peaks)
            if torch.all(param == 0):
                continue

            # Parse parameters: [_, left_std, right_std, mean, height]
            # The first parameter is unused (may be confidence or other auxiliary information)
            _, std1, std2, mean, height = param

            # Generate data for current Gaussian peak
            peak_data = gaussian(x, mean, std1, std2, height)

            # Overlap current peak to total data
            total_data += peak_data

        return total_data


    num_sample = 10000
    data = DataSet(num_samples=num_sample, three_channel=False, data_type='validation')
    file_name = f"results/results_traditional_method.txt"

    lam_list = [1e5, 1e6, 1e7, 1e8]
    p_list = [0.0001, 0.001, 0.01, 0.1]
    n_iter_list = [3, 5, 10]

    height_list = [0.005, 0.01, 0.02, 0.03, 0.05]
    prominence_list = [0.02, 0.05, 0.08, 0.1, 0.15]
    width_list = [3, 5, 10, 20, 30]
    outputs_all = []

    for i in range(num_sample):
        best_ap50_95 = -1e4
        best_ap50 = 0
        best_output = None
        print(i)
        inputs, target, inputs_without_noise = data[i]
        single_sample = data[i]
        # Wrap into a dataset object containing only this single sample
        single_data = SingleSampleDataset(single_sample)
        inputs = np.array(inputs.squeeze(0))
        for lam in lam_list:
            for p in p_list:
                for n in n_iter_list:
                    for height in height_list:
                        for prominence in prominence_list:
                            outputs = []
                            inputs_without_baseline, _ = als_baseline_correction(inputs, lam, p, n)
                            smoothed_sequence = savgol_filter(inputs_without_baseline, 5, 3, deriv=0)
                            # plt.plot(smoothed_sequence)
                            # plt.show()
                            lsf = LSF(mode='test', height=height, prominence=prominence)
                            output = lsf(smoothed_sequence)
                            outputs.append(output)
                            from analysis import AnAlysis

                            analysis = AnAlysis(nms=True, num_analysis=1,
                                                traditional_or_neural='traditional',
                                                show_result=False)
                            pre, rec, ap50, ap50_95, ap, rss = analysis(None, None, single_data, outputs)
                            # print(ap50,ap50_95,rss)
                            if ap50_95 > best_ap50_95:
                                best_ap50_95 = ap50_95
                                best_ap50 = ap50
                                best_output = output
        outputs_all.append(best_output)
        print(best_ap50, best_ap50_95)

    from analysis import AnAlysis

    analysis = AnAlysis(nms=True, num_analysis=num_sample, traditional_or_neural='traditional',
                        show_result=False)
    pre, rec, ap50, ap50_95, ap, rss = analysis(None, None, data, outputs_all)
    print("results:", pre, rec, ap50, ap50_95, ap, rss)

    with open(file_name, 'a') as file:
        file.write("als_peak_fitting\n")
        file.write(f"pre: {pre}, rec: {rec}\n")
        file.write(f"map50: {ap50}, map50_95: {ap50_95}\n")
        file.write(f"rss: {rss}\n")

    # List of polynomial orders (core tuning parameters, covering low→medium-high orders, avoid >15 to prevent overfitting)
    order_list = [5, 10, 12]

    # List of iteration counts (convergence gain is minimal after 100 iterations, covering low→medium-high iterations)
    n_iter_list = [20, 50, 100, 150]

    height_list = [0.005, 0.01, 0.05]
    prominence_list = [0.001, 0.005, 0.01, 0.05]

    outputs_all = []

    for i in range(num_sample):
        best_ap50_95 = -1e4
        best_ap50 = 0
        best_output = None
        print(i)
        inputs, target, inputs_without_noise = data[i]
        single_sample = data[i]
        # Wrap into a dataset object containing only this single sample
        single_data = SingleSampleDataset(single_sample)
        inputs = np.array(inputs.squeeze(0))
        for height in height_list:
            for prominence in prominence_list:
                outputs = []
                inputs_without_baseline = baseline_tools(inputs)
                smoothed_sequence = savgol_filter(inputs_without_baseline, 5, 3, deriv=0)
                lsf = LSF(mode='test', height=height, prominence=prominence)
                output = lsf(smoothed_sequence)
                outputs.append(output)

                outputs = torch.stack(outputs)

                from analysis import AnAlysis

                analysis = AnAlysis(nms=True, num_analysis=1,
                                    traditional_or_neural='traditional',
                                    show_result=False)
                pre, rec, ap50, ap50_95, ap, rss = analysis(None, None, single_data, outputs)
                # print(ap50,ap50_95,rss)
                if ap50_95 > best_ap50_95:
                    best_ap50_95 = ap50_95
                    best_ap50 = ap50
                    best_output = output
        outputs_all.append(best_output)
        print(best_ap50, best_ap50_95)

    from analysis import AnAlysis

    analysis = AnAlysis(nms=True, num_analysis=num_sample, traditional_or_neural='traditional',
                        show_result=False)
    pre, rec, ap50, ap50_95, ap, rss = analysis(None, None, data, outputs_all)
    print("results:", pre, rec, ap50, ap50_95, ap, rss)

    with open(file_name, 'a') as file:
        file.write("airPLS_peak_fitting\n")
        file.write(f"pre: {pre}, rec: {rec}\n")
        file.write(f"map50: {ap50}, map50_95: {ap50_95}\n")
        file.write(f"rss: {rss}\n")
