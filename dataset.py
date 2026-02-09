import torch
import numpy as np
import math
from torch.utils.data import Dataset
from config import *
import matplotlib.pyplot as plt
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------------------------------------#
#               Dataset Generation                 #
#           Symmetric & Asymmetric Gaussian Peaks  #
#               Noise and Baseline Drift           #
#               Generate Sequences and Labels      #
# -------------------------------------------------#

def set_seed(seed):
    """Set random seed to ensure experimental reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


class DataSet(Dataset):
    """Generate synthetic dataset containing multiple Gaussian peaks for training and evaluation of peak detection models"""

    def __init__(self, num_samples=10000, three_channel=False, data_type='predict'):
        """
        Initialize dataset generator

        Args:
            num_samples: Number of samples to generate
            three_channel: Whether to generate three-channel data (original data, first-order difference, second-order difference)
            data_type: Dataset type (train/test/validation/predict) for setting random seed
        """
        super(DataSet, self).__init__()
        self.num_samples = num_samples  # Number of samples
        self.three_channel = three_channel  # Whether to use three channels
        self.sequence_length = DATA_LENGTH  # Sequence length
        self.num_peaks_range = (1, PEAK_NUM_MAX)  # Range of peak counts per sample
        self.amplitude_range = (0.01, 0.9)  # Peak height range
        self.location_range = (0.05 * self.sequence_length, 0.95 * self.sequence_length)  # Peak position range
        self.std_range = (0.0015 * self.sequence_length, 0.04 * self.sequence_length)  # Standard deviation range
        self.noise_level = self.amplitude_range[0] / 20  # Noise level
        self.amplitude_max = 1  # Maximum sequence amplitude
        self.multiplier_range = (0.05, 0.2)  # Baseline drift multiplier range
        self.num_sigmoid = 10

        # Set random seed according to dataset type to ensure reproducibility
        if data_type == 'train':        set_seed(0)
        if data_type == 'test':         set_seed(1)
        if data_type == 'validation':   set_seed(2)

        self.multi_gaussian_params = self.generate_random_list()

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples

    def generate_random(self):
        """
        Generate random Gaussian peak parameters

        Returns:
            locations: List of peak positions
            amplitudes: List of peak heights
            left_stds: List of left standard deviations
            right_stds: List of right standard deviations
        """
        # Randomly determine maximum amplitude and maximum standard deviation
        amplitude_max = np.random.uniform(*self.amplitude_range)
        amplitude_range = (self.amplitude_range[0], amplitude_max)
        std_max = np.random.uniform(*self.std_range)
        std_min = self.std_range[0]

        # Determine number of peaks (avoid overcrowding)
        num_gaussian = np.random.randint(1, min(PEAK_NUM_MAX, math.floor((self.sequence_length * 0.9) / (6 * std_max))))

        # Randomly generate standard deviation pairs (symmetric or asymmetric)
        pairs = []
        choice = np.random.choice([1, 2, 3])
        if choice == 1:
            pairs.append((std_max, std_max))  # Symmetric peak
        elif choice == 2:
            pairs.append((std_max, np.random.uniform(max(std_min, std_max / 4), std_max)))  # Right-skewed peak
        else:
            pairs.append((np.random.uniform(max(std_min, std_max / 4), std_max), std_max))  # Left-skewed peak

        # Generate remaining peak parameters (part symmetric, part asymmetric): fix high<=0 issue
        if num_gaussian == 1:
            # Only 1 peak, no remaining peaks to add, set symmetric peak count to 0 directly
            equal_num_gaussian = 0
        else:
            # When peak count > 1, randomly generate symmetric peak count normally (range 0 ~ num_gaussian-1)
            equal_num_gaussian = np.random.randint(0, num_gaussian - 1)

        unequal_num_gaussian = num_gaussian - 1 - equal_num_gaussian

        # Generate asymmetric peak parameters
        for _ in range(unequal_num_gaussian):
            left_val = np.random.uniform(std_min, std_max)
            right_val = np.random.uniform(max(std_min, left_val / 4), min(std_max, left_val * 4))
            pairs.append((left_val, right_val))

        # Generate symmetric peak parameters
        for _ in range(equal_num_gaussian):
            equal_val = np.random.uniform(std_min, std_max)
            pairs.append((equal_val, equal_val))

        # Shuffle order
        np.random.shuffle(np.array(pairs))
        left_stds = np.array(pairs)[:, 0]
        right_stds = np.array(pairs)[:, 1]

        # Generate peak heights (ensure at least one peak reaches maximum amplitude)
        amplitudes = [np.random.uniform(*amplitude_range) for _ in range(num_gaussian - 1)]
        amplitudes.append(amplitude_max)

        # Generate non-overlapping peak positions
        locations = []
        for i in range(num_gaussian):
            while True:
                # Calculate candidate position range (avoid too close to sequence edges)
                location_range = (max(self.location_range[0], float(left_stds[i]) * 3),
                                  min(self.location_range[1], self.sequence_length - right_stds[i] * 3))
                location = np.random.uniform(*location_range)

                # Check overlap with existing peaks
                is_valid = True
                for j in range(i):
                    min_left_width = (left_stds[j] + left_stds[i]) * 3 / 2
                    min_right_width = (right_stds[j] + right_stds[i]) * 3 / 2
                    lower_bound = locations[j] - min_left_width
                    upper_bound = locations[j] + min_right_width
                    if lower_bound <= location <= upper_bound:
                        is_valid = False
                        break

                if is_valid:
                    locations.append(location)
                    break

        sigmoid_params = []
        for _ in range(self.num_sigmoid):
            multiplier = np.random.uniform(0.05, 0.95 - max(amplitudes))
            a = np.random.uniform(-20, 20)  # Slope
            b = np.random.uniform(-20, 20)  # Offset
            sigmoid_params.append((a, b, multiplier))
        sigmoid_params = np.array(sigmoid_params, dtype=np.float32)  # Convert to array for easy storage

        # Return complete parameters: peak parameters + noise + baseline parameters
        peak_params = (locations, amplitudes, left_stds, right_stds)

        # Add positive noise
        noise = np.random.normal(0, self.noise_level, self.sequence_length).astype(np.float32)
        noise = np.maximum(noise, 0)

        return peak_params, sigmoid_params, noise

    def apply_baseline_drift(self, sequence, sigmoid_params, peak_params):

        def sigmoid(x, a, b, multiplier):
            """Sigmoid function for generating smooth baseline drift"""
            return 1 / (1 + np.exp(-(x * a + b))) * multiplier

        locations, amplitudes, left_stds, right_stds = peak_params

        # Generate points on x-axis
        x = np.linspace(-1, 1, len(sequence))

        # Initialize baseline drift
        baseline_drift = np.zeros(len(sequence), dtype='float32')

        # Superimpose multiple sigmoid curves to generate complex baseline drift
        for i in range(self.num_sigmoid):
            a, b, multiplier = sigmoid_params[i]
            baseline_drift += sigmoid(x, a, b, multiplier) / self.num_sigmoid

        # Normalize and adjust amplitude of baseline drift
        baseline_drift = (baseline_drift / max(baseline_drift)) * (0.95 - max(amplitudes))

        return baseline_drift

    def generate_gaussian_sequence(self, gaussian_params):
        """
        Generate single Gaussian peak sequence

        Args:
            gaussian_params: Gaussian peak parameters (position, amplitude, left std, right std)

        Returns:
            gaussian: Single Gaussian peak sequence
        """
        location, amplitude, left_std, right_std = gaussian_params

        # Use different standard deviations for left and right sides to generate asymmetric Gaussian peak
        left_half = np.exp(-0.5 * ((np.arange(self.sequence_length) - location) / left_std) ** 2)
        right_half = np.exp(-0.5 * ((np.arange(self.sequence_length) - location) / right_std) ** 2)

        # Combine left and right parts
        gaussian = np.where((np.arange(self.sequence_length)) < location, left_half, right_half)
        gaussian *= amplitude

        return gaussian

    def generate_multi_gaussian_sequence(self, multi_gaussian_params):
        """
        Generate sequence with multiple superimposed Gaussian peaks, add noise and baseline drift

        Args:
            multi_gaussian_params: Parameters of multiple Gaussian peaks

        Returns:
            sequence: Sequence with noise and baseline drift added
            sequence_without_noise_baseline: Sequence without noise and baseline drift
        """
        # Initialize sequence
        sequence = np.zeros(self.sequence_length)

        # Superimpose all Gaussian peaks
        for gaussian_param in zip(*multi_gaussian_params):
            sequence += self.generate_gaussian_sequence(gaussian_param)

        # Save sequence without noise and baseline drift (for evaluation)
        sequence_without_noise_baseline = np.copy(sequence)

        return sequence, sequence_without_noise_baseline

    def generate_sequence_target(self, multi_gaussian_params):
        """
        Generate target labels for sequence

        Args:
            multi_gaussian_params: Parameters of multiple Gaussian peaks

        Returns:
            targets: Target labels containing start position, end position, peak position and height for each peak
        """
        # Initialize target labels (supports up to PEAK_NUM_MAX peaks)
        targets = np.zeros((self.num_peaks_range[1], 4), dtype=float)

        # Generate labels for each peak
        for i, (location, amplitude, left_std, right_std) in enumerate(zip(*multi_gaussian_params)):
            x_start = location - 3 * left_std  # Start position
            x_end = location + 3 * right_std  # End position
            x_peak = location  # Peak position
            height = amplitude / self.amplitude_max  # Normalized peak height

            targets[i][0:4] = [x_start, x_end, x_peak, height]

        return torch.tensor(targets)

    def generate_random_list(self):
        multi_gaussian_params = []

        for idx in range(self.num_samples):
            multi_gaussian_param = self.generate_random()
            multi_gaussian_params.append(multi_gaussian_param)

        return multi_gaussian_params

    def __getitem__(self, index):
        """
        Get single sample

        Args:
            index: Sample index

        Returns:
            sequence: Input sequence (may be three-channel: original data, first-order difference, second-order difference)
            targets: Target labels
            sequence_without_noise_baseline: Sequence without noise and baseline drift
        """
        # Generate random Gaussian peak parameters
        multi_gaussian_param, sigmoid_param, noise = self.multi_gaussian_params[index]

        # Generate sequence (add noise and baseline drift)
        sequence, sequence_without_noise_baseline = self.generate_multi_gaussian_sequence(multi_gaussian_param)

        sequence += noise

        # Add baseline drift
        sequence += self.apply_baseline_drift(sequence, sigmoid_param, multi_gaussian_param)

        # Generate target labels
        targets = self.generate_sequence_target(multi_gaussian_param)

        # Convert to PyTorch tensor and adjust shape
        sequence = sequence.reshape(1, len(sequence))
        sequence = torch.tensor(sequence).float()

        # Calculate first-order and second-order differences
        dx = sequence[:, 1:] - sequence[:, :-1]
        dx_pad = torch.cat((torch.zeros_like(dx[:, 0].unsqueeze(-1)), dx), dim=-1)

        ddx = dx[:, 1:] - dx[:, :-1]
        ddx_pad = torch.cat((torch.zeros_like(ddx[:, 0].unsqueeze(-1)), ddx, torch.zeros_like(ddx[:, 0].unsqueeze(-1))),
                            dim=-1)

        # Whether to use three-channel data
        if self.three_channel:
            sequence = torch.cat((sequence, dx_pad, ddx_pad), dim=0)

        return sequence, targets, torch.tensor(sequence_without_noise_baseline)


if __name__ == "__main__":
    """Test dataset generator"""
    data = DataSet(num_samples=10, data_type='train')

    # Visualize generated samples
    for i in range(3):
        sequence, target, _ = data[i]
        sequence = sequence.squeeze(0)

        # print(f"Sequence shape: {sequence.shape}")
        print("Label_format: [left_edge, right_edge, peak_location, peak_height]")
        print(f"Label:\n{target}")

        # Plot sequence
        plt.figure(figsize=(16, 3))
        plt.plot(sequence)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Multi-Gaussian Sequence')
        plt.ylim(0, 1)
        plt.show()